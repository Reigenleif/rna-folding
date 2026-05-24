#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

DATA_ROOT = "data"
ATOM_INDEX = 1  # 0=P, 1=C4', 2=N1/N9
MAX_SEQ_LEN = 12
DEFAULT_CHECKPOINT_DIR = Path("/checkpoints/drfold2")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu.")
        return "cpu"
    return requested


def load_sequences(split: str, data_root: Path, num_to_process: int | None) -> pd.DataFrame:
    seq_path = data_root / ("train_sequences.csv" if split == "train" else "validation_sequences.csv")
    seqs = pd.read_csv(seq_path, usecols=["target_id", "sequence"])
    seqs = seqs.sort_values(by="sequence", key=lambda x: x.str.len())
    if num_to_process is None:
        return seqs
    return seqs.head(num_to_process)


def load_labels(label_path: Path, target_ids: list[str]) -> dict[str, np.ndarray]:
    target_set = set(target_ids)
    cols = ["ID", "resid", "x_1", "y_1", "z_1"]
    chunks = pd.read_csv(label_path, usecols=cols, chunksize=200000)
    kept = []
    for chunk in chunks:
        chunk["target_id"] = chunk["ID"].str.rsplit("_", n=1).str[0]
        filtered = chunk[chunk["target_id"].isin(target_set)]
        if not filtered.empty:
            kept.append(filtered)
    if not kept:
        return {}
    labels = pd.concat(kept, ignore_index=True)
    labels = labels.sort_values(["target_id", "resid"])
    coords = {}
    for target_id, group in labels.groupby("target_id"):
        coords[target_id] = group[["x_1", "y_1", "z_1"]].to_numpy()
    return coords


def filter_sequences_by_max_len(seqs: pd.DataFrame, max_len: int) -> pd.DataFrame:
    return seqs[seqs["sequence"].str.len() <= max_len]


def count_label_rows(label_map: dict[str, np.ndarray], target_ids: list[str]) -> int:
    total = 0
    for target_id in target_ids:
        coords = label_map.get(target_id)
        if coords is not None:
            total += coords.shape[0]
    return total


def build_features(seq: str, base_std: np.ndarray, data_module, device: str):
    aa_type = data_module.parse_seq(seq)
    base = data_module.Get_base(seq, base_std)
    seq_idx = np.arange(len(seq)) + 1

    msa = aa_type[None, :]
    msa = torch.from_numpy(msa).to(device)
    msa = torch.cat([msa, msa], 0)
    msa = F.one_hot(msa.long(), 6).float()

    base_x = torch.from_numpy(base).float().to(device)
    seq_idx = torch.from_numpy(seq_idx).long().to(device)
    return msa, base_x, seq_idx


class SequencePreprocessor:
    def __init__(self, base_std: np.ndarray) -> None:
        self.base_std = np.asarray(base_std)

    def parse_seq(self, seq: str) -> np.ndarray:
        seqnpy = np.zeros(len(seq))
        seq1 = np.array(list(seq.upper()))
        seqnpy[seq1 == "A"] = 1
        seqnpy[seq1 == "G"] = 2
        seqnpy[seq1 == "C"] = 3
        seqnpy[(seq1 == "U") | (seq1 == "T")] = 4
        return seqnpy

    def get_base(self, seq: str) -> np.ndarray:
        basenpy = np.zeros([len(seq), 3, 3])
        seq1 = np.array(list(seq.upper()))
        basenpy[seq1 == "A"] = self.base_std[0]
        basenpy[seq1 == "G"] = self.base_std[1]
        basenpy[seq1 == "C"] = self.base_std[2]
        basenpy[(seq1 == "U") | (seq1 == "T")] = self.base_std[3]
        return basenpy

    def build_features(self, seq: str, device: str):
        aa_type = self.parse_seq(seq)
        base = self.get_base(seq)
        seq_idx = np.arange(len(seq)) + 1

        msa = aa_type[None, :]
        msa = torch.from_numpy(msa).to(device)
        msa = torch.cat([msa, msa], 0)
        msa = F.one_hot(msa.long(), 6).float()

        base_x = torch.from_numpy(base).float().to(device)
        seq_idx = torch.from_numpy(seq_idx).long().to(device)
        return msa, base_x, seq_idx

    def transform_df(self, df: pd.DataFrame, device: str) -> list[dict[str, object]]:
        if "sequence" not in df.columns:
            raise ValueError("Expected a 'sequence' column in the input DataFrame.")
        outputs = []
        for row in df.itertuples(index=False):
            seq = row.sequence
            msa, base_x, seq_idx = self.build_features(seq, device)
            outputs.append(
                {
                    "target_id": getattr(row, "target_id", None),
                    "sequence": seq,
                    "msa": msa,
                    "base_x": base_x,
                    "seq_idx": seq_idx,
                    "alphas": np.array(list(seq)),
                }
            )
        return outputs


class CoordStandardScaler:
    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean_ = np.asarray(mean, dtype=np.float32)
        self.scale_ = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, coords: np.ndarray) -> "CoordStandardScaler":
        mean = coords.mean(axis=0)
        scale = coords.std(axis=0)
        scale = np.where(scale == 0, 1.0, scale)
        return cls(mean, scale)

    def transform(self, coords: np.ndarray) -> np.ndarray:
        return (coords - self.mean_) / self.scale_

    def inverse_transform(self, coords: np.ndarray) -> np.ndarray:
        return coords * self.scale_ + self.mean_


def resolve_weight_path(
    drfold_dir: Path,
    weights: str,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
) -> Path:
    candidates = [checkpoint_dir / weights, drfold_dir / weights]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Checkpoint not found. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def freeze_all_except_rna_transformer(model) -> None:
    for param in model.parameters():
        param.requires_grad = False

    # Keep RNA Transformer (Evoformer) blocks trainable.
    if not hasattr(model, "msaxyzone") or not hasattr(model.msaxyzone, "evmodel"):
        raise AttributeError("Expected model.msaxyzone.evmodel for RNA Transformer blocks.")
    for param in model.msaxyzone.evmodel.parameters():
        param.requires_grad = True


def load_model(drfold_dir: Path, weights: str, device: str):
    sys.argv = [sys.argv[0], device]
    sys.path.insert(0, str(drfold_dir))
    import data  # noqa: E402
    import EvoMSA2XYZ  # noqa: E402, F401, type: ignore[reportMissingImports]

    base_std = np.load(drfold_dir / "base.npy")
    msa_dim = 6 + 1
    m_dim = s_dim = z_dim = 64
    n_ensemble, n_cycle = 3, 8

    model = EvoMSA2XYZ.MSA2XYZ(
        msa_dim - 1, msa_dim, n_ensemble, n_cycle, m_dim, s_dim, z_dim
    )
    # freeze_all_except_rna_transformer(model)
    
    
    # Architecture modifications
    


    weight_path = resolve_weight_path(drfold_dir, weights)
    try:
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    return model, base_std, data


def valid_mask(coords: np.ndarray) -> np.ndarray:
    finite = np.isfinite(coords).all(axis=1)
    within = np.abs(coords).max(axis=1) < 1e17
    return finite & within


def d0_for_length(length: int) -> float:
    if length >= 30:
        return 0.6 * (length - 0.5) ** 0.5 - 2.5
    if length < 12:
        return 0.3
    if length < 16:
        return 0.4
    if length < 20:
        return 0.5
    if length < 24:
        return 0.6
    return 0.7


def tm_score(pred: np.ndarray, ref: np.ndarray) -> float:
    length = len(ref)
    if length == 0:
        return float("nan")
    d0 = d0_for_length(length)
    dist = np.linalg.norm(pred - ref, axis=1)
    score = np.sum(1.0 / (1.0 + (dist / d0) ** 2)) / length
    return float(score)


def kabsch_align(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    pred_center = pred.mean(axis=0)
    ref_center = ref.mean(axis=0)
    pred_c = pred - pred_center
    ref_c = ref - ref_center
    cov = pred_c.T @ ref_c
    u, _, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    return pred_c @ r + ref_center


def compute_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    length = min(len(pred), len(ref))
    pred = pred[:length]
    ref = ref[:length]
    mask = valid_mask(ref) & valid_mask(pred)
    pred = pred[mask]
    ref = ref[mask]
    if len(ref) == 0:
        return {
            "rmse": float("nan"),
            "tm_score": float("nan"),
            "rmse_kabsch": float("nan"),
            "tm_score_kabsch": float("nan"),
        }

    rmse = float(np.sqrt(np.mean((pred - ref) ** 2)))
    tm = tm_score(pred, ref)

    if len(ref) < 3:
        return {
            "rmse": rmse,
            "tm_score": tm,
            "rmse_kabsch": float("nan"),
            "tm_score_kabsch": float("nan"),
        }

    pred_k = kabsch_align(pred, ref)
    rmse_k = float(np.sqrt(np.mean((pred_k - ref) ** 2)))
    tm_k = tm_score(pred_k, ref)
    return {
        "rmse": rmse,
        "tm_score": tm,
        "rmse_kabsch": rmse_k,
        "tm_score_kabsch": tm_k,
    }


def predict_coords(
    model,
    seq: str,
    base_std: np.ndarray,
    data_module,
    device: str,
    atom_index: int,
) -> np.ndarray:
    msa, base_x, seq_idx = build_features(seq, base_std, data_module, device)
    with torch.no_grad():
        ret = model.pred(msa, seq_idx, None, base_x, np.array(list(seq)))
    coor = np.asarray(ret["coor"])
    if coor.ndim == 3:
        return coor[:, atom_index, :]
    return coor


def fit_coord_scaler(items: list[tuple[str, str, np.ndarray]]) -> CoordStandardScaler:
    coords_list = []
    for _, _, coords in items:
        mask = valid_mask(coords)
        if mask.any():
            coords_list.append(coords[mask])
    if not coords_list:
        return CoordStandardScaler(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
    all_coords = np.vstack(coords_list)
    return CoordStandardScaler.fit(all_coords)


def forward_train(model, msa, seq_idx, base_x, alphas):
    length = msa.shape[1]
    m1_pre = 0
    z_pre = 0
    x_pre = torch.zeros(length, 3, 3, device=msa.device)
    previous_dis = torch.zeros(length, length, model.dis_dim, device=msa.device)
    previous_dis[..., 0] = 1
    previous_hb = torch.zeros(length, length, 6, device=msa.device)
    previous_hb[..., 0] = 1

    x = None
    for cycle in range(model.N_cycle):
        m1, z, s = model.msaxyzone.pred(
            msa, seq_idx, None, m1_pre, z_pre, x_pre, cycle, alphas, previous_dis, previous_hb
        )
        x, _, _, _ = model.structurenet.pred(s, z, base_x)
        pred_dis = F.softmax(model.ndis_predor(z), dim=-1)
        pred_hb = torch.sigmoid(model.hb_predor(z))

        m1_pre = m1.detach()
        z_pre = z.detach()
        x_pre = x.detach()
        previous_dis = pred_dis.detach()
        previous_hb = pred_hb.detach()

    return x


def masked_mse(
    pred: torch.Tensor,
    ref: np.ndarray,
    mean_t: torch.Tensor | None = None,
    scale_t: torch.Tensor | None = None,
) -> torch.Tensor | None:
    mask = valid_mask(ref)
    if mask.sum() == 0:
        return None
    ref_t = torch.as_tensor(ref[mask], device=pred.device, dtype=pred.dtype)
    pred_t = pred[mask]
    if mean_t is not None and scale_t is not None:
        pred_t = (pred_t - mean_t) / scale_t
        ref_t = (ref_t - mean_t) / scale_t
    return torch.mean((pred_t - ref_t) ** 2)


def evaluate_items(
    model,
    items: list[tuple[str, str, np.ndarray]],
    base_std: np.ndarray,
    data_module,
    device: str,
    atom_index: int,
) -> dict[str, float]:
    if not items:
        return {}
    model.eval()
    results = []
    for target_id, seq, ref_coords in items:
        pred = predict_coords(model, seq, base_std, data_module, device, atom_index)
        metrics = compute_metrics(pred, ref_coords)
        metrics.update({"target_id": target_id, "length": len(seq)})
        results.append(metrics)
    df = pd.DataFrame(results)
    summary = df[["rmse", "tm_score", "rmse_kabsch", "tm_score_kabsch"]].mean(
        numeric_only=True
    )
    return {
        "rmse": float(summary["rmse"]),
        "tm_score": float(summary["tm_score"]),
        "rmse_kabsch": float(summary["rmse_kabsch"]),
        "tm_score_kabsch": float(summary["tm_score_kabsch"]),
    }


def train_model(
    model,
    items: list[tuple[str, str, np.ndarray]],
    base_std: np.ndarray,
    data_module,
    device: str,
    epochs: int,
    lr: float,
    atom_index: int,
    coord_scaler: CoordStandardScaler | None,
    eval_items: list[tuple[str, str, np.ndarray]] | None,
    metrics_out: Path | None,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
    )
    model.train()
    mean_t = None
    scale_t = None
    if coord_scaler is not None:
        mean_t = torch.as_tensor(coord_scaler.mean_, device=device)
        scale_t = torch.as_tensor(coord_scaler.scale_, device=device)

    metrics_log: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        random.shuffle(items)
        total_loss = 0.0
        count = 0
        progress = tqdm(items, desc=f"Epoch {epoch}/{epochs}", unit="seq")
        for _, seq, ref_coords in progress:
            msa, base_x, seq_idx = build_features(seq, base_std, data_module, device)
            optimizer.zero_grad()
            pred = forward_train(model, msa, seq_idx, base_x, np.array(list(seq)))
            pred_atom = pred[:, atom_index, :]
            loss = masked_mse(pred_atom, ref_coords, mean_t=mean_t, scale_t=scale_t)
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            count += 1
            progress.set_postfix(loss=f"{total_loss / count:.6f}")
        avg_loss = total_loss / max(count, 1)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.6f}")
        if eval_items:
            eval_summary = evaluate_items(
                model, eval_items, base_std, data_module, device, atom_index
            )
            eval_summary.update({"epoch": epoch, "train_loss": avg_loss})
            metrics_log.append(eval_summary)
            if metrics_out is not None:
                pd.DataFrame(metrics_log).to_csv(metrics_out, index=False)
        else:
            metrics_log.append({"epoch": epoch, "train_loss": avg_loss})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.train()

    return metrics_log


def save_checkpoint(model, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)


def save_artifacts(
    model,
    preprocessor: SequencePreprocessor,
    coord_scaler: CoordStandardScaler,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_state_dict": model.state_dict(),
        "preprocessor": preprocessor,
        "coord_scaler": coord_scaler,
    }
    with out_path.open("wb") as f:
        pickle.dump(artifact, f)


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Fine-tune DRFold2-model99")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--drfold-dir", default="DRFold2-model99")
    parser.add_argument("--weights", default="model_19")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=default_device)
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--atom-index", type=int, default=ATOM_INDEX)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--out", default="finetuned_model.pt")
    parser.add_argument("--artifact-out", default="finetune_artifacts.pkl")
    parser.add_argument("--metrics-out", default="training_metrics.csv")
    parser.add_argument("--eval-split", choices=["train", "val", "none"], default="val")
    parser.add_argument("--eval-num", type=int, default=None)
    parser.add_argument("--eval-max-seq-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    data_root = Path(DATA_ROOT)
    seqs = load_sequences(args.split, data_root, args.num)
    seq_count_all = len(seqs)
    label_path = data_root / ("train_labels.csv" if args.split == "train" else "validation_labels.csv")
    label_map = load_labels(label_path, seqs["target_id"].tolist())
    label_count_all = count_label_rows(label_map, seqs["target_id"].tolist())
    print(
        "Before max_len="
        f"{args.max_seq_len} -> sequences: {seq_count_all}, labels: {label_count_all}"
    )

    seqs = filter_sequences_by_max_len(seqs, args.max_seq_len)
    seq_count_cut = len(seqs)
    label_count_cut = count_label_rows(label_map, seqs["target_id"].tolist())
    print(
        "After max_len="
        f"{args.max_seq_len} -> sequences: {seq_count_cut}, labels: {label_count_cut}"
    )

    drfold_dir = Path(args.drfold_dir)
    model, base_std, data_module = load_model(drfold_dir, args.weights, device)

    items = []
    for row in seqs.itertuples(index=False):
        coords = label_map.get(row.target_id)
        if coords is None:
            continue
        items.append((row.target_id, row.sequence, coords))

    if not items:
        print("No training items found.")
        return

    coord_scaler = fit_coord_scaler(items)
    preprocessor = SequencePreprocessor(base_std)

    eval_items: list[tuple[str, str, np.ndarray]] | None = None
    if args.eval_split != "none":
        eval_max_len = args.eval_max_seq_len
        if eval_max_len is None:
            eval_max_len = args.max_seq_len
        eval_seqs = load_sequences(args.eval_split, data_root, args.eval_num)
        eval_seqs = filter_sequences_by_max_len(eval_seqs, eval_max_len)
        eval_label_path = data_root / (
            "train_labels.csv" if args.eval_split == "train" else "validation_labels.csv"
        )
        eval_label_map = load_labels(eval_label_path, eval_seqs["target_id"].tolist())
        eval_items = []
        for row in eval_seqs.itertuples(index=False):
            coords = eval_label_map.get(row.target_id)
            if coords is None:
                continue
            eval_items.append((row.target_id, row.sequence, coords))

    train_model(
        model,
        items,
        base_std,
        data_module,
        device,
        args.epochs,
        args.lr,
        args.atom_index,
        coord_scaler,
        eval_items,
        Path(args.metrics_out) if args.metrics_out else None,
    )
    save_checkpoint(model, Path(args.out))
    save_artifacts(model, preprocessor, coord_scaler, Path(args.artifact_out))
    print(f"Saved: {args.out}")
    print(f"Saved artifacts: {args.artifact_out}")


if __name__ == "__main__":
    main()
