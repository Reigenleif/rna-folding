#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

DATA_ROOT = "data"
NUM_TO_PROCESS = 50
MAX_LEN = 50
ATOM_INDEX = 1  # 0=P, 1=C4', 2=N1/N9
RESULTS_DIR = Path("results")
DEFAULT_FINETUNED_WEIGHTS = RESULTS_DIR / "finetuned_model.pt"
DEFAULT_RESULTS_OUT = RESULTS_DIR / "evaluation_finetuned_results.csv"
DEFAULT_SUMMARY_OUT = RESULTS_DIR / "evaluation_finetuned_summary.csv"


def pick_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu.")
        return "cpu"
    return requested


def load_sequences(
    split: str,
    data_root: Path,
    num_to_process: int | None,
    max_len: int | None,
) -> pd.DataFrame:
    seq_path = data_root / (
        "train_sequences.csv" if split == "train" else "validation_sequences.csv"
    )
    seqs = pd.read_csv(seq_path, usecols=["target_id", "sequence"])
    if max_len is not None:
        seqs = seqs[seqs["sequence"].str.len() <= max_len]
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


def load_modified_model_module(drfold_dir: Path, device: str):
    module_path = Path(__file__).resolve().with_name("drfold-mod.py")
    sys.argv = [sys.argv[0], device]
    if str(drfold_dir) not in sys.path:
        sys.path.insert(0, str(drfold_dir))
    spec = importlib.util.spec_from_file_location("drfold_mod", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load modified model module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_model(drfold_dir: Path, weights: Path, device: str):
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if str(drfold_dir) not in sys.path:
        sys.path.insert(0, str(drfold_dir))
    import data  # noqa: E402

    modified_module = load_modified_model_module(drfold_dir, device)

    base_std = np.load(drfold_dir / "base.npy")
    msa_dim = 6 + 1
    m_dim = s_dim = z_dim = 64
    n_ensemble, n_cycle = 3, 8

    model = modified_module.MSA2XYZ(
        msa_dim - 1, msa_dim, n_ensemble, n_cycle, m_dim, s_dim, z_dim
    )
    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, base_std, data


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


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned DRFold2-model99")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--drfold-dir", default="DRFold2-model99")
    parser.add_argument("--weights", default=DEFAULT_FINETUNED_WEIGHTS)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=default_device)
    parser.add_argument("--num", type=int, default=None, help="Override NUM_TO_PROCESS")
    parser.add_argument("--max-seq-len", type=int, default=MAX_LEN)
    parser.add_argument("--atom-index", type=int, default=ATOM_INDEX)
    parser.add_argument("--out", default=str(DEFAULT_RESULTS_OUT))
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_OUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    num_to_process = args.num if args.num is not None else NUM_TO_PROCESS

    data_root = Path(DATA_ROOT)
    seqs = load_sequences(args.split, data_root, num_to_process, args.max_seq_len)
    label_path = data_root / (
        "train_labels.csv" if args.split == "train" else "validation_labels.csv"
    )
    label_map = load_labels(label_path, seqs["target_id"].tolist())

    drfold_dir = Path(args.drfold_dir)
    weight_path = Path(args.weights)
    model, base_std, data_module = load_model(drfold_dir, weight_path, device)

    results = []
    for row in tqdm(seqs.itertuples(index=False), total=len(seqs), desc="Evaluating"):
        target_id = row.target_id
        seq = row.sequence
        ref = label_map.get(target_id)
        if ref is None:
            continue
        pred = predict_coords(model, seq, base_std, data_module, device, args.atom_index)
        metrics = compute_metrics(pred, ref)
        metrics.update({"target_id": target_id, "length": len(seq)})
        results.append(metrics)

    if not results:
        print("No targets evaluated.")
        return

    df = pd.DataFrame(results)
    summary = df[["rmse", "tm_score", "rmse_kabsch", "tm_score_kabsch"]].mean(
        numeric_only=True
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_frame().T.to_csv(summary_path, index=False)
    print(f"Saved detailed results: {out_path}")
    print(f"Saved summary: {summary_path}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
