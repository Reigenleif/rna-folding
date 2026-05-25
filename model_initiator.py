from __future__ import annotations

import math
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

DATA_ROOT = Path("data")
DEFAULT_DRFOLD_DIR = Path("DRFold2-model99")
DEFAULT_WEIGHTS = "model_19"


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_keywords: tuple[str, ...] = ()
    train_bias: bool = False


@dataclass
class ModelBundle:
    model: nn.Module
    data_module: object
    base_std: np.ndarray
    preprocessor: "SequencePreprocessor"
    coord_scaler: "CoordStandardScaler"
    device: str
    drfold_dir: Path
    weights_path: Path
    lora_config: LoRAConfig | None


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
        outputs: list[dict[str, object]] = []
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


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout)

        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def pick_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu.")
        return "cpu"
    return requested


def resolve_weight_path(
    drfold_dir: Path,
    weights: str,
    checkpoint_dir: Path = Path("/checkpoints/drfold2"),
) -> Path:
    candidates = [checkpoint_dir / weights, drfold_dir / weights]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Checkpoint not found. Checked: " + ", ".join(str(p) for p in candidates))


def _should_wrap(full_name: str, module: nn.Module, config: LoRAConfig) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if not config.target_keywords:
        return True
    lowered = full_name.lower()
    return any(keyword.lower() in lowered for keyword in config.target_keywords)


def apply_lora(model: nn.Module, config: LoRAConfig) -> int:
    wrapped = 0

    def _inject(module: nn.Module, prefix: str = "") -> None:
        nonlocal wrapped
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LoRALinear):
                continue
            if _should_wrap(full_name, child, config):
                setattr(module, name, LoRALinear(child, config.rank, config.alpha, config.dropout))
                wrapped += 1
            else:
                _inject(child, full_name)

    _inject(model)
    return wrapped


def freeze_base_model(model: nn.Module, train_bias: bool = False) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            for param in module.lora_A.parameters():
                param.requires_grad = True
            for param in module.lora_B.parameters():
                param.requires_grad = True
            if train_bias and module.base_layer.bias is not None:
                module.base_layer.bias.requires_grad = True


def import_drfold_modules(drfold_dir: Path, device: str):
    sys.argv = [sys.argv[0], device]
    if str(drfold_dir) not in sys.path:
        sys.path.insert(0, str(drfold_dir))
    import data  # noqa: E402
    import EvoMSA2XYZ  # noqa: E402

    return data, EvoMSA2XYZ


def create_model_bundle(
    drfold_dir: Path = DEFAULT_DRFOLD_DIR,
    weights: str = DEFAULT_WEIGHTS,
    device: str = "cpu",
    lora_config: LoRAConfig | None = None,
) -> ModelBundle:
    device = pick_device(device)
    data_module, EvoMSA2XYZ = import_drfold_modules(drfold_dir, device)
    base_std = np.load(drfold_dir / "base.npy")
    preprocessor = SequencePreprocessor(base_std)

    msa_dim = 6 + 1
    m_dim = s_dim = z_dim = 64
    n_ensemble, n_cycle = 3, 8
    model = EvoMSA2XYZ.MSA2XYZ(msa_dim - 1, msa_dim, n_ensemble, n_cycle, m_dim, s_dim, z_dim)

    weight_path = resolve_weight_path(drfold_dir, weights)
    try:
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weight_path, map_location="cpu")
    print(f"Params before lora: {sum(p.numel() for p in model.parameters())}")
    model.load_state_dict(state, strict=False)

    wrapped = 0
    if lora_config is not None:
        wrapped = apply_lora(model, lora_config)
        freeze_base_model(model, train_bias=lora_config.train_bias)
    model.to(device)
    if lora_config is not None:
        print(f"Injected LoRA into {wrapped} linear layers.")
        print(f"Params after lora: {sum(p.numel() for p in model.parameters())}")
    
    coord_scaler = CoordStandardScaler(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
    
    
    return ModelBundle(
        model=model,
        data_module=data_module,
        base_std=base_std,
        preprocessor=preprocessor,
        coord_scaler=coord_scaler,
        device=device,
        drfold_dir=drfold_dir,
        weights_path=weight_path,
        lora_config=lora_config,
    )


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
    model: nn.Module,
    seq: str,
    preprocessor: SequencePreprocessor,
    data_module: object,
    device: str,
    atom_index: int = 1,
) -> np.ndarray:
    msa, base_x, seq_idx = preprocessor.build_features(seq, device)
    with torch.no_grad():
        ret = model.pred(msa, seq_idx, None, base_x, np.array(list(seq)))
    coor = np.asarray(ret["coor"])
    if coor.ndim == 3:
        return coor[:, atom_index, :]
    return coor


def forward_train(
    model: nn.Module,
    msa: torch.Tensor,
    seq_idx: torch.Tensor,
    base_x: torch.Tensor,
    alphas: np.ndarray,
) -> torch.Tensor:
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
    scaler: CoordStandardScaler | None = None,
) -> torch.Tensor | None:
    mask = valid_mask(ref)
    if mask.sum() == 0:
        return None
    pred_t = pred[mask]
    ref_t = torch.as_tensor(ref[mask], device=pred.device, dtype=pred.dtype)
    if scaler is not None:
        mean_t = torch.as_tensor(scaler.mean_, device=pred.device, dtype=pred.dtype)
        scale_t = torch.as_tensor(scaler.scale_, device=pred.device, dtype=pred.dtype)
        pred_t = (pred_t - mean_t) / scale_t
        ref_t = (ref_t - mean_t) / scale_t
    return torch.mean((pred_t - ref_t) ** 2)


def evaluate_items(
    model: nn.Module,
    items: Iterable[tuple[str, str, np.ndarray]],
    preprocessor: SequencePreprocessor,
    data_module: object,
    device: str,
    atom_index: int,
) -> dict[str, float]:
    results = []
    for target_id, seq, ref_coords in items:
        pred = predict_coords(model, seq, preprocessor, data_module, device, atom_index)
        metrics = compute_metrics(pred, ref_coords)
        metrics.update({"target_id": target_id, "length": len(seq)})
        results.append(metrics)

    if not results:
        return {}

    df = pd.DataFrame(results)
    summary = df[["rmse", "tm_score", "rmse_kabsch", "tm_score_kabsch"]].mean(numeric_only=True)
    return {key: float(value) for key, value in summary.to_dict().items()}


def save_artifact(
    artifact_path: Path,
    bundle: ModelBundle,
    history: list[dict[str, float]],
) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_state_dict": bundle.model.state_dict(),
        "base_weights": str(bundle.weights_path),
        "drfold_dir": str(bundle.drfold_dir),
        "lora_config": asdict(bundle.lora_config),
        "coord_scaler": bundle.coord_scaler,
        "preprocessor": bundle.preprocessor,
        "history": history,
    }
    with artifact_path.open("wb") as f:
        pickle.dump(artifact, f)


def load_artifact(artifact_path: Path) -> dict:
    with artifact_path.open("rb") as f:
        return pickle.load(f)


def load_bundle_from_artifact(
    artifact_path: Path,
    device: str = "cpu",
) -> ModelBundle:
    artifact = load_artifact(artifact_path)
    lora_cfg = LoRAConfig(**artifact["lora_config"])
    bundle = create_model_bundle(
        drfold_dir=Path(artifact["drfold_dir"]),
        weights=Path(artifact["base_weights"]).name,
        device=device,
        lora_config=lora_cfg,
    )
    bundle.model.load_state_dict(artifact["model_state_dict"], strict=False)
    bundle.coord_scaler = artifact["coord_scaler"]
    bundle.preprocessor = artifact["preprocessor"]
    return bundle