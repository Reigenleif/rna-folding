#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model_initiator import (
    CoordStandardScaler,
    LoRAConfig,
    SequencePreprocessor,
    create_model_bundle,
    evaluate_items,
    forward_train,
    masked_mse,
    save_artifact,
)

DATA_ROOT = Path("data")
DEFAULT_OUT = Path("results/finetuned_lora_artifact.pkl")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def build_items(seqs: pd.DataFrame, label_map: dict[str, np.ndarray]) -> list[tuple[str, str, np.ndarray]]:
    items = []
    for row in seqs.itertuples(index=False):
        coords = label_map.get(row.target_id)
        if coords is None:
            continue
        items.append((row.target_id, row.sequence, coords))
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DRFold2 with LoRA/PEFT")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--eval-split", choices=["train", "val", "none"], default="val")
    parser.add_argument("--drfold-dir", default="DRFold2-model99")
    parser.add_argument("--weights", default="model_19")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--eval-num", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=32)
    parser.add_argument("--eval-max-seq-len", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--atom-index", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=2)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--artifact-out", default=str(DEFAULT_OUT))
    parser.add_argument("--metrics-out", default="results/drfold2_lora_training_metrics.csv")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = DATA_ROOT
    seqs = load_sequences(args.split, data_root, args.num)
    seqs = filter_sequences_by_max_len(seqs, args.max_seq_len)
    label_path = data_root / ("train_labels.csv" if args.split == "train" else "validation_labels.csv")
    label_map = load_labels(label_path, seqs["target_id"].tolist())
    train_items = build_items(seqs, label_map)
    if not train_items:
        print("No training items found.")
        return

    coord_scaler = CoordStandardScaler.fit(np.vstack([coords for _, _, coords in train_items]))
    lora_config = LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
    bundle = create_model_bundle(
        drfold_dir=Path(args.drfold_dir),
        weights=args.weights,
        device=args.device,
        lora_config=lora_config,
    )
    bundle.coord_scaler = coord_scaler

    eval_items = None
    if args.eval_split != "none":
        eval_seq = load_sequences(args.eval_split, data_root, args.eval_num)
        eval_max_len = args.eval_max_seq_len if args.eval_max_seq_len is not None else args.max_seq_len
        eval_seq = filter_sequences_by_max_len(eval_seq, eval_max_len)
        eval_label_path = data_root / ("train_labels.csv" if args.eval_split == "train" else "validation_labels.csv")
        eval_label_map = load_labels(eval_label_path, eval_seq["target_id"].tolist())
        eval_items = build_items(eval_seq, eval_label_map)

    optimizer = torch.optim.Adam((p for p in bundle.model.parameters() if p.requires_grad), lr=args.lr)
    metrics_path = Path(args.metrics_out) if args.metrics_out else None
    history: list[dict[str, float]] = []

    mean_t = torch.as_tensor(coord_scaler.mean_, device=bundle.device)
    scale_t = torch.as_tensor(coord_scaler.scale_, device=bundle.device)

    for epoch in range(1, args.epochs + 1):
        bundle.model.train()
        random.shuffle(train_items)
        total_loss = 0.0
        count = 0
        progress = tqdm(train_items, desc=f"Epoch {epoch}/{args.epochs}", unit="seq")
        for _, seq, ref_coords in progress:
            msa, base_x, seq_idx = bundle.preprocessor.build_features(seq, bundle.device)
            optimizer.zero_grad()
            pred = forward_train(bundle.model, msa, seq_idx, base_x, np.array(list(seq)))
            pred_atom = pred[:, args.atom_index, :]
            loss = masked_mse(pred_atom, ref_coords, scaler=coord_scaler)
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            count += 1
            progress.set_postfix(loss=f"{total_loss / count:.6f}")

        train_loss = total_loss / max(count, 1)
        epoch_record: dict[str, float] = {"epoch": float(epoch), "train_loss": float(train_loss)}

        if eval_items:
            eval_summary = evaluate_items(
                bundle.model,
                eval_items,
                bundle.preprocessor,
                bundle.data_module,
                bundle.device,
                args.atom_index,
            )
            epoch_record.update(eval_summary)

        history.append(epoch_record)
        if metrics_path is not None:
            pd.DataFrame(history).to_csv(metrics_path, index=False)

        print(
            f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.6f}"
            + (
                f", eval_rmse: {epoch_record.get('rmse', float('nan')):.6f}"
                if "rmse" in epoch_record
                else ""
            )
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_artifact(Path(args.artifact_out), bundle, history)
    print(f"Saved artifact: {args.artifact_out}")


if __name__ == "__main__":
    main()