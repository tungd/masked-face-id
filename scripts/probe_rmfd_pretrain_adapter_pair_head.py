#!/usr/bin/env python3
"""Self-supervised masked adapter pretraining before pair-head fine-tuning."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from probe_frozen_adapter import FaceRecord, Pair, discover_records, sample_pairs, select_records
from probe_occlusion_ensemble import RAW_VIEWS, compute_embeddings
from probe_pair_verifier_head import (
    MODEL_BASELINE,
    MODEL_BLACKOUT_GATE,
    MODEL_BLUR_GATE,
    MODEL_PAIR_HEAD,
    MODEL_PAIR_HEAD_MASKED_ONLY,
    MODEL_UPPER_GATE,
    feature_matrix,
    predict_scores,
    raw_policy_score,
    score_eval_pairs,
    stable_split,
    standardize,
    summarize,
    train_pair_head,
)


class ResidualEmbeddingAdapter(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float, residual_scale: float):
        super().__init__()
        self.residual_scale = residual_scale
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + self.residual_scale * self.net(x), dim=-1)


def unique_records(records: Sequence[FaceRecord]) -> list[FaceRecord]:
    seen: set[Path] = set()
    out: list[FaceRecord] = []
    for record in records:
        if record.path in seen:
            continue
        seen.add(record.path)
        out.append(record)
    return out


def masked_records(records: Sequence[FaceRecord], max_images: int, seed: int) -> list[FaceRecord]:
    records = [record for record in records if record.condition == "masked"]
    rng = random.Random(seed)
    rng.shuffle(records)
    if max_images > 0:
        records = records[:max_images]
    return records


def stack_pretrain_views(
    records: Sequence[FaceRecord],
    embeddings: dict[Path, dict[str, np.ndarray]],
) -> tuple[torch.Tensor, list[FaceRecord]]:
    rows = []
    kept = []
    for record in records:
        views = embeddings.get(record.path)
        if not views:
            continue
        rows.append(np.stack([views[view].astype(np.float32) for view in RAW_VIEWS]))
        kept.append(record)
    if not rows:
        raise ValueError("No masked pretraining records survived embedding")
    return torch.tensor(np.stack(rows), dtype=torch.float32), kept


def train_residual_adapter(
    view_tensor: torch.Tensor,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    dropout: float,
    residual_scale: float,
    lr: float,
    weight_decay: float,
    temperature: float,
    preserve_weight: float,
    seed: int,
    device: str,
) -> tuple[ResidualEmbeddingAdapter, pd.DataFrame]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    view_tensor = view_tensor.to(device)
    model = ResidualEmbeddingAdapter(view_tensor.shape[-1], hidden_dim, dropout, residual_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rows = []
    n_images, n_views, _dim = view_tensor.shape

    for epoch in tqdm(range(1, epochs + 1), desc="Pretraining residual adapter"):
        model.train()
        order = rng.permutation(n_images)
        losses = []
        contrastive_losses = []
        preserve_losses = []
        for start in range(0, n_images, batch_size):
            idx_np = order[start : start + batch_size]
            if len(idx_np) < 2:
                continue
            idx = torch.tensor(idx_np, dtype=torch.long, device=device)
            first_views = torch.tensor(rng.integers(0, n_views, size=len(idx_np)), dtype=torch.long, device=device)
            second_offsets = torch.tensor(rng.integers(1, n_views, size=len(idx_np)), dtype=torch.long, device=device)
            second_views = (first_views + second_offsets) % n_views
            x1 = view_tensor[idx, first_views]
            x2 = view_tensor[idx, second_views]
            z1 = model(x1)
            z2 = model(x2)
            labels = torch.arange(len(idx_np), device=device)
            logits = z1 @ z2.T / temperature
            contrastive = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            preserve = 0.5 * (F.mse_loss(z1, x1) + F.mse_loss(z2, x2))
            loss = contrastive + preserve_weight * preserve
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            contrastive_losses.append(float(contrastive.detach().cpu()))
            preserve_losses.append(float(preserve.detach().cpu()))
        rows.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "contrastive_loss": float(np.mean(contrastive_losses)),
                "preserve_loss": float(np.mean(preserve_losses)),
            }
        )
    return model.eval(), pd.DataFrame(rows)


@torch.inference_mode()
def adapt_embeddings(
    embeddings: dict[Path, dict[str, np.ndarray]],
    records: Sequence[FaceRecord],
    adapter: ResidualEmbeddingAdapter,
    batch_size: int,
    device: str,
) -> dict[Path, dict[str, np.ndarray]]:
    masked_paths = {record.path for record in records if record.condition == "masked" and record.path in embeddings}
    adapted = {path: {view: vector.copy() for view, vector in views.items()} for path, views in embeddings.items()}
    paths = sorted(masked_paths)
    adapter = adapter.to(device).eval()
    for view in RAW_VIEWS:
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            x = torch.tensor(np.stack([embeddings[path][view].astype(np.float32) for path in batch_paths]), device=device)
            z = adapter(x).detach().cpu().numpy()
            for path, vector in zip(batch_paths, z, strict=True):
                adapted[path][view] = vector.astype(np.float32)
    return adapted


def rename_score_models(scores: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = scores.copy()
    out["model"] = prefix + out["model"].astype(str)
    return out


def fit_pair_head_scores(
    prefix: str,
    train_pairs: Sequence[Pair],
    eval_pairs: Sequence[Pair],
    embeddings: dict[Path, dict[str, np.ndarray]],
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x_all, train_y_all, kept_train_pairs = feature_matrix(train_pairs, embeddings)
    eval_x, _eval_y, kept_eval_pairs = feature_matrix(eval_pairs, embeddings)
    cal_mask = stable_split(kept_train_pairs, calibration_fraction=0.2, seed=seed + 99)
    fit_x = train_x_all[~cal_mask]
    fit_y = train_y_all[~cal_mask]
    cal_x = train_x_all[cal_mask]
    cal_y = train_y_all[cal_mask]
    fit_x, standardized, _mean, _std = standardize(fit_x, cal_x, eval_x)
    cal_x, eval_x = standardized
    print(
        {
            "prefix": prefix,
            "fit_pairs": len(fit_x),
            "calibration_pairs": len(cal_x),
            "eval_pairs": len(eval_x),
            "feature_dim": fit_x.shape[1],
        }
    )
    head, curve = train_pair_head(
        fit_x,
        fit_y,
        cal_x,
        cal_y,
        epochs=epochs,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
    )
    head_scores = predict_scores(head, eval_x, batch_size=batch_size, device=device)
    scores = score_eval_pairs(kept_eval_pairs, embeddings, head_scores)
    curve.insert(0, "run", prefix.rstrip("_"))
    return rename_score_models(scores, prefix), curve


def write_conclusion(metrics: pd.DataFrame, summary: dict[str, object], out_dir: Path) -> None:
    def auc(model: str, case: str) -> float:
        row = metrics[(metrics["model"] == model) & (metrics["case"] == case)]
        return float(row.iloc[0].roc_auc) if len(row) else math.nan

    raw_mu = auc("raw_pair_head_masked_cases_only", "masked-unmasked")
    adapted_mu = auc("adapted_pair_head_masked_cases_only", "masked-unmasked")
    raw_uu = auc("raw_pair_head_masked_cases_only", "unmasked-unmasked")
    adapted_uu = auc("adapted_pair_head_masked_cases_only", "unmasked-unmasked")
    baseline_mu = auc("raw_baseline_full", "masked-unmasked")
    verdict = "HELPFUL" if adapted_mu > raw_mu else "NOT HELPFUL"
    text = f"""# RMFD Masked Pretraining + Pair Head Probe

Recommendation: {verdict}

- Total masked images discovered for pretraining: {summary["pretrain_masked_images_requested"]}
- Masked images embedded for pretraining: {summary["pretrain_masked_images_embedded"]}
- Train identities: {summary["train_identities"]}
- Eval identities: {summary["eval_identities"]}
- Raw FaceNet masked-unmasked ROC-AUC: {baseline_mu:.4f}
- Pair-head-only masked-unmasked ROC-AUC: {raw_mu:.4f}
- Pretrained-adapter pair-head masked-unmasked ROC-AUC: {adapted_mu:.4f}
- Adapter gain vs pair-head-only: {adapted_mu - raw_mu:.4f}
- Pair-head-only unmasked-unmasked ROC-AUC: {raw_uu:.4f}
- Pretrained-adapter pair-head unmasked-unmasked ROC-AUC: {adapted_uu:.4f}
- Adapter unmasked-unmasked regression: {raw_uu - adapted_uu:.4f}

The adapter is a residual MLP over frozen FaceNet view embeddings. It is
pretrained without identity-pair labels by pulling different occlusion-view
embeddings of the same masked image together with an InfoNCE objective, with a
small preservation penalty to avoid moving embeddings too far from FaceNet.
During pair-head training and evaluation, the adapter is applied only to masked
image embeddings; unmasked embeddings remain raw FaceNet embeddings.
"""
    (out_dir / "pretrain_adapter_pair_head_conclusion.md").write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-identities", type=int, default=300)
    parser.add_argument("--eval-identities", type=int, default=100)
    parser.add_argument("--max-images-per-condition", type=int, default=8)
    parser.add_argument("--max-pretrain-masked-images", type=int, default=0)
    parser.add_argument("--train-pairs-per-case", type=int, default=10000)
    parser.add_argument("--eval-pairs-per-case", type=int, default=2000)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--pair-head-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pair-hidden-dim", type=int, default=512)
    parser.add_argument("--pair-dropout", type=float, default=0.25)
    parser.add_argument("--pair-lr", type=float, default=3e-4)
    parser.add_argument("--pair-weight-decay", type=float, default=1e-3)
    parser.add_argument("--adapter-hidden-dim", type=int, default=512)
    parser.add_argument("--adapter-dropout", type=float, default=0.10)
    parser.add_argument("--adapter-residual-scale", type=float, default=0.35)
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    parser.add_argument("--adapter-weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--preserve-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print({"device": device, "data_root": str(args.data_root)})

    all_records = discover_records(args.data_root)
    pretrain_records = masked_records(all_records, args.max_pretrain_masked_images, args.seed + 1001)
    train_records, eval_records, train_ids, eval_ids = select_records(
        args.data_root,
        args.train_identities,
        args.eval_identities,
        args.max_images_per_condition,
        args.seed,
    )
    all_embedding_records = unique_records(pretrain_records + train_records + eval_records)
    print(
        {
            "pretrain_masked_records": len(pretrain_records),
            "train_identities": len(train_ids),
            "eval_identities": len(eval_ids),
            "embedding_records": len(all_embedding_records),
        }
    )
    embeddings = compute_embeddings(all_embedding_records, image_size=args.image_size, device=device)
    pretrain_tensor, kept_pretrain_records = stack_pretrain_views(pretrain_records, embeddings)
    adapter, pretrain_curve = train_residual_adapter(
        pretrain_tensor,
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        hidden_dim=args.adapter_hidden_dim,
        dropout=args.adapter_dropout,
        residual_scale=args.adapter_residual_scale,
        lr=args.adapter_lr,
        weight_decay=args.adapter_weight_decay,
        temperature=args.temperature,
        preserve_weight=args.preserve_weight,
        seed=args.seed,
        device=device,
    )

    train_pairs = sample_pairs(train_records, args.train_pairs_per_case, args.seed + 17)
    eval_pairs = sample_pairs(eval_records, args.eval_pairs_per_case, args.seed)
    pair_records = unique_records(train_records + eval_records)
    adapted = adapt_embeddings(embeddings, pair_records, adapter, args.batch_size, device)

    raw_scores, raw_curve = fit_pair_head_scores(
        "raw_",
        train_pairs,
        eval_pairs,
        embeddings,
        epochs=args.pair_head_epochs,
        batch_size=args.batch_size,
        hidden_dim=args.pair_hidden_dim,
        dropout=args.pair_dropout,
        lr=args.pair_lr,
        weight_decay=args.pair_weight_decay,
        seed=args.seed,
        device=device,
    )
    adapted_scores, adapted_curve = fit_pair_head_scores(
        "adapted_",
        train_pairs,
        eval_pairs,
        adapted,
        epochs=args.pair_head_epochs,
        batch_size=args.batch_size,
        hidden_dim=args.pair_hidden_dim,
        dropout=args.pair_dropout,
        lr=args.pair_lr,
        weight_decay=args.pair_weight_decay,
        seed=args.seed,
        device=device,
    )
    scores = pd.concat([raw_scores, adapted_scores], ignore_index=True)
    metrics = summarize(scores)

    summary = {
        "data_root": str(args.data_root),
        "device": device,
        "pretrain_masked_images_requested": len(pretrain_records),
        "pretrain_masked_images_embedded": len(kept_pretrain_records),
        "embedding_records_requested": len(all_embedding_records),
        "embedding_records_embedded": len(embeddings),
        "embedding_failures": len(all_embedding_records) - len(embeddings),
        "train_identities": len(train_ids),
        "eval_identities": len(eval_ids),
        "train_images": len(train_records),
        "eval_images": len(eval_records),
        "train_pairs_sampled": len(train_pairs),
        "eval_pairs_sampled": len(eval_pairs),
        "raw_eval_pairs_scored": int(raw_scores[raw_scores["model"] == "raw_baseline_full"].shape[0]),
        "adapted_eval_pairs_scored": int(adapted_scores[adapted_scores["model"] == "adapted_baseline_full"].shape[0]),
        "seed": args.seed,
        "pretrain_epochs": args.pretrain_epochs,
        "pair_head_epochs": args.pair_head_epochs,
    }

    scores.to_csv(args.out_dir / "pretrain_adapter_pair_head_pair_scores.csv", index=False)
    metrics.to_csv(args.out_dir / "pretrain_adapter_pair_head_results.csv", index=False)
    pretrain_curve.to_csv(args.out_dir / "pretrain_adapter_curve.csv", index=False)
    pd.concat([raw_curve, adapted_curve], ignore_index=True).to_csv(args.out_dir / "pair_head_training_curves.csv", index=False)
    pd.DataFrame({"identity": train_ids, "split": "train"}).to_csv(args.out_dir / "train_identities.csv", index=False)
    pd.DataFrame({"identity": eval_ids, "split": "eval"}).to_csv(args.out_dir / "eval_identities.csv", index=False)
    (args.out_dir / "pretrain_adapter_pair_head_summary.json").write_text(json.dumps(summary, indent=2))
    print(metrics.to_string(index=False))
    write_conclusion(metrics, summary, args.out_dir)


if __name__ == "__main__":
    main()
