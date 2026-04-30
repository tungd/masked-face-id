#!/usr/bin/env python3
"""Train the pair head on one dataset and evaluate on a separate real-mask root."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from probe_frozen_adapter import sample_pairs, select_records
from probe_pair_verifier_head import (
    MODEL_BASELINE,
    MODEL_BLACKOUT_GATE,
    MODEL_BLUR_GATE,
    MODEL_PAIR_HEAD,
    MODEL_PAIR_HEAD_MASKED_ONLY,
    MODEL_UPPER_GATE,
    compute_embeddings,
    feature_matrix,
    predict_scores,
    score_eval_pairs,
    stable_split,
    standardize,
    summarize,
    train_pair_head,
)


def write_conclusion(metrics: pd.DataFrame, out_dir: Path, train_root: Path, eval_root: Path) -> None:
    def auc(model: str, case: str) -> float:
        row = metrics[(metrics["model"] == model) & (metrics["case"] == case)]
        return float(row.iloc[0].roc_auc) if len(row) else math.nan

    baseline_mu = auc(MODEL_BASELINE, "masked-unmasked")
    baseline_uu = auc(MODEL_BASELINE, "unmasked-unmasked")
    pair_mu = auc(MODEL_PAIR_HEAD_MASKED_ONLY, "masked-unmasked")
    pair_uu = auc(MODEL_PAIR_HEAD_MASKED_ONLY, "unmasked-unmasked")
    gain = pair_mu - baseline_mu
    regression = baseline_uu - pair_uu
    practical_gain_threshold = 0.01
    verdict = "PROMISING" if gain >= practical_gain_threshold and regression <= 0.03 else "MARGINAL"
    text = f"""# Synthetic-Train Real-Eval Pair Head Conclusion

Recommendation: {verdict}

- Train root: `{train_root}`
- Eval root: `{eval_root}`
- Baseline full FaceNet masked-unmasked ROC-AUC: {baseline_mu:.4f}
- Pair head masked-only masked-unmasked ROC-AUC: {pair_mu:.4f}
- Pair head masked-only gain vs baseline: {gain:.4f}
- Baseline unmasked-unmasked ROC-AUC: {baseline_uu:.4f}
- Pair head masked-only unmasked-unmasked ROC-AUC: {pair_uu:.4f}
- Pair head masked-only unmasked regression vs baseline: {regression:.4f}
- Practical masked-unmasked gain threshold: {practical_gain_threshold:.4f}

This extension trains the pair-level verifier head on a synthetic or larger
masked/unmasked training root, then evaluates on a separate real-mask identity
split. FaceNet remains frozen and unmasked-unmasked pairs are bypassed to raw
FaceNet for the practical masked-only policy.
"""
    (out_dir / "synthetic_train_real_eval_conclusion.md").write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-root", type=Path, required=True)
    parser.add_argument("--eval-data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-identities", type=int, default=1000)
    parser.add_argument("--eval-identities", type=int, default=80)
    parser.add_argument("--max-train-images-per-condition", type=int, default=4)
    parser.add_argument("--max-eval-images-per-condition", type=int, default=8)
    parser.add_argument("--train-pairs-per-case", type=int, default=6000)
    parser.add_argument("--eval-pairs-per-case", type=int, default=800)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print({"device": device, "train_data_root": str(args.train_data_root), "eval_data_root": str(args.eval_data_root)})

    train_records, _unused_eval_records, train_ids, _unused_eval_ids = select_records(
        args.train_data_root,
        train_identities=args.train_identities,
        eval_identities=0,
        max_images_per_condition=args.max_train_images_per_condition,
        seed=args.seed,
    )
    _unused_train_records, eval_records, _unused_train_ids, eval_ids = select_records(
        args.eval_data_root,
        train_identities=0,
        eval_identities=args.eval_identities,
        max_images_per_condition=args.max_eval_images_per_condition,
        seed=args.seed,
    )
    print(
        {
            "train_identities": len(train_ids),
            "eval_identities": len(eval_ids),
            "train_images": len(train_records),
            "eval_images": len(eval_records),
        }
    )

    embeddings = compute_embeddings(train_records + eval_records, image_size=args.image_size, device=device)
    train_pairs = sample_pairs(train_records, args.train_pairs_per_case, args.seed + 17)
    eval_pairs = sample_pairs(eval_records, args.eval_pairs_per_case, args.seed)

    train_x_all, train_y_all, kept_train_pairs = feature_matrix(train_pairs, embeddings)
    eval_x, _eval_y, kept_eval_pairs = feature_matrix(eval_pairs, embeddings)
    cal_mask = stable_split(kept_train_pairs, calibration_fraction=0.2, seed=args.seed + 99)
    fit_x = train_x_all[~cal_mask]
    fit_y = train_y_all[~cal_mask]
    cal_x = train_x_all[cal_mask]
    cal_y = train_y_all[cal_mask]
    fit_x, standardized, mean, std = standardize(fit_x, cal_x, eval_x)
    cal_x, eval_x = standardized
    print({"fit_pairs": len(fit_x), "calibration_pairs": len(cal_x), "eval_pairs": len(eval_x), "feature_dim": fit_x.shape[1]})

    head, curve = train_pair_head(
        fit_x,
        fit_y,
        cal_x,
        cal_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )
    head_scores = predict_scores(head, eval_x, batch_size=args.batch_size, device=device)
    scores = score_eval_pairs(kept_eval_pairs, embeddings, head_scores)
    metrics = summarize(scores)

    scores.to_csv(args.out_dir / "synthetic_train_real_eval_pair_scores.csv", index=False)
    metrics.to_csv(args.out_dir / "synthetic_train_real_eval_results.csv", index=False)
    curve.to_csv(args.out_dir / "synthetic_train_real_eval_training_curve.csv", index=False)
    np.savez(args.out_dir / "synthetic_train_real_eval_standardizer.npz", mean=mean, std=std)
    pd.DataFrame({"identity": train_ids, "split": "train"}).to_csv(args.out_dir / "train_identities.csv", index=False)
    pd.DataFrame({"identity": eval_ids, "split": "eval"}).to_csv(args.out_dir / "eval_identities.csv", index=False)

    print(metrics.to_string(index=False))
    write_conclusion(metrics, args.out_dir, train_root=args.train_data_root, eval_root=args.eval_data_root)


if __name__ == "__main__":
    main()
