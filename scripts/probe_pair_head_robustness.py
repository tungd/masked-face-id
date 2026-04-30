#!/usr/bin/env python3
"""Run pair-head robustness, feature ablation, and threshold analyses."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from probe_frozen_adapter import Pair, best_threshold, cosine, sample_pairs, select_records
from probe_occlusion_ensemble import RAW_VIEWS, compute_embeddings
from probe_pair_verifier_head import (
    MODEL_BASELINE,
    MODEL_BLACKOUT_GATE,
    MODEL_BLUR_GATE,
    MODEL_UPPER_GATE,
    raw_policy_score,
    stable_split,
    standardize,
    train_pair_head,
    predict_scores,
)


FEATURE_MODES = [
    "full_all_features",
    "cosine_scores_only",
    "cosine_scores_stats",
    "dense_interactions_only",
    "full_face_dense_only",
]


def case_one_hot(case: str) -> np.ndarray:
    return np.array([case == "masked-masked", case == "masked-unmasked", case == "unmasked-unmasked"], dtype=np.float32)


def score_stats(scores: np.ndarray) -> np.ndarray:
    return np.array(
        [
            scores.mean(),
            scores.std(),
            scores.max(),
            scores.min(),
            np.sort(scores)[-2:].mean(),
            scores[0] - scores[1],
            scores[0] - scores[2],
            scores[0] - scores[3],
            scores[0] - scores[4],
        ],
        dtype=np.float32,
    )


def dense_interactions(left: dict[str, np.ndarray], right: dict[str, np.ndarray], views: Sequence[str]) -> list[np.ndarray]:
    parts = []
    for view in views:
        l_vec = left[view].astype(np.float32)
        r_vec = right[view].astype(np.float32)
        parts.extend([np.abs(l_vec - r_vec), l_vec * r_vec])
    return parts


def pair_features_mode(pair: Pair, embeddings: dict[Path, dict[str, np.ndarray]], mode: str) -> np.ndarray | None:
    if pair.left not in embeddings or pair.right not in embeddings:
        return None
    left = embeddings[pair.left]
    right = embeddings[pair.right]
    scores = np.array([cosine(left[view], right[view]) for view in RAW_VIEWS], dtype=np.float32)
    prefix = [case_one_hot(pair.case)]
    if mode == "cosine_scores_only":
        return np.concatenate([*prefix, scores]).astype(np.float32)
    if mode == "cosine_scores_stats":
        return np.concatenate([*prefix, scores, score_stats(scores)]).astype(np.float32)
    if mode == "dense_interactions_only":
        return np.concatenate([*prefix, *dense_interactions(left, right, RAW_VIEWS)]).astype(np.float32)
    if mode == "full_face_dense_only":
        return np.concatenate([*prefix, scores[:1], *dense_interactions(left, right, ["full"])]).astype(np.float32)
    if mode == "full_all_features":
        return np.concatenate([*prefix, scores, score_stats(scores), *dense_interactions(left, right, RAW_VIEWS)]).astype(np.float32)
    raise KeyError(mode)


def feature_matrix_mode(
    pairs: Sequence[Pair],
    embeddings: dict[Path, dict[str, np.ndarray]],
    mode: str,
) -> tuple[np.ndarray, np.ndarray, list[Pair]]:
    features = []
    labels = []
    kept = []
    for pair in pairs:
        vector = pair_features_mode(pair, embeddings, mode)
        if vector is None:
            continue
        features.append(vector)
        labels.append(pair.label)
        kept.append(pair)
    if not features:
        raise ValueError(f"No pairs survived feature construction for mode={mode}")
    return np.stack(features), np.array(labels, dtype=np.float32), kept


def raw_score_rows(seed: int, split: str, pairs: Sequence[Pair], embeddings: dict[Path, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    rows = []
    for pair in pairs:
        if pair.left not in embeddings or pair.right not in embeddings:
            continue
        for model in [MODEL_BASELINE, MODEL_UPPER_GATE, MODEL_BLUR_GATE, MODEL_BLACKOUT_GATE]:
            rows.append(
                {
                    **asdict(pair),
                    "left": str(pair.left),
                    "right": str(pair.right),
                    "seed": seed,
                    "split": split,
                    "model": model,
                    "feature_mode": "raw_policy",
                    "score": raw_policy_score(pair, embeddings, model),
                }
            )
    return rows


def head_score_rows(
    seed: int,
    split: str,
    mode: str,
    pairs: Sequence[Pair],
    embeddings: dict[Path, dict[str, np.ndarray]],
    head_scores: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    all_cases_model = f"pair_head_{mode}_all_cases"
    masked_only_model = f"pair_head_{mode}_masked_only"
    for idx, pair in enumerate(pairs):
        rows.append(
            {
                **asdict(pair),
                "left": str(pair.left),
                "right": str(pair.right),
                "seed": seed,
                "split": split,
                "model": all_cases_model,
                "feature_mode": mode,
                "score": float(head_scores[idx]),
            }
        )
        masked_only_score = raw_policy_score(pair, embeddings, MODEL_BASELINE) if pair.case == "unmasked-unmasked" else float(head_scores[idx])
        rows.append(
            {
                **asdict(pair),
                "left": str(pair.left),
                "right": str(pair.right),
                "seed": seed,
                "split": split,
                "model": masked_only_model,
                "feature_mode": mode,
                "score": masked_only_score,
            }
        )
    return rows


def train_mode(
    seed: int,
    mode: str,
    train_pairs: Sequence[Pair],
    eval_pairs: Sequence[Pair],
    embeddings: dict[Path, dict[str, np.ndarray]],
    args: argparse.Namespace,
    device: str,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    train_x_all, train_y_all, kept_train_pairs = feature_matrix_mode(train_pairs, embeddings, mode)
    eval_x, _eval_y, kept_eval_pairs = feature_matrix_mode(eval_pairs, embeddings, mode)
    cal_mask = stable_split(kept_train_pairs, calibration_fraction=args.calibration_fraction, seed=seed + 99)
    fit_x = train_x_all[~cal_mask]
    fit_y = train_y_all[~cal_mask]
    cal_x = train_x_all[cal_mask]
    cal_y = train_y_all[cal_mask]
    cal_pairs = [pair for pair, keep in zip(kept_train_pairs, cal_mask, strict=True) if keep]
    fit_x, standardized, _mean, _std = standardize(fit_x, cal_x, eval_x)
    cal_x, eval_x = standardized
    print({"seed": seed, "mode": mode, "fit_pairs": len(fit_x), "calibration_pairs": len(cal_x), "eval_pairs": len(eval_x), "feature_dim": fit_x.shape[1]})
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
        seed=seed,
        device=device,
    )
    cal_scores = predict_scores(head, cal_x, batch_size=args.batch_size, device=device)
    eval_scores = predict_scores(head, eval_x, batch_size=args.batch_size, device=device)
    curve.insert(0, "feature_mode", mode)
    curve.insert(0, "seed", seed)
    rows = []
    rows.extend(head_score_rows(seed, "calibration", mode, cal_pairs, embeddings, cal_scores))
    rows.extend(head_score_rows(seed, "eval", mode, kept_eval_pairs, embeddings, eval_scores))
    return rows, curve


def auc_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eval_scores = scores[scores["split"] == "eval"]
    for (seed, model, case), case_df in eval_scores.groupby(["seed", "model", "case"]):
        labels = case_df["label"].to_numpy()
        values = case_df["score"].to_numpy()
        rows.append(
            {
                "seed": seed,
                "model": model,
                "case": case,
                "pairs": len(case_df),
                "roc_auc": float(roc_auc_score(labels, values)) if len(np.unique(labels)) == 2 else math.nan,
                **best_threshold(labels, values),
            }
        )
    return pd.DataFrame(rows).sort_values(["case", "model", "seed"])


def apply_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores >= threshold).astype(int)
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tp = int(((preds == 1) & (labels == 1)).sum())
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "far": fp / max(fp + tn, 1),
        "frr": fn / max(fn + tp, 1),
        "tar": tp / max(tp + fn, 1),
    }


def far_threshold(labels: np.ndarray, scores: np.ndarray, target_far: float) -> float:
    negatives = np.sort(scores[labels == 0])
    if len(negatives) == 0:
        return float(scores.max())
    allowed_false = int(math.floor(target_far * len(negatives)))
    index = max(0, len(negatives) - allowed_false - 1)
    return float(negatives[index])


def threshold_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    calibration = scores[scores["split"] == "calibration"]
    eval_scores = scores[scores["split"] == "eval"]
    for (seed, model, case), cal_df in calibration.groupby(["seed", "model", "case"]):
        eval_df = eval_scores[(eval_scores["seed"] == seed) & (eval_scores["model"] == model) & (eval_scores["case"] == case)]
        if eval_df.empty:
            continue
        cal_labels = cal_df["label"].to_numpy()
        cal_values = cal_df["score"].to_numpy()
        eval_labels = eval_df["label"].to_numpy()
        eval_values = eval_df["score"].to_numpy()
        policies = {
            "cal_best_accuracy": best_threshold(cal_labels, cal_values)["threshold"] if len(np.unique(cal_labels)) == 2 else float(np.median(cal_values)),
            "cal_far_0.01": far_threshold(cal_labels, cal_values, 0.01),
            "cal_far_0.05": far_threshold(cal_labels, cal_values, 0.05),
            "cal_far_0.10": far_threshold(cal_labels, cal_values, 0.10),
        }
        for policy, threshold in policies.items():
            rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "case": case,
                    "threshold_policy": policy,
                    "threshold": threshold,
                    "eval_pairs": len(eval_df),
                    **apply_threshold(eval_labels, eval_values, threshold),
                }
            )
    return pd.DataFrame(rows).sort_values(["case", "model", "threshold_policy", "seed"])


def aggregate_metrics(metrics: pd.DataFrame, value_col: str = "roc_auc") -> pd.DataFrame:
    rows = []
    for (model, case), df in metrics.groupby(["model", "case"]):
        values = df[value_col].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "case": case,
                "seeds": len(values),
                f"{value_col}_mean": float(np.mean(values)),
                f"{value_col}_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                f"{value_col}_min": float(np.min(values)),
                f"{value_col}_max": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows).sort_values(["case", f"{value_col}_mean", "model"], ascending=[True, False, True])


def write_readme(out_dir: Path, args: argparse.Namespace, summary: dict[str, object]) -> None:
    text = f"""# Pair-Head Robustness, Ablation, and Threshold Probe

This artifact evaluates the FaceNet pair-head method across multiple identity
splits, feature ablations, and calibration-derived thresholds.

Configuration:

- Data root: `{args.data_root}`
- Seeds: `{args.seeds}`
- Train identities per seed: {args.train_identities}
- Eval identities per seed: {args.eval_identities}
- Max images per condition: {args.max_images_per_condition}
- Train pairs per case: {args.train_pairs_per_case}
- Eval pairs per case: {args.eval_pairs_per_case}
- Pair-head epochs: {args.epochs}
- Feature modes: `{FEATURE_MODES}`

Key masked-unmasked ROC-AUC means:

- Raw FaceNet baseline: {summary["baseline_mu_mean"]:.4f}
- Full pair head, masked-only policy: {summary["pair_head_mu_mean"]:.4f}
- Full pair head gain: {summary["pair_head_mu_gain"]:.4f}

Best feature ablation on masked-unmasked mean ROC-AUC:

- Model: `{summary["best_ablation_model"]}`
- ROC-AUC mean: {summary["best_ablation_mu_mean"]:.4f}

Threshold analysis is saved in `threshold_metrics.csv`. These thresholds are
chosen on calibration pairs and applied to held-out eval pairs.
"""
    (out_dir / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 99])
    parser.add_argument("--train-identities", type=int, default=300)
    parser.add_argument("--eval-identities", type=int, default=100)
    parser.add_argument("--max-images-per-condition", type=int, default=8)
    parser.add_argument("--train-pairs-per-case", type=int, default=10000)
    parser.add_argument("--eval-pairs-per-case", type=int, default=2000)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_score_rows = []
    curves = []
    run_summaries = []
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_records, eval_records, train_ids, eval_ids = select_records(
            args.data_root,
            args.train_identities,
            args.eval_identities,
            args.max_images_per_condition,
            seed,
        )
        print({"seed": seed, "device": device, "train_identities": len(train_ids), "eval_identities": len(eval_ids), "train_images": len(train_records), "eval_images": len(eval_records)})
        embeddings = compute_embeddings(train_records + eval_records, image_size=args.image_size, device=device)
        train_pairs = sample_pairs(train_records, args.train_pairs_per_case, seed + 17)
        eval_pairs = sample_pairs(eval_records, args.eval_pairs_per_case, seed)
        cal_mask_for_raw = stable_split(train_pairs, calibration_fraction=args.calibration_fraction, seed=seed + 99)
        cal_pairs_for_raw = [pair for pair, keep in zip(train_pairs, cal_mask_for_raw, strict=True) if keep]
        all_score_rows.extend(raw_score_rows(seed, "calibration", cal_pairs_for_raw, embeddings))
        all_score_rows.extend(raw_score_rows(seed, "eval", eval_pairs, embeddings))
        for mode in FEATURE_MODES:
            rows, curve = train_mode(seed, mode, train_pairs, eval_pairs, embeddings, args, device)
            all_score_rows.extend(rows)
            curves.append(curve)
        run_summaries.append(
            {
                "seed": seed,
                "train_identities": len(train_ids),
                "eval_identities": len(eval_ids),
                "train_images": len(train_records),
                "eval_images": len(eval_records),
                "embedded_images": len(embeddings),
                "embedding_failures": len(train_records) + len(eval_records) - len(embeddings),
            }
        )

    scores = pd.DataFrame(all_score_rows)
    metrics = auc_metrics(scores)
    thresholds = threshold_metrics(scores)
    aggregate = aggregate_metrics(metrics)
    threshold_aggregate = aggregate_metrics(
        thresholds[thresholds["threshold_policy"] == "cal_far_0.05"].rename(columns={"tar": "roc_auc"}),
        value_col="roc_auc",
    )
    curves_df = pd.concat(curves, ignore_index=True)

    scores.to_csv(args.out_dir / "pair_head_robustness_pair_scores.csv", index=False)
    metrics.to_csv(args.out_dir / "pair_head_robustness_metrics.csv", index=False)
    aggregate.to_csv(args.out_dir / "pair_head_robustness_aggregate.csv", index=False)
    thresholds.to_csv(args.out_dir / "threshold_metrics.csv", index=False)
    threshold_aggregate.to_csv(args.out_dir / "threshold_far05_tar_aggregate.csv", index=False)
    curves_df.to_csv(args.out_dir / "pair_head_robustness_training_curves.csv", index=False)
    pd.DataFrame(run_summaries).to_csv(args.out_dir / "run_summaries.csv", index=False)

    def aggregate_auc(model: str, case: str) -> float:
        row = aggregate[(aggregate["model"] == model) & (aggregate["case"] == case)]
        return float(row.iloc[0].roc_auc_mean) if len(row) else math.nan

    pair_model = "pair_head_full_all_features_masked_only"
    baseline_mu = aggregate_auc(MODEL_BASELINE, "masked-unmasked")
    pair_mu = aggregate_auc(pair_model, "masked-unmasked")
    ablations = aggregate[(aggregate["case"] == "masked-unmasked") & (aggregate["model"].str.startswith("pair_head_"))]
    best = ablations.sort_values("roc_auc_mean", ascending=False).iloc[0]
    summary = {
        "device": device,
        "seeds": args.seeds,
        "baseline_mu_mean": baseline_mu,
        "pair_head_mu_mean": pair_mu,
        "pair_head_mu_gain": pair_mu - baseline_mu,
        "best_ablation_model": best["model"],
        "best_ablation_mu_mean": float(best["roc_auc_mean"]),
        "runs": run_summaries,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_readme(args.out_dir, args, summary)
    print(json.dumps(summary, indent=2))
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
