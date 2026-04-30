#!/usr/bin/env python3
"""Analyze calibration-derived operating points from pair score CSVs.

The main GPU probes run in Colab and can emit raw score rows with calibration
and eval splits. This helper is intentionally post-hoc: run it after downloading
or mounting a generated `pair_head_robustness_pair_scores.csv`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"seed", "split", "model", "case", "label", "score"}
DEFAULT_MODELS = [
    "baseline_full",
    "pair_head_cosine_scores_only_masked_only",
    "pair_head_full_all_features_masked_only",
    "pair_head_full_face_dense_only_masked_only",
]
DEFAULT_TARGET_FARS = [0.01, 0.05, 0.10]


def validate_scores(scores: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(scores.columns))
    if missing:
        raise ValueError(f"Score CSV is missing required columns: {missing}")
    splits = set(scores["split"].unique())
    if "calibration" not in splits or "eval" not in splits:
        raise ValueError("Score CSV must contain both calibration and eval rows")


def confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tp = int(((preds == 1) & (labels == 1)).sum())
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "far": fp / max(fp + tn, 1),
        "frr": fn / max(fn + tp, 1),
        "tar": tp / max(tp + fn, 1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def best_accuracy_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    best_acc = -math.inf
    best_threshold = float(scores.min())
    for threshold in np.unique(scores):
        acc = confusion(labels, scores, float(threshold))["accuracy"]
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(threshold)
    return best_threshold


def threshold_for_far(labels: np.ndarray, scores: np.ndarray, target_far: float) -> float:
    negatives = np.sort(scores[labels == 0])
    if len(negatives) == 0:
        return float(scores.max())
    allowed_false_accepts = int(math.floor(target_far * len(negatives)))
    if allowed_false_accepts <= 0:
        return float(np.nextafter(negatives[-1], np.inf))
    index = max(0, len(negatives) - allowed_false_accepts)
    return float(negatives[index])


def iter_groups(scores: pd.DataFrame, models: Iterable[str]) -> Iterable[tuple[int, str, str, pd.DataFrame, pd.DataFrame]]:
    wanted = set(models)
    calibration = scores[scores["split"] == "calibration"]
    evaluation = scores[scores["split"] == "eval"]
    for (seed, model, case), cal_df in calibration.groupby(["seed", "model", "case"], sort=True):
        if wanted and model not in wanted:
            continue
        eval_df = evaluation[
            (evaluation["seed"] == seed)
            & (evaluation["model"] == model)
            & (evaluation["case"] == case)
        ]
        if eval_df.empty:
            continue
        yield int(seed), str(model), str(case), cal_df, eval_df


def operating_points(scores: pd.DataFrame, models: list[str], target_fars: list[float]) -> pd.DataFrame:
    rows = []
    for seed, model, case, cal_df, eval_df in iter_groups(scores, models):
        cal_labels = cal_df["label"].to_numpy(dtype=np.int64)
        cal_scores = cal_df["score"].to_numpy(dtype=float)
        eval_labels = eval_df["label"].to_numpy(dtype=np.int64)
        eval_scores = eval_df["score"].to_numpy(dtype=float)

        policies = {"cal_best_accuracy": best_accuracy_threshold(cal_labels, cal_scores)}
        for target_far in target_fars:
            policies[f"cal_far_{target_far:g}"] = threshold_for_far(cal_labels, cal_scores, target_far)

        for policy, threshold in policies.items():
            rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "case": case,
                    "threshold_policy": policy,
                    "threshold": threshold,
                    "cal_pairs": len(cal_df),
                    "eval_pairs": len(eval_df),
                    **confusion(eval_labels, eval_scores, threshold),
                }
            )
    columns = [
        "seed",
        "model",
        "case",
        "threshold_policy",
        "threshold",
        "cal_pairs",
        "eval_pairs",
        "accuracy",
        "far",
        "frr",
        "tar",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["case", "model", "threshold_policy", "seed"])


def aggregate(points: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["accuracy", "far", "frr", "tar"]
    rows = []
    for (model, case, policy), df in points.groupby(["model", "case", "threshold_policy"], sort=True):
        row: dict[str, object] = {
            "model": model,
            "case": case,
            "threshold_policy": policy,
            "seeds": int(df["seed"].nunique()),
        }
        for metric in metric_cols:
            values = df[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        rows.append(row)
    columns = ["model", "case", "threshold_policy", "seeds"]
    for metric in metric_cols:
        columns.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_min", f"{metric}_max"])
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["case", "threshold_policy", "tar_mean"],
        ascending=[True, True, False],
    )


def baseline_deltas(agg: pd.DataFrame, baseline_model: str) -> pd.DataFrame:
    rows = []
    baseline = agg[agg["model"] == baseline_model]
    for _, row in agg.iterrows():
        base = baseline[
            (baseline["case"] == row["case"])
            & (baseline["threshold_policy"] == row["threshold_policy"])
        ]
        if base.empty or row["model"] == baseline_model:
            continue
        base_row = base.iloc[0]
        rows.append(
            {
                "model": row["model"],
                "case": row["case"],
                "threshold_policy": row["threshold_policy"],
                "tar_delta_vs_baseline": float(row["tar_mean"] - base_row["tar_mean"]),
                "far_delta_vs_baseline": float(row["far_mean"] - base_row["far_mean"]),
                "accuracy_delta_vs_baseline": float(row["accuracy_mean"] - base_row["accuracy_mean"]),
            }
        )
    columns = [
        "model",
        "case",
        "threshold_policy",
        "tar_delta_vs_baseline",
        "far_delta_vs_baseline",
        "accuracy_delta_vs_baseline",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["case", "threshold_policy", "tar_delta_vs_baseline"],
        ascending=[True, True, False],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True, help="Raw score CSV with calibration and eval rows.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--target-fars", type=float, nargs="+", default=DEFAULT_TARGET_FARS)
    parser.add_argument("--baseline-model", default="baseline_full")
    args = parser.parse_args()

    scores = pd.read_csv(args.scores)
    validate_scores(scores)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    points = operating_points(scores, args.models, args.target_fars)
    if points.empty:
        raise ValueError("No operating points were produced. Check --models and score CSV contents.")
    agg = aggregate(points)
    deltas = baseline_deltas(agg, args.baseline_model)

    points.to_csv(args.out_dir / "calibration_operating_points.csv", index=False)
    agg.to_csv(args.out_dir / "calibration_operating_points_aggregate.csv", index=False)
    deltas.to_csv(args.out_dir / "calibration_baseline_deltas.csv", index=False)

    summary = {
        "scores": str(args.scores),
        "models": args.models,
        "target_fars": args.target_fars,
        "baseline_model": args.baseline_model,
        "rows": {
            "operating_points": len(points),
            "aggregate": len(agg),
            "baseline_deltas": len(deltas),
        },
    }
    (args.out_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
