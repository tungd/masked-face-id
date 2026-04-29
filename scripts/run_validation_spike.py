#!/usr/bin/env python3
"""Run the masked face verification validation spike.

The real benchmark path expects a small identity dataset with both masked and
unmasked images. When `--smoke` is used, the script runs a synthetic proxy to
verify metrics and artifact writing in Colab before the dataset is mounted.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    best = {
        "accuracy": -1.0,
        "threshold": float(scores.min()),
        "far": math.nan,
        "frr": math.nan,
    }
    for threshold in np.unique(scores):
        preds = (scores >= threshold).astype(int)
        acc = accuracy_score(labels, preds)
        fp = int(((preds == 1) & (labels == 0)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tp = int(((preds == 1) & (labels == 1)).sum())
        if acc > best["accuracy"]:
            best = {
                "accuracy": float(acc),
                "threshold": float(threshold),
                "far": fp / max(fp + tn, 1),
                "frr": fn / max(fn + tp, 1),
            }
    return best


def run_smoke(results_dir: Path, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = np.r_[np.ones(300), np.zeros(300)]
    rows: list[dict[str, float | int | str]] = []

    cases = ["masked-masked", "masked-unmasked", "unmasked-unmasked"]
    for case in cases:
        model_specs = [
            ("baseline_facenet_proxy", 0.70 if case != "masked-unmasked" else 0.58, 0.35),
            ("upper_face_proxy", 0.68, 0.35),
            ("masked_specific_candidate_proxy", 0.72 if case != "unmasked-unmasked" else 0.69, 0.34),
        ]
        for model, pos_mu, neg_mu in model_specs:
            scores = np.r_[rng.normal(pos_mu, 0.08, 300), rng.normal(neg_mu, 0.08, 300)]
            rows.append(
                {
                    "model": model,
                    "case": case,
                    "pairs": len(scores),
                    "roc_auc": float(roc_auc_score(labels, scores)),
                    **best_threshold(labels, scores),
                }
            )

    metrics = pd.DataFrame(rows).sort_values(["case", "model"])
    metrics.to_csv(results_dir / "validation_results_smoke.csv", index=False)

    base_mu = metrics.query("model == 'baseline_facenet_proxy' and case == 'masked-unmasked'").iloc[0].roc_auc
    upper_mu = metrics.query("model == 'upper_face_proxy' and case == 'masked-unmasked'").iloc[0].roc_auc
    base_uu = metrics.query("model == 'baseline_facenet_proxy' and case == 'unmasked-unmasked'").iloc[0].roc_auc
    upper_uu = metrics.query("model == 'upper_face_proxy' and case == 'unmasked-unmasked'").iloc[0].roc_auc
    recommendation = (
        "GO for real-data benchmark"
        if upper_mu - base_mu >= 0.03 and base_uu - upper_uu <= 0.02
        else "NO-GO from smoke proxy alone"
    )
    conclusion = (
        f"# Smoke Spike Conclusion\n\n"
        f"Recommendation: {recommendation}\n\n"
        f"This is a synthetic proxy smoke run, not the real benchmark. Load "
        f"PKU-Masked-Face or RMFRD/SMFRD before making the real go/no-go call.\n\n"
        f"- masked-unmasked gain: {upper_mu - base_mu:.4f}\n"
        f"- unmasked regression: {base_uu - upper_uu:.4f}\n"
    )
    (results_dir / "validation_conclusion_smoke.md").write_text(conclusion)
    return metrics


def run_real_placeholder(data_root: Path) -> None:
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {data_root}. Mount or upload a small real "
            "masked-recognition subset, then rerun without --smoke."
        )
    raise NotImplementedError(
        "Real-data CLI runner is intentionally pending. Use notebooks/"
        "validation_spike.ipynb for the full FaceNet-based first pass."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/content/pku_masked_face_subset"))
    parser.add_argument("--results-dir", type=Path, default=Path("/content/masked_face_spike_results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Run a synthetic metrics smoke test.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        metrics = run_smoke(args.results_dir, args.seed)
        print(metrics.to_string(index=False))
        print(f"wrote {args.results_dir / 'validation_results_smoke.csv'}")
        print(f"wrote {args.results_dir / 'validation_conclusion_smoke.md'}")
    else:
        run_real_placeholder(args.data_root)


if __name__ == "__main__":
    main()
