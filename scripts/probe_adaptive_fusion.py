#!/usr/bin/env python3
"""Probe adaptive score fusion on saved validation pair scores.

This script does not recompute embeddings. It takes the long-form
`validation_pair_scores.csv` produced by the notebook, reconstructs the same
stable calibration/eval split, trains small logistic-regression fusion policies
on calibration pairs, and evaluates them on held-out pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


KEY_COLS = ["case", "label", "left", "right", "left_id", "right_id"]
BASELINE = "baseline_facenet_vggface2"
UPPER = "upper_face_facenet_vggface2"
BLACKOUT = "lower_blackout_facenet_vggface2"
BLUR = "lower_blur_facenet_vggface2"
RAW_MODELS = [BASELINE, UPPER, BLACKOUT, BLUR]
FEATURE_MODELS = [BASELINE, UPPER, BLACKOUT, BLUR]
CASES = ["masked-masked", "masked-unmasked", "unmasked-unmasked"]


def stable_fraction(*parts: object) -> float:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16**12)


def add_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = df.copy()
    out["split"] = [
        "calibration" if stable_fraction(row.left, row.right, row.case, seed) < 0.5 else "eval"
        for row in out.itertuples()
    ]
    return out


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    best = {"accuracy": -1.0, "threshold": float(scores.min()), "far": math.nan, "frr": math.nan}
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


def summarize(score_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, case), case_df in score_df.groupby(["model", "case"]):
        labels = case_df["label"].to_numpy()
        scores = case_df["score"].to_numpy()
        rows.append(
            {
                "model": model,
                "case": case,
                "pairs": len(case_df),
                "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else math.nan,
                **best_threshold(labels, scores),
            }
        )
    return pd.DataFrame(rows).sort_values(["case", "model"])


def make_wide(scores: pd.DataFrame, seed: int) -> pd.DataFrame:
    wide = scores.pivot_table(index=KEY_COLS, columns="model", values="score").reset_index()
    required = set(FEATURE_MODELS)
    missing = sorted(required - set(wide.columns))
    if missing:
        raise SystemExit(f"Missing required model score columns: {missing}")
    wide = wide.dropna(subset=FEATURE_MODELS).copy()
    return add_split(wide, seed)


def feature_frame(wide: pd.DataFrame, include_case: bool) -> pd.DataFrame:
    features = wide[FEATURE_MODELS].copy()
    features["full_minus_upper"] = wide[BASELINE] - wide[UPPER]
    features["full_minus_blackout"] = wide[BASELINE] - wide[BLACKOUT]
    features["full_minus_blur"] = wide[BASELINE] - wide[BLUR]
    features["blackout_minus_upper"] = wide[BLACKOUT] - wide[UPPER]
    features["blur_minus_upper"] = wide[BLUR] - wide[UPPER]
    if include_case:
        case_features = pd.get_dummies(wide["case"], prefix="case")
        for case in CASES:
            col = f"case_{case}"
            if col not in case_features:
                case_features[col] = 0
        features = pd.concat([features, case_features[[f"case_{case}" for case in CASES]]], axis=1)
    return features


def train_logreg(train: pd.DataFrame, include_case: bool) -> object | None:
    if train.empty or len(train["label"].unique()) < 2:
        return None
    x_train = feature_frame(train, include_case=include_case)
    y_train = train["label"].astype(int)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=0),
    )
    model.fit(x_train, y_train)
    return model


def predict_policy(model: object, rows: pd.DataFrame, name: str, include_case: bool) -> pd.DataFrame:
    x_eval = feature_frame(rows, include_case=include_case)
    out = rows[KEY_COLS].copy()
    out["model"] = name
    out["score"] = model.predict_proba(x_eval)[:, 1]
    return out


def build_adaptive_scores(wide: pd.DataFrame) -> pd.DataFrame:
    calibration = wide[wide["split"] == "calibration"].copy()
    eval_rows = wide[wide["split"] == "eval"].copy()
    outputs = []

    global_model = train_logreg(calibration, include_case=True)
    if global_model is not None:
        outputs.append(predict_policy(global_model, eval_rows, "adaptive_logreg_global", include_case=True))

    masked_unmasked_train = calibration[calibration["case"] == "masked-unmasked"].copy()
    masked_unmasked_eval = eval_rows[eval_rows["case"] == "masked-unmasked"].copy()
    masked_unmasked_model = train_logreg(masked_unmasked_train, include_case=False)
    if masked_unmasked_model is not None and not masked_unmasked_eval.empty:
        outputs.append(
            predict_policy(
                masked_unmasked_model,
                masked_unmasked_eval,
                "adaptive_logreg_masked_unmasked_only",
                include_case=False,
            )
        )

    per_case_outputs = []
    for case in CASES:
        case_train = calibration[calibration["case"] == case].copy()
        case_eval = eval_rows[eval_rows["case"] == case].copy()
        case_model = train_logreg(case_train, include_case=False)
        if case_model is not None and not case_eval.empty:
            per_case_outputs.append(predict_policy(case_model, case_eval, "adaptive_logreg_per_case", include_case=False))
    if per_case_outputs:
        outputs.append(pd.concat(per_case_outputs, ignore_index=True))

    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def model_auc(rows: pd.DataFrame, model: str) -> float:
    labels = rows["label"].to_numpy()
    scores = rows[model].to_numpy()
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else -math.inf


def build_gated_scores(wide: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    calibration = wide[wide["split"] == "calibration"].copy()
    eval_rows = wide[wide["split"] == "eval"].copy()
    outputs = []
    selections: dict[str, str] = {}

    per_case = []
    for case in CASES:
        case_train = calibration[calibration["case"] == case]
        case_eval = eval_rows[eval_rows["case"] == case]
        if case_train.empty or case_eval.empty:
            continue
        best_model = max(RAW_MODELS, key=lambda model: model_auc(case_train, model))
        selections[case] = best_model
        out = case_eval[KEY_COLS].copy()
        out["model"] = "case_gated_best_raw"
        out["score"] = case_eval[best_model]
        per_case.append(out)
    if per_case:
        outputs.append(pd.concat(per_case, ignore_index=True))

    for gate_name, masked_model in [
        ("mask_presence_gated_blackout", BLACKOUT),
        ("mask_presence_gated_blur", BLUR),
        ("mask_presence_gated_upper", UPPER),
    ]:
        out = eval_rows[KEY_COLS].copy()
        out["model"] = gate_name
        out["score"] = np.where(eval_rows["case"] == "unmasked-unmasked", eval_rows[BASELINE], eval_rows[masked_model])
        outputs.append(out)

    return (pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(), selections)


def build_eval_raw_scores(wide: pd.DataFrame) -> pd.DataFrame:
    eval_rows = wide[wide["split"] == "eval"].copy()
    outputs = []
    for model in RAW_MODELS:
        out = eval_rows[KEY_COLS].copy()
        out["model"] = model
        out["score"] = eval_rows[model]
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True)


def conclusion(metrics: pd.DataFrame) -> str:
    def auc(model: str, case: str) -> float:
        row = metrics[(metrics["model"] == model) & (metrics["case"] == case)]
        return float(row.iloc[0].roc_auc) if len(row) else math.nan

    baseline_mu = auc(BASELINE, "masked-unmasked")
    baseline_uu = auc(BASELINE, "unmasked-unmasked")
    policy_models = [
        m
        for m in metrics["model"].unique()
        if str(m).startswith("adaptive_") or str(m).startswith("case_gated_") or str(m).startswith("mask_presence_")
    ]
    primary = metrics[(metrics["case"] == "masked-unmasked") & (metrics["model"].isin(policy_models))]
    best_model = str(primary.sort_values("roc_auc", ascending=False).iloc[0].model) if len(primary) else "none"
    best_mu = auc(best_model, "masked-unmasked")
    best_uu = auc(best_model, "unmasked-unmasked")
    gain = best_mu - baseline_mu
    regression = baseline_uu - best_uu if not math.isnan(best_uu) else math.nan
    if math.isnan(best_uu):
        note = "Best adaptive model only reports the masked-unmasked case, so unmasked preservation must be checked with a global/per-case policy."
    else:
        note = "Best adaptive model includes unmasked-unmasked evaluation."
    verdict = "PROMISING SIGNAL" if gain > 0.0 else "NO POSITIVE SIGNAL"
    return f"""# Adaptive Fusion Probe Conclusion

Recommendation: {verdict}

- Best fusion policy on masked-unmasked: {best_model}
- Masked-unmasked ROC-AUC baseline: {baseline_mu:.4f}
- Masked-unmasked ROC-AUC best policy: {best_mu:.4f}
- Masked-unmasked gain vs baseline: {gain:.4f}
- Unmasked-unmasked ROC-AUC baseline: {baseline_uu:.4f}
- Unmasked-unmasked ROC-AUC best adaptive: {best_uu:.4f}
- Unmasked regression vs baseline: {regression:.4f}

Interpretation: this probe only tests whether a calibrated score policy has
signal using already-computed unmasked-recognizer variants. It still needs a
real mask-aware model for the final comparison.

{note}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.scores_csv)
    wide = make_wide(scores, args.seed)
    raw_scores = build_eval_raw_scores(wide)
    adaptive_scores = build_adaptive_scores(wide)
    gated_scores, selections = build_gated_scores(wide)
    all_scores = pd.concat([raw_scores, adaptive_scores, gated_scores], ignore_index=True)
    metrics = summarize(all_scores)

    all_scores.to_csv(args.out_dir / "adaptive_fusion_probe_pair_scores.csv", index=False)
    metrics.to_csv(args.out_dir / "adaptive_fusion_probe_results.csv", index=False)
    (args.out_dir / "case_gated_model_selection.json").write_text(json.dumps(selections, indent=2, sort_keys=True))
    text = conclusion(metrics)
    (args.out_dir / "adaptive_fusion_probe_conclusion.md").write_text(text)
    print(metrics.to_string(index=False))
    print(text)


if __name__ == "__main__":
    main()
