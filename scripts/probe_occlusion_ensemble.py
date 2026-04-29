#!/usr/bin/env python3
"""Probe test-time occlusion ensembles for masked face verification.

This keeps FaceNet frozen and treats aligned-face occlusion views as a
test-time evidence set. It evaluates raw views, mask-presence gates, consensus
policies, and reliability/abstention metrics on held-out identities.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm

from probe_frozen_adapter import CASES, FaceRecord, Pair, best_threshold, cosine, sample_pairs, select_records


RAW_VIEWS = ["full", "lower_blackout", "lower_blur", "upper_only", "eye_band"]
SAFE_VIEWS = ["full", "lower_blackout", "lower_blur", "upper_only", "eye_band"]
MASKED_POLICIES = [
    "same_full",
    "same_lower_blackout",
    "same_lower_blur",
    "same_upper_only",
    "same_eye_band",
    "gated_lower_blackout",
    "gated_lower_blur",
    "gated_upper_only",
    "gated_eye_band",
    "ensemble_mean_safe",
    "ensemble_top1_safe",
    "ensemble_top2_safe",
    "ensemble_top3_safe",
    "ensemble_disagreement_penalty_0.10",
    "ensemble_disagreement_penalty_0.25",
    "ensemble_disagreement_penalty_0.50",
]


def lower_blackout(face_tensor: torch.Tensor) -> torch.Tensor:
    _, h, _ = face_tensor.shape
    out = face_tensor.clone()
    out[:, int(h * 0.55) :, :] = 0.0
    return out


def lower_blur(face_tensor: torch.Tensor) -> torch.Tensor:
    _, h, w = face_tensor.shape
    split = int(h * 0.55)
    out = face_tensor.clone()
    lower = out[:, split:, :].unsqueeze(0)
    small = F.interpolate(lower, size=(max(2, (h - split) // 8), max(2, w // 8)), mode="bilinear", align_corners=False)
    blurred = F.interpolate(small, size=(h - split, w), mode="bilinear", align_corners=False).squeeze(0)
    out[:, split:, :] = blurred
    return out


def upper_only(face_tensor: torch.Tensor) -> torch.Tensor:
    _, h, _ = face_tensor.shape
    out = face_tensor.clone()
    out[:, int(h * 0.60) :, :] = 0.0
    return out


def eye_band(face_tensor: torch.Tensor) -> torch.Tensor:
    _, h, _ = face_tensor.shape
    out = torch.zeros_like(face_tensor)
    start = int(h * 0.18)
    stop = int(h * 0.58)
    out[:, start:stop, :] = face_tensor[:, start:stop, :]
    return out


def make_views(face: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "full": face,
        "lower_blackout": lower_blackout(face),
        "lower_blur": lower_blur(face),
        "upper_only": upper_only(face),
        "eye_band": eye_band(face),
    }


class ViewEmbedder:
    def __init__(self, image_size: int, device: str):
        self.device = device
        self.mtcnn = MTCNN(image_size=image_size, margin=16, post_process=True, device=device)
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    @torch.inference_mode()
    def embed(self, path: Path) -> dict[str, np.ndarray] | None:
        try:
            img = Image.open(path).convert("RGB")
            face = self.mtcnn(img)
        except Exception:
            return None
        if face is None:
            return None
        views = make_views(face)
        names = list(views)
        batch = torch.stack([views[name] for name in names]).to(self.device)
        emb = self.model(batch).detach().cpu().numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-12)
        return dict(zip(names, emb, strict=True))


def compute_embeddings(records: Sequence[FaceRecord], image_size: int, device: str) -> dict[Path, dict[str, np.ndarray]]:
    embedder = ViewEmbedder(image_size=image_size, device=device)
    embeddings: dict[Path, dict[str, np.ndarray]] = {}
    failures = 0
    for record in tqdm(records, desc="Embedding occlusion views"):
        result = embedder.embed(record.path)
        if result is None:
            failures += 1
            continue
        embeddings[record.path] = result
    print({"embedded": len(embeddings), "failures": failures, "views": RAW_VIEWS})
    return embeddings


def same_view_scores(pair: Pair, embeddings: dict[Path, dict[str, np.ndarray]]) -> dict[str, float]:
    return {
        view: cosine(embeddings[pair.left][view], embeddings[pair.right][view])
        for view in RAW_VIEWS
    }


def policy_score(policy: str, pair: Pair, view_scores: dict[str, float]) -> tuple[float, float]:
    safe = np.array([view_scores[view] for view in SAFE_VIEWS], dtype=np.float64)
    safe_std = float(np.std(safe))
    reliability = -safe_std if pair.case != "unmasked-unmasked" else 1.0

    if policy.startswith("same_"):
        return view_scores[policy.removeprefix("same_")], reliability
    if policy.startswith("gated_"):
        view = policy.removeprefix("gated_")
        return (view_scores["full"] if pair.case == "unmasked-unmasked" else view_scores[view]), reliability
    if policy == "ensemble_mean_safe":
        score = view_scores["full"] if pair.case == "unmasked-unmasked" else float(np.mean(safe))
        return score, reliability
    if policy.startswith("ensemble_top"):
        if pair.case == "unmasked-unmasked":
            return view_scores["full"], reliability
        k = int(policy.removeprefix("ensemble_top").removesuffix("_safe"))
        return float(np.mean(np.sort(safe)[-k:])), reliability
    if policy.startswith("ensemble_disagreement_penalty_"):
        if pair.case == "unmasked-unmasked":
            return view_scores["full"], reliability
        penalty = float(policy.rsplit("_", 1)[1])
        return float(np.mean(safe) - penalty * safe_std), reliability
    raise KeyError(policy)


def score_pairs(
    pairs: Sequence[Pair],
    split: str,
    embeddings: dict[Path, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    skipped = 0
    for pair in pairs:
        if pair.left not in embeddings or pair.right not in embeddings:
            skipped += 1
            continue
        view_scores = same_view_scores(pair, embeddings)
        for policy in MASKED_POLICIES:
            score, reliability = policy_score(policy, pair, view_scores)
            rows.append(
                {
                    **asdict(pair),
                    "left": str(pair.left),
                    "right": str(pair.right),
                    "split": split,
                    "model": policy,
                    "score": score,
                    "reliability": reliability,
                    "view_std": -reliability if pair.case != "unmasked-unmasked" else 0.0,
                    "skipped_so_far": skipped,
                    **{f"score_{view}": view_scores[view] for view in RAW_VIEWS},
                }
            )
    return pd.DataFrame(rows)


def eval_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eval_rows = scores[scores["split"] == "eval"]
    for (model, case), case_df in eval_rows.groupby(["model", "case"]):
        labels = case_df["label"].to_numpy()
        values = case_df["score"].to_numpy()
        rows.append(
            {
                "model": model,
                "case": case,
                "pairs": len(case_df),
                "roc_auc": float(roc_auc_score(labels, values)) if len(np.unique(labels)) == 2 else math.nan,
                **best_threshold(labels, values),
            }
        )
    return pd.DataFrame(rows).sort_values(["case", "model"])


def calibration_thresholds(scores: pd.DataFrame) -> dict[tuple[str, str], float]:
    thresholds = {}
    calibration = scores[scores["split"] == "calibration"]
    for (model, case), case_df in calibration.groupby(["model", "case"]):
        labels = case_df["label"].to_numpy()
        values = case_df["score"].to_numpy()
        if len(case_df) and len(np.unique(labels)) == 2:
            thresholds[(model, case)] = best_threshold(labels, values)["threshold"]
    return thresholds


def selective_metrics(scores: pd.DataFrame, coverages: Sequence[float]) -> pd.DataFrame:
    rows = []
    thresholds = calibration_thresholds(scores)
    eval_rows = scores[scores["split"] == "eval"]
    for (model, case), case_df in eval_rows.groupby(["model", "case"]):
        ranked = case_df.sort_values("reliability", ascending=False)
        threshold = thresholds.get((model, case), float(ranked["score"].median()))
        for coverage in coverages:
            keep_n = max(2, int(round(len(ranked) * coverage)))
            kept = ranked.head(keep_n)
            labels = kept["label"].to_numpy()
            values = kept["score"].to_numpy()
            preds = (values >= threshold).astype(int)
            rows.append(
                {
                    "model": model,
                    "case": case,
                    "coverage": coverage,
                    "pairs": len(kept),
                    "roc_auc": float(roc_auc_score(labels, values)) if len(np.unique(labels)) == 2 else math.nan,
                    "accuracy_at_calibration_threshold": float(accuracy_score(labels, preds)),
                    "threshold": threshold,
                    "mean_view_std": float(kept["view_std"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["case", "model", "coverage"])


def write_conclusion(metrics: pd.DataFrame, selective: pd.DataFrame, out_dir: Path) -> None:
    def auc(model: str, case: str) -> float:
        row = metrics[(metrics["model"] == model) & (metrics["case"] == case)]
        return float(row.iloc[0].roc_auc) if len(row) else math.nan

    baseline_mu = auc("same_full", "masked-unmasked")
    backup_mu = auc("gated_lower_blackout", "masked-unmasked")
    baseline_uu = auc("same_full", "unmasked-unmasked")
    candidates = [m for m in metrics["model"].unique() if m.startswith("ensemble_")]
    best = max(candidates, key=lambda model: auc(model, "masked-unmasked"))
    best_mu = auc(best, "masked-unmasked")
    best_uu = auc(best, "unmasked-unmasked")

    selective_mu = selective[
        (selective["case"] == "masked-unmasked")
        & (selective["coverage"] == 0.8)
    ]
    best_selective = "none"
    best_selective_auc = math.nan
    if len(selective_mu):
        top = selective_mu.sort_values("roc_auc", ascending=False).iloc[0]
        best_selective = str(top.model)
        best_selective_auc = float(top.roc_auc)

    verdict = "FULL-COVERAGE PROMISING" if best_mu > baseline_mu else "NO FULL-COVERAGE GAIN"
    text = f"""# Occlusion Ensemble Probe Conclusion

Recommendation: {verdict}

- Baseline full FaceNet masked-unmasked ROC-AUC: {baseline_mu:.4f}
- Backup gated-blackout masked-unmasked ROC-AUC: {backup_mu:.4f}
- Best ensemble on masked-unmasked: {best}
- Best ensemble masked-unmasked ROC-AUC: {best_mu:.4f}
- Best ensemble gain vs baseline: {best_mu - baseline_mu:.4f}
- Best ensemble gain vs backup: {best_mu - backup_mu:.4f}
- Baseline unmasked-unmasked ROC-AUC: {baseline_uu:.4f}
- Best ensemble unmasked-unmasked ROC-AUC: {best_uu:.4f}
- Best selective policy at 80% masked-unmasked coverage: {best_selective}
- Best selective 80% masked-unmasked ROC-AUC: {best_selective_auc:.4f}

This probe keeps FaceNet frozen and evaluates test-time occlusion evidence:
full face, lower blackout, lower blur, upper-only, and eye-band views.
Reliability is estimated from disagreement across views.
"""
    (out_dir / "occlusion_ensemble_conclusion.md").write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-identities", type=int, default=140)
    parser.add_argument("--eval-identities", type=int, default=80)
    parser.add_argument("--max-images-per-condition", type=int, default=8)
    parser.add_argument("--pairs-per-case", type=int, default=400)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print({"device": device, "data_root": str(args.data_root)})

    train_records, eval_records, train_ids, eval_ids = select_records(
        args.data_root,
        args.train_identities,
        args.eval_identities,
        args.max_images_per_condition,
        args.seed,
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
    train_pairs = sample_pairs(train_records, args.pairs_per_case, args.seed + 10_000)
    eval_pairs = sample_pairs(eval_records, args.pairs_per_case, args.seed)
    scores = pd.concat(
        [
            score_pairs(train_pairs, "calibration", embeddings),
            score_pairs(eval_pairs, "eval", embeddings),
        ],
        ignore_index=True,
    )
    metrics = eval_metrics(scores)
    selective = selective_metrics(scores, coverages=[1.0, 0.8, 0.6])

    scores.to_csv(args.out_dir / "occlusion_ensemble_pair_scores.csv", index=False)
    metrics.to_csv(args.out_dir / "occlusion_ensemble_results.csv", index=False)
    selective.to_csv(args.out_dir / "occlusion_ensemble_selective_results.csv", index=False)
    pd.DataFrame({"identity": train_ids, "split": "train"}).to_csv(args.out_dir / "train_identities.csv", index=False)
    pd.DataFrame({"identity": eval_ids, "split": "eval"}).to_csv(args.out_dir / "eval_identities.csv", index=False)
    print(metrics.to_string(index=False))
    print(selective.to_string(index=False))
    write_conclusion(metrics, selective, args.out_dir)


if __name__ == "__main__":
    main()
