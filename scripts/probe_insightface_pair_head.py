#!/usr/bin/env python3
"""Compare InsightFace/ArcFace embeddings against FaceNet pair-head results."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from probe_frozen_adapter import FaceRecord, Pair, best_threshold, cosine, sample_pairs, select_records
from probe_pair_verifier_head import PairVerifierHead, case_one_hot, predict_scores, stable_split, standardize, summarize, train_pair_head


MODEL_RAW = "insightface_arcface_raw"
MODEL_PAIR = "insightface_arcface_pair_head_all_cases"
MODEL_PAIR_MASKED_ONLY = "insightface_arcface_pair_head_masked_only"


def ensure_imports() -> None:
    try:
        import insightface  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing insightface. In Colab run: python -m pip install -q insightface onnxruntime-gpu"
        ) from exc


def load_app(model_name: str, providers: Sequence[str], det_size: int):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name=model_name, providers=list(providers))
    app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1, det_size=(det_size, det_size))
    return app


def largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])))


def embed_records(records: Sequence[FaceRecord], model_name: str, det_size: int, providers: Sequence[str]) -> dict[Path, np.ndarray]:
    app = load_app(model_name, providers, det_size)
    embeddings: dict[Path, np.ndarray] = {}
    failures = 0
    for record in tqdm(records, desc=f"Embedding {model_name}"):
        try:
            img = cv2.imread(str(record.path))
            if img is None:
                face = None
            else:
                face = largest_face(app.get(img))
        except Exception:
            face = None
        if face is None:
            failures += 1
            continue
        vector = np.asarray(face.normed_embedding, dtype=np.float32)
        norm = np.linalg.norm(vector)
        embeddings[record.path] = vector / norm if norm else vector
    print({"model": model_name, "embedded": len(embeddings), "failures": failures})
    return embeddings


def pair_features(pair: Pair, embeddings: dict[Path, np.ndarray]) -> np.ndarray | None:
    if pair.left not in embeddings or pair.right not in embeddings:
        return None
    left = embeddings[pair.left].astype(np.float32)
    right = embeddings[pair.right].astype(np.float32)
    score = np.array([cosine(left, right)], dtype=np.float32)
    return np.concatenate([case_one_hot(pair.case), score, np.abs(left - right), left * right]).astype(np.float32)


def feature_matrix(pairs: Sequence[Pair], embeddings: dict[Path, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[Pair]]:
    features = []
    labels = []
    kept = []
    for pair in pairs:
        vector = pair_features(pair, embeddings)
        if vector is None:
            continue
        features.append(vector)
        labels.append(pair.label)
        kept.append(pair)
    if not features:
        raise ValueError("No InsightFace pairs survived feature construction")
    return np.stack(features), np.array(labels, dtype=np.float32), kept


def raw_score(pair: Pair, embeddings: dict[Path, np.ndarray]) -> float:
    return cosine(embeddings[pair.left], embeddings[pair.right])


def score_rows(seed: int, pairs: Sequence[Pair], embeddings: dict[Path, np.ndarray], head_scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for idx, pair in enumerate(pairs):
        if pair.left not in embeddings or pair.right not in embeddings:
            continue
        rows.append({**asdict(pair), "left": str(pair.left), "right": str(pair.right), "seed": seed, "model": MODEL_RAW, "score": raw_score(pair, embeddings)})
        rows.append({**asdict(pair), "left": str(pair.left), "right": str(pair.right), "seed": seed, "model": MODEL_PAIR, "score": float(head_scores[idx])})
        masked_only_score = raw_score(pair, embeddings) if pair.case == "unmasked-unmasked" else float(head_scores[idx])
        rows.append({**asdict(pair), "left": str(pair.left), "right": str(pair.right), "seed": seed, "model": MODEL_PAIR_MASKED_ONLY, "score": masked_only_score})
    return pd.DataFrame(rows)


def write_readme(out_dir: Path, args: argparse.Namespace, summary: dict[str, object]) -> None:
    text = f"""# InsightFace / ArcFace Pair-Head Comparison

This artifact compares a stronger public recognizer family against the FaceNet
pair-head result.

Configuration:

- Data root: `{args.data_root}`
- InsightFace model: `{args.model_name}`
- Seeds: `{args.seeds}`
- Train identities per seed: {args.train_identities}
- Eval identities per seed: {args.eval_identities}
- Max images per condition: {args.max_images_per_condition}
- Pair-head epochs: {args.epochs}

Masked-unmasked mean ROC-AUC:

- InsightFace raw cosine: {summary["raw_mu_mean"]:.4f}
- InsightFace pair head, masked-only policy: {summary["pair_mu_mean"]:.4f}
- Gain: {summary["pair_mu_gain"]:.4f}

Interpretation: this comparison tests whether the project claim survives a
stronger off-the-shelf recognizer. The pair head here uses only full-image
InsightFace embeddings, not the five FaceNet occlusion views.
"""
    (out_dir / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--model-name", default="buffalo_l")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--providers", nargs="+", default=["CUDAExecutionProvider", "CPUExecutionProvider"])
    parser.add_argument("--train-identities", type=int, default=300)
    parser.add_argument("--eval-identities", type=int, default=100)
    parser.add_argument("--max-images-per-condition", type=int, default=8)
    parser.add_argument("--train-pairs-per-case", type=int, default=10000)
    parser.add_argument("--eval-pairs-per-case", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    args = parser.parse_args()

    ensure_imports()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_scores = []
    curves = []
    run_summaries = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        records = train_records + eval_records
        embeddings = embed_records(records, args.model_name, args.det_size, args.providers)
        train_pairs = sample_pairs(train_records, args.train_pairs_per_case, seed + 17)
        eval_pairs = sample_pairs(eval_records, args.eval_pairs_per_case, seed)
        train_x_all, train_y_all, kept_train_pairs = feature_matrix(train_pairs, embeddings)
        eval_x, _eval_y, kept_eval_pairs = feature_matrix(eval_pairs, embeddings)
        cal_mask = stable_split(kept_train_pairs, calibration_fraction=0.2, seed=seed + 99)
        fit_x = train_x_all[~cal_mask]
        fit_y = train_y_all[~cal_mask]
        cal_x = train_x_all[cal_mask]
        cal_y = train_y_all[cal_mask]
        fit_x, standardized, _mean, _std = standardize(fit_x, cal_x, eval_x)
        cal_x, eval_x = standardized
        print({"seed": seed, "fit_pairs": len(fit_x), "calibration_pairs": len(cal_x), "eval_pairs": len(eval_x), "feature_dim": fit_x.shape[1]})
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
        head_scores = predict_scores(head, eval_x, batch_size=args.batch_size, device=device)
        scores = score_rows(seed, kept_eval_pairs, embeddings, head_scores)
        all_scores.append(scores)
        curve.insert(0, "seed", seed)
        curves.append(curve)
        run_summaries.append(
            {
                "seed": seed,
                "train_identities": len(train_ids),
                "eval_identities": len(eval_ids),
                "train_images": len(train_records),
                "eval_images": len(eval_records),
                "embedded_images": len(embeddings),
                "embedding_failures": len(records) - len(embeddings),
            }
        )

    scores_df = pd.concat(all_scores, ignore_index=True)
    metrics = summarize(scores_df)
    rows = []
    for (model, case), df in metrics.groupby(["model", "case"]):
        values = df["roc_auc"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "case": case,
                "seeds": len(values),
                "roc_auc_mean": float(np.mean(values)),
                "roc_auc_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "roc_auc_min": float(np.min(values)),
                "roc_auc_max": float(np.max(values)),
            }
        )
    aggregate = pd.DataFrame(rows).sort_values(["case", "roc_auc_mean"], ascending=[True, False])
    curves_df = pd.concat(curves, ignore_index=True)

    scores_df.to_csv(args.out_dir / "insightface_pair_scores.csv", index=False)
    metrics.to_csv(args.out_dir / "insightface_metrics.csv", index=False)
    aggregate.to_csv(args.out_dir / "insightface_aggregate.csv", index=False)
    curves_df.to_csv(args.out_dir / "insightface_pair_head_training_curves.csv", index=False)
    pd.DataFrame(run_summaries).to_csv(args.out_dir / "run_summaries.csv", index=False)

    def agg_auc(model: str, case: str) -> float:
        row = aggregate[(aggregate["model"] == model) & (aggregate["case"] == case)]
        return float(row.iloc[0].roc_auc_mean) if len(row) else math.nan

    summary = {
        "device": device,
        "model_name": args.model_name,
        "seeds": args.seeds,
        "raw_mu_mean": agg_auc(MODEL_RAW, "masked-unmasked"),
        "pair_mu_mean": agg_auc(MODEL_PAIR_MASKED_ONLY, "masked-unmasked"),
        "pair_mu_gain": agg_auc(MODEL_PAIR_MASKED_ONLY, "masked-unmasked") - agg_auc(MODEL_RAW, "masked-unmasked"),
        "runs": run_summaries,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_readme(args.out_dir, args, summary)
    print(json.dumps(summary, indent=2))
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
