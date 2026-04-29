#!/usr/bin/env python3
"""Train a frozen-recognizer pair verifier head for masked verification.

FaceNet remains frozen. The trainable component is a small MLP over pair-level
features built from multiple occlusion-view embeddings.
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm

from probe_frozen_adapter import FaceRecord, Pair, best_threshold, cosine, sample_pairs, select_records
from probe_occlusion_ensemble import RAW_VIEWS, compute_embeddings


MODEL_BASELINE = "baseline_full"
MODEL_PAIR_HEAD = "pair_head_all_cases"
MODEL_PAIR_HEAD_MASKED_ONLY = "pair_head_masked_cases_only"
MODEL_UPPER_GATE = "gated_upper_only"
MODEL_BLUR_GATE = "gated_lower_blur"
MODEL_BLACKOUT_GATE = "gated_lower_blackout"


class PairVerifierHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def case_one_hot(case: str) -> np.ndarray:
    return np.array([case == "masked-masked", case == "masked-unmasked", case == "unmasked-unmasked"], dtype=np.float32)


def pair_features(pair: Pair, embeddings: dict[Path, dict[str, np.ndarray]]) -> np.ndarray | None:
    if pair.left not in embeddings or pair.right not in embeddings:
        return None
    left = embeddings[pair.left]
    right = embeddings[pair.right]
    scores = np.array([cosine(left[view], right[view]) for view in RAW_VIEWS], dtype=np.float32)
    score_stats = np.array(
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
    dense_parts = []
    for view in RAW_VIEWS:
        l_vec = left[view].astype(np.float32)
        r_vec = right[view].astype(np.float32)
        dense_parts.extend([np.abs(l_vec - r_vec), l_vec * r_vec])
    return np.concatenate([case_one_hot(pair.case), scores, score_stats, *dense_parts]).astype(np.float32)


def feature_matrix(
    pairs: Sequence[Pair],
    embeddings: dict[Path, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, list[Pair]]:
    features = []
    labels = []
    kept_pairs = []
    for pair in pairs:
        vector = pair_features(pair, embeddings)
        if vector is None:
            continue
        features.append(vector)
        labels.append(pair.label)
        kept_pairs.append(pair)
    if not features:
        raise ValueError("No pairs survived feature construction")
    return np.stack(features), np.array(labels, dtype=np.float32), kept_pairs


def stable_split(pairs: Sequence[Pair], calibration_fraction: float, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    values = np.array([rng.random() for _ in pairs])
    return values < calibration_fraction


def standardize(train_x: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return (train_x - mean) / std, [(x - mean) / std for x in others], mean.squeeze(0), std.squeeze(0)


def train_pair_head(
    train_x: np.ndarray,
    train_y: np.ndarray,
    cal_x: np.ndarray,
    cal_y: np.ndarray,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> tuple[PairVerifierHead, pd.DataFrame]:
    torch.manual_seed(seed)
    model = PairVerifierHead(train_x.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = torch.tensor([(len(train_y) - train_y.sum()) / max(train_y.sum(), 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_x_t = torch.tensor(train_x, dtype=torch.float32)
    train_y_t = torch.tensor(train_y, dtype=torch.float32)
    cal_x_t = torch.tensor(cal_x, dtype=torch.float32, device=device)
    rows = []
    best_auc = -math.inf
    best_state = None
    rng = np.random.default_rng(seed)

    for epoch in tqdm(range(1, epochs + 1), desc="Training pair verifier head"):
        model.train()
        order = rng.permutation(len(train_x_t))
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            x = train_x_t[idx].to(device)
            y = train_y_t[idx].to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.inference_mode():
            cal_scores = torch.sigmoid(model(cal_x_t)).detach().cpu().numpy()
        cal_auc = float(roc_auc_score(cal_y, cal_scores)) if len(np.unique(cal_y)) == 2 else math.nan
        if cal_auc > best_auc:
            best_auc = cal_auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        rows.append({"epoch": epoch, "loss": float(np.mean(losses)), "calibration_auc": cal_auc})

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), pd.DataFrame(rows)


@torch.inference_mode()
def predict_scores(model: PairVerifierHead, x: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    out = []
    for start in range(0, len(x), batch_size):
        batch = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
        out.append(torch.sigmoid(model(batch)).detach().cpu().numpy())
    return np.concatenate(out)


def raw_policy_score(pair: Pair, embeddings: dict[Path, dict[str, np.ndarray]], model: str) -> float:
    left = embeddings[pair.left]
    right = embeddings[pair.right]
    if model == MODEL_BASELINE:
        return cosine(left["full"], right["full"])
    if model == MODEL_UPPER_GATE:
        view = "full" if pair.case == "unmasked-unmasked" else "upper_only"
        return cosine(left[view], right[view])
    if model == MODEL_BLUR_GATE:
        view = "full" if pair.case == "unmasked-unmasked" else "lower_blur"
        return cosine(left[view], right[view])
    if model == MODEL_BLACKOUT_GATE:
        view = "full" if pair.case == "unmasked-unmasked" else "lower_blackout"
        return cosine(left[view], right[view])
    raise KeyError(model)


def score_eval_pairs(
    pairs: Sequence[Pair],
    embeddings: dict[Path, dict[str, np.ndarray]],
    head_scores: np.ndarray,
) -> pd.DataFrame:
    rows = []
    raw_models = [MODEL_BASELINE, MODEL_UPPER_GATE, MODEL_BLUR_GATE, MODEL_BLACKOUT_GATE]
    for idx, pair in enumerate(pairs):
        for model in raw_models:
            rows.append({**asdict(pair), "left": str(pair.left), "right": str(pair.right), "model": model, "score": raw_policy_score(pair, embeddings, model)})
        rows.append({**asdict(pair), "left": str(pair.left), "right": str(pair.right), "model": MODEL_PAIR_HEAD, "score": float(head_scores[idx])})
        masked_only_score = raw_policy_score(pair, embeddings, MODEL_BASELINE) if pair.case == "unmasked-unmasked" else float(head_scores[idx])
        rows.append(
            {
                **asdict(pair),
                "left": str(pair.left),
                "right": str(pair.right),
                "model": MODEL_PAIR_HEAD_MASKED_ONLY,
                "score": masked_only_score,
            }
        )
    return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, case), case_df in scores.groupby(["model", "case"]):
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


def write_conclusion(metrics: pd.DataFrame, out_dir: Path) -> None:
    def auc(model: str, case: str) -> float:
        row = metrics[(metrics["model"] == model) & (metrics["case"] == case)]
        return float(row.iloc[0].roc_auc) if len(row) else math.nan

    baseline_mu = auc(MODEL_BASELINE, "masked-unmasked")
    baseline_uu = auc(MODEL_BASELINE, "unmasked-unmasked")
    candidates = [MODEL_PAIR_HEAD, MODEL_PAIR_HEAD_MASKED_ONLY, MODEL_UPPER_GATE, MODEL_BLUR_GATE, MODEL_BLACKOUT_GATE]
    best = max(candidates, key=lambda model: (auc(model, "masked-unmasked"), auc(model, "unmasked-unmasked")))
    best_mu = auc(best, "masked-unmasked")
    best_uu = auc(best, "unmasked-unmasked")
    verdict = "PROMISING" if best_mu > baseline_mu and (math.isnan(best_uu) or baseline_uu - best_uu <= 0.03) else "NOT YET PROMISING"
    text = f"""# Pair Verifier Head Probe Conclusion

Recommendation: {verdict}

- Baseline full FaceNet masked-unmasked ROC-AUC: {baseline_mu:.4f}
- Best candidate: {best}
- Best masked-unmasked ROC-AUC: {best_mu:.4f}
- Best gain vs baseline: {best_mu - baseline_mu:.4f}
- Baseline unmasked-unmasked ROC-AUC: {baseline_uu:.4f}
- Best unmasked-unmasked ROC-AUC: {best_uu:.4f}
- Best unmasked regression vs baseline: {baseline_uu - best_uu:.4f}

This probe freezes FaceNet and trains only a pair-level MLP verifier head over
multi-view embedding features.
"""
    (out_dir / "pair_verifier_head_conclusion.md").write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-identities", type=int, default=140)
    parser.add_argument("--eval-identities", type=int, default=80)
    parser.add_argument("--max-images-per-condition", type=int, default=8)
    parser.add_argument("--train-pairs-per-case", type=int, default=3000)
    parser.add_argument("--eval-pairs-per-case", type=int, default=800)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
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
    train_pairs = sample_pairs(train_records, args.train_pairs_per_case, args.seed + 17)
    eval_pairs = sample_pairs(eval_records, args.eval_pairs_per_case, args.seed)

    train_x_all, train_y_all, kept_train_pairs = feature_matrix(train_pairs, embeddings)
    eval_x, eval_y, kept_eval_pairs = feature_matrix(eval_pairs, embeddings)
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

    scores.to_csv(args.out_dir / "pair_verifier_head_pair_scores.csv", index=False)
    metrics.to_csv(args.out_dir / "pair_verifier_head_results.csv", index=False)
    curve.to_csv(args.out_dir / "pair_verifier_head_training_curve.csv", index=False)
    np.savez(args.out_dir / "pair_verifier_head_standardizer.npz", mean=mean, std=std)
    pd.DataFrame({"identity": train_ids, "split": "train"}).to_csv(args.out_dir / "train_identities.csv", index=False)
    pd.DataFrame({"identity": eval_ids, "split": "eval"}).to_csv(args.out_dir / "eval_identities.csv", index=False)
    print(metrics.to_string(index=False))
    write_conclusion(metrics, args.out_dir)


if __name__ == "__main__":
    main()
