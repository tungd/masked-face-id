#!/usr/bin/env python3
"""Fine-tune a frozen-recognizer tail with an ArcFace identity objective."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from probe_frozen_adapter import FaceRecord, Pair, best_threshold, cosine, sample_pairs, select_records
from probe_training_adaptation import (
    align_faces,
    embed_tensors,
    records_by_identity,
    set_tail_train_mode,
    set_trainable_tail,
)


MODEL_BASELINE = "baseline_full"
MODEL_ARCFACE_FULL = "arcface_finetune_full"
MODEL_ARCFACE_MASKED_IMAGES = "arcface_finetune_masked_images_only"
MODEL_ARCFACE_MASKED_PAIRS = "arcface_finetune_masked_pairs_only"


class ArcMarginHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, scale: float, margin: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine_logits = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - cosine_logits.pow(2), min=1e-7))
        phi = cosine_logits * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine_logits > self.threshold, phi, cosine_logits - self.mm)
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).to(dtype=cosine_logits.dtype)
        logits = one_hot * phi + (1.0 - one_hot) * cosine_logits
        return logits * self.scale


def sample_arcface_batch(
    grouped: dict[str, dict[str, list[FaceRecord]]],
    identities: Sequence[str],
    identity_to_label: dict[str, int],
    identities_per_batch: int,
    samples_per_identity: int,
    rng: random.Random,
) -> tuple[list[FaceRecord], list[int]]:
    chosen = rng.sample(list(identities), min(identities_per_batch, len(identities)))
    batch_records: list[FaceRecord] = []
    labels: list[int] = []
    masked_count = max(1, samples_per_identity // 2)
    unmasked_count = max(1, samples_per_identity - masked_count)
    for identity in chosen:
        groups = grouped[identity]
        selected = [rng.choice(groups["masked"]) for _ in range(masked_count)]
        selected.extend(rng.choice(groups["unmasked"]) for _ in range(unmasked_count))
        batch_records.extend(selected)
        labels.extend([identity_to_label[identity]] * len(selected))
    return batch_records, labels


def train_arcface_tail(
    base_model: nn.Module,
    records: Sequence[FaceRecord],
    tensors: dict[Path, torch.Tensor],
    baseline_embeddings: dict[Path, np.ndarray],
    trainable_prefixes: Sequence[str],
    steps: int,
    identities_per_batch: int,
    samples_per_identity: int,
    lr: float,
    weight_decay: float,
    scale: float,
    margin: float,
    all_distill_weight: float,
    unmasked_distill_weight: float,
    seed: int,
    device: str,
) -> tuple[nn.Module, ArcMarginHead, pd.DataFrame]:
    grouped = records_by_identity(records, tensors)
    identities = sorted(grouped)
    if not identities:
        raise ValueError("No train identities with aligned masked and unmasked tensors")
    identity_to_label = {identity: idx for idx, identity in enumerate(identities)}

    trainable = set_trainable_tail(base_model, trainable_prefixes)
    if trainable == 0:
        raise ValueError(f"No trainable parameters matched prefixes: {trainable_prefixes}")
    print({"arcface_trainable_parameters": trainable, "classes": len(identities), "prefixes": list(trainable_prefixes)})

    head = ArcMarginHead(embedding_dim=512, num_classes=len(identities), scale=scale, margin=margin).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": (p for p in base_model.parameters() if p.requires_grad), "lr": lr},
            {"params": head.parameters(), "lr": lr * 10.0},
        ],
        weight_decay=weight_decay,
    )
    rng = random.Random(seed)
    rows = []

    for step in tqdm(range(1, steps + 1), desc="Fine-tuning with ArcFace"):
        set_tail_train_mode(base_model, trainable_prefixes)
        head.train()
        batch_records, labels_list = sample_arcface_batch(
            grouped,
            identities,
            identity_to_label,
            identities_per_batch=identities_per_batch,
            samples_per_identity=samples_per_identity,
            rng=rng,
        )
        batch = torch.stack([tensors[record.path] for record in batch_records]).to(device)
        labels = torch.tensor(labels_list, dtype=torch.long, device=device)
        embeddings = F.normalize(base_model(batch), dim=-1)
        logits = head(embeddings, labels)
        ce_loss = F.cross_entropy(logits, labels)

        frozen = torch.tensor(
            np.stack([baseline_embeddings[record.path] for record in batch_records]),
            dtype=torch.float32,
            device=device,
        )
        cosine_distance = 1.0 - (embeddings * frozen).sum(dim=1)
        all_distill_loss = cosine_distance.mean()
        unmasked_mask = torch.tensor(
            [record.condition == "unmasked" for record in batch_records],
            dtype=torch.bool,
            device=device,
        )
        unmasked_distill_loss = (
            cosine_distance[unmasked_mask].mean() if bool(unmasked_mask.any()) else embeddings.new_tensor(0.0)
        )
        loss = ce_loss + all_distill_weight * all_distill_loss + unmasked_distill_weight * unmasked_distill_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in base_model.parameters() if p.requires_grad), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
        optimizer.step()

        if step == 1 or step % max(1, steps // 20) == 0:
            rows.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "arcface_ce": float(ce_loss.detach().cpu()),
                    "all_distill": float(all_distill_loss.detach().cpu()),
                    "unmasked_distill": float(unmasked_distill_loss.detach().cpu()),
                }
            )

    base_model.eval()
    head.eval()
    return base_model, head, pd.DataFrame(rows)


def vector_for_model(
    pair: Pair,
    path: Path,
    condition: str,
    model: str,
    baseline: dict[Path, np.ndarray],
    finetuned: dict[Path, np.ndarray],
) -> np.ndarray | None:
    if model == MODEL_BASELINE:
        return baseline.get(path)
    if model == MODEL_ARCFACE_FULL:
        return finetuned.get(path)
    if model == MODEL_ARCFACE_MASKED_IMAGES:
        return finetuned.get(path) if condition == "masked" else baseline.get(path)
    if model == MODEL_ARCFACE_MASKED_PAIRS:
        return baseline.get(path) if pair.case == "unmasked-unmasked" else finetuned.get(path)
    raise KeyError(model)


def evaluate(
    pairs: Sequence[Pair],
    baseline: dict[Path, np.ndarray],
    finetuned: dict[Path, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = [MODEL_BASELINE, MODEL_ARCFACE_FULL, MODEL_ARCFACE_MASKED_IMAGES, MODEL_ARCFACE_MASKED_PAIRS]
    score_rows = []
    for model in models:
        skipped = 0
        for pair in pairs:
            left = vector_for_model(pair, pair.left, pair.left_condition, model, baseline, finetuned)
            right = vector_for_model(pair, pair.right, pair.right_condition, model, baseline, finetuned)
            if left is None or right is None:
                skipped += 1
                continue
            score_rows.append(
                {
                    "model": model,
                    "case": pair.case,
                    "label": pair.label,
                    "score": cosine(left, right),
                    "left": str(pair.left),
                    "right": str(pair.right),
                    "left_id": pair.left_id,
                    "right_id": pair.right_id,
                    "skipped_for_model": skipped,
                }
            )
    scores = pd.DataFrame(score_rows)
    metric_rows = []
    for (model, case), case_df in scores.groupby(["model", "case"]):
        labels = case_df["label"].to_numpy()
        values = case_df["score"].to_numpy()
        metric_rows.append(
            {
                "model": model,
                "case": case,
                "pairs": len(case_df),
                "roc_auc": float(roc_auc_score(labels, values)) if len(np.unique(labels)) == 2 else math.nan,
                **best_threshold(labels, values),
            }
        )
    return scores, pd.DataFrame(metric_rows).sort_values(["case", "model"])


def write_conclusion(metrics: pd.DataFrame, out_dir: Path) -> None:
    def auc(model: str, case: str) -> float:
        row = metrics[(metrics["model"] == model) & (metrics["case"] == case)]
        return float(row.iloc[0].roc_auc) if len(row) else math.nan

    baseline_mu = auc(MODEL_BASELINE, "masked-unmasked")
    baseline_uu = auc(MODEL_BASELINE, "unmasked-unmasked")
    candidates = [MODEL_ARCFACE_FULL, MODEL_ARCFACE_MASKED_IMAGES, MODEL_ARCFACE_MASKED_PAIRS]
    best = max(candidates, key=lambda model: (auc(model, "masked-unmasked"), auc(model, "unmasked-unmasked")))
    best_mu = auc(best, "masked-unmasked")
    best_uu = auc(best, "unmasked-unmasked")
    verdict = "PROMISING" if best_mu > baseline_mu and baseline_uu - best_uu <= 0.03 else "NOT YET PROMISING"
    text = f"""# ArcFace Fine-Tune Probe Conclusion

Recommendation: {verdict}

- Baseline full FaceNet masked-unmasked ROC-AUC: {baseline_mu:.4f}
- Best ArcFace candidate: {best}
- Best masked-unmasked ROC-AUC: {best_mu:.4f}
- Best gain vs baseline: {best_mu - baseline_mu:.4f}
- Baseline unmasked-unmasked ROC-AUC: {baseline_uu:.4f}
- Best unmasked-unmasked ROC-AUC: {best_uu:.4f}
- Best unmasked regression vs baseline: {baseline_uu - best_uu:.4f}

This probe fine-tunes the FaceNet tail with an ArcFace identity-classification
objective and frozen-embedding distillation. Evaluation identities are disjoint
from training identities.
"""
    (out_dir / "arcface_finetune_conclusion.md").write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-identities", type=int, default=140)
    parser.add_argument("--eval-identities", type=int, default=80)
    parser.add_argument("--max-images-per-condition", type=int, default=8)
    parser.add_argument("--pairs-per-case", type=int, default=800)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--identities-per-batch", type=int, default=16)
    parser.add_argument("--samples-per-identity", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scale", type=float, default=32.0)
    parser.add_argument("--margin", type=float, default=0.35)
    parser.add_argument("--all-distill-weight", type=float, default=0.05)
    parser.add_argument("--unmasked-distill-weight", type=float, default=1.0)
    parser.add_argument(
        "--trainable-prefixes",
        default="repeat_3,block8,last_linear,last_bn",
        help="Comma-separated FaceNet module prefixes to unfreeze.",
    )
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

    all_records = train_records + eval_records
    all_paths = [record.path for record in all_records]
    tensors = align_faces(all_records, image_size=args.image_size, device=device)

    frozen_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    baseline_embeddings = embed_tensors(
        frozen_model,
        tensors,
        all_paths,
        device=device,
        batch_size=args.embed_batch_size,
        desc="Embedding baseline",
    )

    trainable_prefixes = [part.strip() for part in args.trainable_prefixes.split(",") if part.strip()]
    finetuned_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    finetuned_model, _head, curve = train_arcface_tail(
        finetuned_model,
        train_records,
        tensors,
        baseline_embeddings,
        trainable_prefixes=trainable_prefixes,
        steps=args.steps,
        identities_per_batch=args.identities_per_batch,
        samples_per_identity=args.samples_per_identity,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scale=args.scale,
        margin=args.margin,
        all_distill_weight=args.all_distill_weight,
        unmasked_distill_weight=args.unmasked_distill_weight,
        seed=args.seed + 1,
        device=device,
    )
    curve.to_csv(args.out_dir / "arcface_finetune_training_curve.csv", index=False)
    finetuned_embeddings = embed_tensors(
        finetuned_model,
        tensors,
        all_paths,
        device=device,
        batch_size=args.embed_batch_size,
        desc="Embedding ArcFace fine-tuned model",
    )

    pairs = sample_pairs(eval_records, args.pairs_per_case, args.seed)
    scores, metrics = evaluate(pairs, baseline_embeddings, finetuned_embeddings)
    scores.to_csv(args.out_dir / "arcface_finetune_pair_scores.csv", index=False)
    metrics.to_csv(args.out_dir / "arcface_finetune_results.csv", index=False)
    pd.DataFrame({"identity": train_ids, "split": "train"}).to_csv(args.out_dir / "train_identities.csv", index=False)
    pd.DataFrame({"identity": eval_ids, "split": "eval"}).to_csv(args.out_dir / "eval_identities.csv", index=False)
    print(metrics.to_string(index=False))
    write_conclusion(metrics, args.out_dir)


if __name__ == "__main__":
    main()
