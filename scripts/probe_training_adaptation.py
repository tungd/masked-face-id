#!/usr/bin/env python3
"""Train stronger adaptation probes for masked face verification.

This script tests two training-heavy ideas while keeping the evaluation protocol
compatible with the earlier RMFRD probes:

1. A contrastive residual adapter on top of frozen FaceNet embeddings.
2. Partial fine-tuning of the FaceNet tail with supervised contrastive loss.
"""

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
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm

from probe_frozen_adapter import FaceRecord, Pair, best_threshold, cosine, group_records, sample_pairs, select_records


class ResidualEmbeddingAdapter(nn.Module):
    def __init__(self, dim: int = 512, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + self.scale.tanh() * self.net(x), dim=-1)


def align_faces(records: Sequence[FaceRecord], image_size: int, device: str) -> dict[Path, torch.Tensor]:
    mtcnn = MTCNN(image_size=image_size, margin=16, post_process=True, device=device)
    tensors: dict[Path, torch.Tensor] = {}
    failures = 0
    for record in tqdm(records, desc="Aligning faces"):
        try:
            img = Image.open(record.path).convert("RGB")
            face = mtcnn(img)
        except Exception:
            face = None
        if face is None:
            failures += 1
            continue
        tensors[record.path] = face.detach().cpu()
    print({"aligned": len(tensors), "failures": failures})
    return tensors


@torch.inference_mode()
def embed_tensors(
    model: nn.Module,
    tensors: dict[Path, torch.Tensor],
    paths: Sequence[Path],
    device: str,
    batch_size: int,
    desc: str,
) -> dict[Path, np.ndarray]:
    model.eval()
    embeddings: dict[Path, np.ndarray] = {}
    valid_paths = [path for path in paths if path in tensors]
    for start in tqdm(range(0, len(valid_paths), batch_size), desc=desc):
        batch_paths = valid_paths[start : start + batch_size]
        batch = torch.stack([tensors[path] for path in batch_paths]).to(device)
        emb = F.normalize(model(batch), dim=-1).detach().cpu().numpy()
        for path, vector in zip(batch_paths, emb, strict=True):
            embeddings[path] = vector
    return embeddings


def mean_normalized(vectors: Sequence[np.ndarray]) -> np.ndarray:
    mean = np.mean(np.stack(vectors), axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm else mean


def build_unmasked_templates(
    records: Sequence[FaceRecord],
    embeddings: dict[Path, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {}
    for record in records:
        if record.condition == "unmasked" and record.path in embeddings:
            grouped.setdefault(record.identity, []).append(embeddings[record.path])
    identities = sorted(grouped)
    templates = np.stack([mean_normalized(grouped[identity]) for identity in identities])
    return identities, templates


def train_contrastive_adapter(
    records: Sequence[FaceRecord],
    embeddings: dict[Path, np.ndarray],
    steps: int,
    batch_size: int,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    temperature: float,
    align_weight: float,
    seed: int,
    device: str,
) -> tuple[ResidualEmbeddingAdapter, pd.DataFrame]:
    identities, templates_np = build_unmasked_templates(records, embeddings)
    identity_to_idx = {identity: idx for idx, identity in enumerate(identities)}
    masked = [
        record
        for record in records
        if record.condition == "masked" and record.path in embeddings and record.identity in identity_to_idx
    ]
    if not masked:
        raise ValueError("No masked records with frozen embeddings for adapter training")

    rng = random.Random(seed)
    adapter = ResidualEmbeddingAdapter(dim=templates_np.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    templates = torch.tensor(templates_np, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    rows = []

    for step in tqdm(range(1, steps + 1), desc="Training contrastive adapter"):
        batch_records = [rng.choice(masked) for _ in range(batch_size)]
        x = torch.tensor(np.stack([embeddings[record.path] for record in batch_records]), dtype=torch.float32, device=device)
        targets = torch.tensor([identity_to_idx[record.identity] for record in batch_records], dtype=torch.long, device=device)
        adapted = adapter(x)
        logits = adapted @ templates.T / temperature
        ce_loss = F.cross_entropy(logits, targets)
        target_templates = templates[targets]
        align_loss = 1.0 - (adapted * target_templates).sum(dim=1).mean()
        loss = ce_loss + align_weight * align_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % max(1, steps // 20) == 0:
            rows.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "cross_entropy": float(ce_loss.detach().cpu()),
                    "alignment": float(align_loss.detach().cpu()),
                    "scale": float(adapter.scale.tanh().detach().cpu()),
                }
            )
    return adapter.eval(), pd.DataFrame(rows)


def records_by_identity(records: Sequence[FaceRecord], tensors: dict[Path, torch.Tensor]) -> dict[str, dict[str, list[FaceRecord]]]:
    grouped = group_records([record for record in records if record.path in tensors])
    return {identity: groups for identity, groups in grouped.items() if groups.get("masked") and groups.get("unmasked")}


def sample_identity_batch(
    grouped: dict[str, dict[str, list[FaceRecord]]],
    identities: Sequence[str],
    identities_per_batch: int,
    samples_per_identity: int,
    rng: random.Random,
) -> tuple[list[FaceRecord], list[int]]:
    chosen = rng.sample(list(identities), min(identities_per_batch, len(identities)))
    batch_records: list[FaceRecord] = []
    labels: list[int] = []
    masked_count = max(1, samples_per_identity // 2)
    unmasked_count = max(1, samples_per_identity - masked_count)
    for label, identity in enumerate(chosen):
        groups = grouped[identity]
        selected = [rng.choice(groups["masked"]) for _ in range(masked_count)]
        selected.extend(rng.choice(groups["unmasked"]) for _ in range(unmasked_count))
        batch_records.extend(selected)
        labels.extend([label] * len(selected))
    return batch_records, labels


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    sim = embeddings @ embeddings.T / temperature
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    logits_mask = ~self_mask
    sim = sim.masked_fill(~logits_mask, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    positive_counts = positive_mask.sum(dim=1)
    valid = positive_counts > 0
    if not bool(valid.any()):
        return embeddings.new_tensor(0.0)
    mean_log_prob = (log_prob * positive_mask).sum(dim=1)[valid] / positive_counts[valid]
    return -mean_log_prob.mean()


def set_trainable_tail(model: nn.Module, prefixes: Sequence[str]) -> int:
    for param in model.parameters():
        param.requires_grad_(False)
    trainable = 0
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            param.requires_grad_(True)
            trainable += param.numel()
    return trainable


def set_tail_train_mode(model: nn.Module, prefixes: Sequence[str]) -> None:
    model.eval()
    for name, module in model.named_modules():
        if any(name.startswith(prefix) for prefix in prefixes):
            module.train()


def train_partial_finetune(
    base_model: nn.Module,
    records: Sequence[FaceRecord],
    tensors: dict[Path, torch.Tensor],
    trainable_prefixes: Sequence[str],
    steps: int,
    identities_per_batch: int,
    samples_per_identity: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    seed: int,
    device: str,
) -> tuple[nn.Module, pd.DataFrame]:
    grouped = records_by_identity(records, tensors)
    identities = sorted(grouped)
    if not identities:
        raise ValueError("No train identities with aligned masked and unmasked tensors")
    trainable = set_trainable_tail(base_model, trainable_prefixes)
    if trainable == 0:
        raise ValueError(f"No trainable parameters matched prefixes: {trainable_prefixes}")
    print({"fine_tune_trainable_parameters": trainable, "prefixes": list(trainable_prefixes)})

    optimizer = torch.optim.AdamW((p for p in base_model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    rows = []

    for step in tqdm(range(1, steps + 1), desc="Fine-tuning recognizer tail"):
        set_tail_train_mode(base_model, trainable_prefixes)
        batch_records, labels_list = sample_identity_batch(
            grouped,
            identities,
            identities_per_batch=identities_per_batch,
            samples_per_identity=samples_per_identity,
            rng=rng,
        )
        batch = torch.stack([tensors[record.path] for record in batch_records]).to(device)
        labels = torch.tensor(labels_list, dtype=torch.long, device=device)
        embeddings = F.normalize(base_model(batch), dim=-1)
        loss = supervised_contrastive_loss(embeddings, labels, temperature=temperature)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in base_model.parameters() if p.requires_grad), max_norm=5.0)
        optimizer.step()

        if step == 1 or step % max(1, steps // 20) == 0:
            rows.append({"step": step, "loss": float(loss.detach().cpu())})
    base_model.eval()
    return base_model, pd.DataFrame(rows)


@torch.inference_mode()
def adapted_embeddings(
    adapter: ResidualEmbeddingAdapter,
    embeddings: dict[Path, np.ndarray],
    device: str,
    batch_size: int,
) -> dict[Path, np.ndarray]:
    out: dict[Path, np.ndarray] = {}
    paths = list(embeddings)
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = torch.tensor(np.stack([embeddings[path] for path in batch_paths]), dtype=torch.float32, device=device)
        adapted = adapter(batch).detach().cpu().numpy()
        for path, vector in zip(batch_paths, adapted, strict=True):
            out[path] = vector
    return out


def pair_vector(
    pair_path: Path,
    condition: str,
    model: str,
    baseline: dict[Path, np.ndarray],
    adapter: dict[Path, np.ndarray],
    finetuned: dict[Path, np.ndarray],
) -> np.ndarray:
    if model == "baseline_full":
        return baseline[pair_path]
    if model == "contrastive_adapter_masked_only":
        return adapter[pair_path] if condition == "masked" else baseline[pair_path]
    if model == "partial_finetune_full":
        return finetuned[pair_path]
    raise KeyError(model)


def evaluate(
    pairs: Sequence[Pair],
    baseline: dict[Path, np.ndarray],
    adapter: dict[Path, np.ndarray],
    finetuned: dict[Path, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = ["baseline_full", "contrastive_adapter_masked_only", "partial_finetune_full"]
    score_rows = []
    for model in models:
        skipped = 0
        for pair in pairs:
            required = baseline if model != "partial_finetune_full" else finetuned
            if pair.left not in required or pair.right not in required:
                skipped += 1
                continue
            left = pair_vector(pair.left, pair.left_condition, model, baseline, adapter, finetuned)
            right = pair_vector(pair.right, pair.right_condition, model, baseline, adapter, finetuned)
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

    baseline_mu = auc("baseline_full", "masked-unmasked")
    baseline_uu = auc("baseline_full", "unmasked-unmasked")
    candidates = ["contrastive_adapter_masked_only", "partial_finetune_full"]
    best = max(candidates, key=lambda model: auc(model, "masked-unmasked"))
    best_mu = auc(best, "masked-unmasked")
    best_uu = auc(best, "unmasked-unmasked")
    verdict = "PROMISING" if best_mu > baseline_mu and baseline_uu - best_uu <= 0.03 else "NOT YET PROMISING"
    text = f"""# Training Adaptation Probe Conclusion

Recommendation: {verdict}

- Baseline full FaceNet masked-unmasked ROC-AUC: {baseline_mu:.4f}
- Contrastive adapter masked-unmasked ROC-AUC: {auc("contrastive_adapter_masked_only", "masked-unmasked"):.4f}
- Partial fine-tune masked-unmasked ROC-AUC: {auc("partial_finetune_full", "masked-unmasked"):.4f}
- Best trained method: {best}
- Best masked-unmasked ROC-AUC: {best_mu:.4f}
- Best gain vs baseline: {best_mu - baseline_mu:.4f}
- Baseline unmasked-unmasked ROC-AUC: {baseline_uu:.4f}
- Best unmasked-unmasked ROC-AUC: {best_uu:.4f}
- Best unmasked regression vs baseline: {baseline_uu - best_uu:.4f}

This probe tests actual training: a supervised contrastive residual adapter on
frozen embeddings and partial supervised-contrastive fine-tuning of the FaceNet
tail. Evaluation identities are disjoint from training identities.
"""
    (out_dir / "training_adaptation_conclusion.md").write_text(text)
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
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--adapter-steps", type=int, default=1500)
    parser.add_argument("--adapter-batch-size", type=int, default=128)
    parser.add_argument("--adapter-hidden-dim", type=int, default=512)
    parser.add_argument("--adapter-dropout", type=float, default=0.1)
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    parser.add_argument("--adapter-weight-decay", type=float, default=1e-4)
    parser.add_argument("--adapter-temperature", type=float, default=0.07)
    parser.add_argument("--adapter-align-weight", type=float, default=0.25)

    parser.add_argument("--finetune-steps", type=int, default=1500)
    parser.add_argument("--finetune-identities-per-batch", type=int, default=16)
    parser.add_argument("--finetune-samples-per-identity", type=int, default=4)
    parser.add_argument("--finetune-lr", type=float, default=2e-5)
    parser.add_argument("--finetune-weight-decay", type=float, default=1e-4)
    parser.add_argument("--finetune-temperature", type=float, default=0.07)
    parser.add_argument(
        "--finetune-trainable-prefixes",
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

    adapter, adapter_curve = train_contrastive_adapter(
        train_records,
        baseline_embeddings,
        steps=args.adapter_steps,
        batch_size=args.adapter_batch_size,
        hidden_dim=args.adapter_hidden_dim,
        dropout=args.adapter_dropout,
        lr=args.adapter_lr,
        weight_decay=args.adapter_weight_decay,
        temperature=args.adapter_temperature,
        align_weight=args.adapter_align_weight,
        seed=args.seed,
        device=device,
    )
    adapter_curve.to_csv(args.out_dir / "contrastive_adapter_training_curve.csv", index=False)
    adapter_embeddings = adapted_embeddings(adapter, baseline_embeddings, device=device, batch_size=args.embed_batch_size)

    trainable_prefixes = [part.strip() for part in args.finetune_trainable_prefixes.split(",") if part.strip()]
    finetuned_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    finetuned_model, finetune_curve = train_partial_finetune(
        finetuned_model,
        train_records,
        tensors,
        trainable_prefixes=trainable_prefixes,
        steps=args.finetune_steps,
        identities_per_batch=args.finetune_identities_per_batch,
        samples_per_identity=args.finetune_samples_per_identity,
        lr=args.finetune_lr,
        weight_decay=args.finetune_weight_decay,
        temperature=args.finetune_temperature,
        seed=args.seed + 1,
        device=device,
    )
    finetune_curve.to_csv(args.out_dir / "partial_finetune_training_curve.csv", index=False)
    finetuned_embeddings = embed_tensors(
        finetuned_model,
        tensors,
        all_paths,
        device=device,
        batch_size=args.embed_batch_size,
        desc="Embedding fine-tuned model",
    )

    pairs = sample_pairs(eval_records, args.pairs_per_case, args.seed)
    scores, metrics = evaluate(pairs, baseline_embeddings, adapter_embeddings, finetuned_embeddings)
    scores.to_csv(args.out_dir / "training_adaptation_pair_scores.csv", index=False)
    metrics.to_csv(args.out_dir / "training_adaptation_results.csv", index=False)
    pd.DataFrame({"identity": train_ids, "split": "train"}).to_csv(args.out_dir / "train_identities.csv", index=False)
    pd.DataFrame({"identity": eval_ids, "split": "eval"}).to_csv(args.out_dir / "eval_identities.csv", index=False)
    print(metrics.to_string(index=False))
    write_conclusion(metrics, args.out_dir)


if __name__ == "__main__":
    main()
