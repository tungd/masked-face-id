#!/usr/bin/env python3
"""Train a periocular specialist head and fuse it with frozen FaceNet."""

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
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from periocular_crop import PeriocularCropper
from probe_frozen_adapter import FaceRecord, Pair, best_threshold, cosine, group_records, sample_pairs, select_records


MODEL_BASELINE = "baseline_full"
MODEL_PERIOCULAR = "periocular_only"
MODEL_FUSED_FIXED = "fused_periocular_fixed_0.50"
MODEL_FUSED_TUNED = "fused_periocular_tuned"


class PeriocularEmbeddingNet(nn.Module):
    def __init__(self, num_identities: int, embedding_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 32),
            nn.MaxPool2d(2),
            self._conv_block(32, 64),
            nn.MaxPool2d(2),
            self._conv_block(64, 128),
            nn.MaxPool2d(2),
            self._conv_block(128, 192),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_identities)

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.features(x)), dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.embed(x)
        return embedding, self.classifier(embedding)


def crop_periocular_records(
    records: Sequence[FaceRecord],
    crop_width: int,
    crop_height: int,
    min_detection_confidence: float,
) -> tuple[dict[Path, torch.Tensor], pd.DataFrame]:
    crops: dict[Path, torch.Tensor] = {}
    rows = []
    with PeriocularCropper(min_detection_confidence=min_detection_confidence, refine_landmarks=True) as cropper:
        for record in tqdm(records, desc="Cropping periocular regions"):
            result = cropper.crop(record.path, output_size=(crop_width, crop_height))
            if result is None:
                rows.append(
                    {
                        "identity": record.identity,
                        "condition": record.condition,
                        "path": str(record.path),
                        "success": False,
                    }
                )
                continue
            crops[record.path] = result.tensor
            x0, y0, x1, y1 = result.bbox
            rows.append(
                {
                    "identity": record.identity,
                    "condition": record.condition,
                    "path": str(record.path),
                    "success": True,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "source_width": result.source_size[0],
                    "source_height": result.source_size[1],
                    "landmark_count": result.landmark_count,
                }
            )
    metadata = pd.DataFrame(rows)
    print(
        {
            "periocular_crops": len(crops),
            "crop_failures": int((~metadata["success"]).sum()) if len(metadata) else 0,
            "crop_width": crop_width,
            "crop_height": crop_height,
        }
    )
    return crops, metadata


def crop_records_by_identity(
    records: Sequence[FaceRecord],
    crops: dict[Path, torch.Tensor],
) -> dict[str, dict[str, list[FaceRecord]]]:
    grouped = group_records([record for record in records if record.path in crops])
    return {identity: groups for identity, groups in grouped.items() if groups.get("masked") and groups.get("unmasked")}


def sample_identity_batch(
    grouped: dict[str, dict[str, list[FaceRecord]]],
    identities: Sequence[str],
    identity_to_idx: dict[str, int],
    identities_per_batch: int,
    samples_per_identity: int,
    rng: random.Random,
) -> tuple[list[FaceRecord], list[int]]:
    chosen = rng.sample(list(identities), min(identities_per_batch, len(identities)))
    masked_count = max(1, samples_per_identity // 2)
    unmasked_count = max(1, samples_per_identity - masked_count)
    records: list[FaceRecord] = []
    labels: list[int] = []
    for identity in chosen:
        groups = grouped[identity]
        selected = [rng.choice(groups["masked"]) for _ in range(masked_count)]
        selected.extend(rng.choice(groups["unmasked"]) for _ in range(unmasked_count))
        records.extend(selected)
        labels.extend([identity_to_idx[identity]] * len(selected))
    return records, labels


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    sim = embeddings @ embeddings.T / temperature
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    sim = sim.masked_fill(self_mask, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    positive_counts = positive_mask.sum(dim=1)
    valid = positive_counts > 0
    if not bool(valid.any()):
        return embeddings.new_tensor(0.0)
    mean_log_prob = (log_prob * positive_mask).sum(dim=1)[valid] / positive_counts[valid]
    return -mean_log_prob.mean()


def train_periocular_specialist(
    records: Sequence[FaceRecord],
    crops: dict[Path, torch.Tensor],
    embedding_dim: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    steps_per_epoch: int,
    identities_per_batch: int,
    samples_per_identity: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    supcon_weight: float,
    seed: int,
    device: str,
) -> tuple[PeriocularEmbeddingNet, list[str], pd.DataFrame]:
    grouped = crop_records_by_identity(records, crops)
    identities = sorted(grouped)
    if not identities:
        raise ValueError("No train identities have both masked and unmasked periocular crops")
    identity_to_idx = {identity: idx for idx, identity in enumerate(identities)}
    model = PeriocularEmbeddingNet(
        num_identities=len(identities),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    rows = []

    for epoch in tqdm(range(1, epochs + 1), desc="Training periocular specialist"):
        model.train()
        losses = []
        ce_losses = []
        supcon_losses = []
        for _step in range(steps_per_epoch):
            batch_records, labels_list = sample_identity_batch(
                grouped,
                identities,
                identity_to_idx,
                identities_per_batch=identities_per_batch,
                samples_per_identity=samples_per_identity,
                rng=rng,
            )
            batch = torch.stack([crops[record.path] for record in batch_records]).to(device)
            labels = torch.tensor(labels_list, dtype=torch.long, device=device)
            embeddings, logits = model(batch)
            ce_loss = F.cross_entropy(logits, labels)
            supcon_loss = supervised_contrastive_loss(embeddings, labels, temperature=temperature)
            loss = ce_loss + supcon_weight * supcon_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            ce_losses.append(float(ce_loss.detach().cpu()))
            supcon_losses.append(float(supcon_loss.detach().cpu()))

        rows.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "cross_entropy": float(np.mean(ce_losses)),
                "supervised_contrastive": float(np.mean(supcon_losses)),
            }
        )
    return model.eval(), identities, pd.DataFrame(rows)


@torch.inference_mode()
def embed_periocular(
    model: PeriocularEmbeddingNet,
    crops: dict[Path, torch.Tensor],
    paths: Sequence[Path],
    device: str,
    batch_size: int,
) -> dict[Path, np.ndarray]:
    model.eval()
    embeddings: dict[Path, np.ndarray] = {}
    valid_paths = [path for path in paths if path in crops]
    for start in tqdm(range(0, len(valid_paths), batch_size), desc="Embedding periocular crops"):
        batch_paths = valid_paths[start : start + batch_size]
        batch = torch.stack([crops[path] for path in batch_paths]).to(device)
        vectors = model.embed(batch).detach().cpu().numpy()
        for path, vector in zip(batch_paths, vectors, strict=True):
            embeddings[path] = vector
    return embeddings


def align_full_faces(records: Sequence[FaceRecord], image_size: int, device: str) -> dict[Path, torch.Tensor]:
    mtcnn = MTCNN(image_size=image_size, margin=16, post_process=True, device=device)
    tensors: dict[Path, torch.Tensor] = {}
    failures = 0
    for record in tqdm(records, desc="Aligning full faces"):
        try:
            img = Image.open(record.path).convert("RGB")
            face = mtcnn(img)
        except Exception:
            face = None
        if face is None:
            failures += 1
            continue
        tensors[record.path] = face.detach().cpu()
    print({"full_face_aligned": len(tensors), "full_face_failures": failures})
    return tensors


@torch.inference_mode()
def compute_facenet_full_embeddings(
    records: Sequence[FaceRecord],
    image_size: int,
    device: str,
    batch_size: int,
) -> dict[Path, np.ndarray]:
    tensors = align_full_faces(records, image_size=image_size, device=device)
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    paths = [record.path for record in records if record.path in tensors]
    embeddings: dict[Path, np.ndarray] = {}
    for start in tqdm(range(0, len(paths), batch_size), desc="Embedding full faces"):
        batch_paths = paths[start : start + batch_size]
        batch = torch.stack([tensors[path] for path in batch_paths]).to(device)
        vectors = F.normalize(model(batch), dim=-1).detach().cpu().numpy()
        for path, vector in zip(batch_paths, vectors, strict=True):
            embeddings[path] = vector
    return embeddings


def component_scores(
    pairs: Sequence[Pair],
    split: str,
    facenet_embeddings: dict[Path, np.ndarray],
    periocular_embeddings: dict[Path, np.ndarray],
) -> pd.DataFrame:
    rows = []
    skipped = 0
    for pair in pairs:
        if (
            pair.left not in facenet_embeddings
            or pair.right not in facenet_embeddings
            or pair.left not in periocular_embeddings
            or pair.right not in periocular_embeddings
        ):
            skipped += 1
            continue
        rows.append(
            {
                **asdict(pair),
                "left": str(pair.left),
                "right": str(pair.right),
                "split": split,
                "full_score": cosine(facenet_embeddings[pair.left], facenet_embeddings[pair.right]),
                "periocular_score": cosine(periocular_embeddings[pair.left], periocular_embeddings[pair.right]),
                "skipped_so_far": skipped,
            }
        )
    return pd.DataFrame(rows)


def fused_score_value(case: str, full_score: float, periocular_score: float, alpha: float) -> float:
    if case == "unmasked-unmasked":
        return full_score
    return (1.0 - alpha) * full_score + alpha * periocular_score


def tune_alpha(components: pd.DataFrame, grid_size: int) -> tuple[float, pd.DataFrame]:
    rows = []
    alphas = np.linspace(0.0, 1.0, grid_size)
    masked_cases = ["masked-masked", "masked-unmasked"]
    for alpha in alphas:
        row: dict[str, float] = {"alpha": float(alpha)}
        aucs = []
        for case in masked_cases:
            case_df = components[components["case"] == case]
            if len(case_df) == 0 or len(case_df["label"].unique()) < 2:
                row[f"{case}_roc_auc"] = math.nan
                continue
            scores = (1.0 - alpha) * case_df["full_score"].to_numpy() + alpha * case_df["periocular_score"].to_numpy()
            auc = float(roc_auc_score(case_df["label"].to_numpy(), scores))
            row[f"{case}_roc_auc"] = auc
            aucs.append(auc)
        row["mean_masked_roc_auc"] = float(np.mean(aucs)) if aucs else math.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    valid = table.dropna(subset=["mean_masked_roc_auc"])
    if len(valid) == 0:
        return 0.5, table
    best = valid.sort_values(["mean_masked_roc_auc", "alpha"], ascending=[False, True]).iloc[0]
    return float(best.alpha), table


def score_models(components: pd.DataFrame, tuned_alpha: float) -> pd.DataFrame:
    if components.empty:
        raise ValueError("No scored pairs survived FaceNet and periocular embedding")
    rows = []
    for row in components.itertuples(index=False):
        base = row._asdict()
        common = {key: value for key, value in base.items() if key not in {"full_score", "periocular_score"}}
        rows.append({**common, "model": MODEL_BASELINE, "score": float(row.full_score), "fusion_alpha": 0.0})
        rows.append({**common, "model": MODEL_PERIOCULAR, "score": float(row.periocular_score), "fusion_alpha": 1.0})
        rows.append(
            {
                **common,
                "model": MODEL_FUSED_FIXED,
                "score": fused_score_value(row.case, float(row.full_score), float(row.periocular_score), 0.5),
                "fusion_alpha": 0.5,
            }
        )
        rows.append(
            {
                **common,
                "model": MODEL_FUSED_TUNED,
                "score": fused_score_value(row.case, float(row.full_score), float(row.periocular_score), tuned_alpha),
                "fusion_alpha": tuned_alpha,
            }
        )
    return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        raise ValueError("No model scores to summarize")
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


def write_conclusion(metrics: pd.DataFrame, out_dir: Path, tuned_alpha: float) -> None:
    def auc(model: str, case: str) -> float:
        row = metrics[(metrics["model"] == model) & (metrics["case"] == case)]
        return float(row.iloc[0].roc_auc) if len(row) else math.nan

    baseline_mu = auc(MODEL_BASELINE, "masked-unmasked")
    baseline_uu = auc(MODEL_BASELINE, "unmasked-unmasked")
    peri_mu = auc(MODEL_PERIOCULAR, "masked-unmasked")
    fixed_mu = auc(MODEL_FUSED_FIXED, "masked-unmasked")
    tuned_mu = auc(MODEL_FUSED_TUNED, "masked-unmasked")
    tuned_uu = auc(MODEL_FUSED_TUNED, "unmasked-unmasked")
    gain = tuned_mu - baseline_mu
    regression = baseline_uu - tuned_uu
    verdict = "PROMISING" if gain > 0 and regression <= 0.03 else "NOT YET PROMISING"
    text = f"""# Periocular Specialist Probe Conclusion

Recommendation: {verdict}

- Tuned fusion alpha: {tuned_alpha:.2f}
- Baseline full FaceNet masked-unmasked ROC-AUC: {baseline_mu:.4f}
- Periocular-only masked-unmasked ROC-AUC: {peri_mu:.4f}
- Fixed 0.50 fusion masked-unmasked ROC-AUC: {fixed_mu:.4f}
- Tuned fusion masked-unmasked ROC-AUC: {tuned_mu:.4f}
- Tuned fusion gain vs baseline: {gain:.4f}
- Baseline unmasked-unmasked ROC-AUC: {baseline_uu:.4f}
- Tuned fusion unmasked-unmasked ROC-AUC: {tuned_uu:.4f}
- Tuned fusion unmasked regression vs baseline: {regression:.4f}

This probe keeps FaceNet frozen for full-face embeddings, trains a dedicated
identity-supervised embedding head on MediaPipe Face Mesh periocular crops, and
uses the specialist only as fused evidence for masked verification cases.
"""
    (out_dir / "periocular_specialist_conclusion.md").write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-identities", type=int, default=140)
    parser.add_argument("--eval-identities", type=int, default=80)
    parser.add_argument("--max-images-per-condition", type=int, default=8)
    parser.add_argument("--train-pairs-per-case", type=int, default=600)
    parser.add_argument("--eval-pairs-per-case", type=int, default=600)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--crop-width", type=int, default=160)
    parser.add_argument("--crop-height", type=int, default=96)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=80)
    parser.add_argument("--identities-per-batch", type=int, default=16)
    parser.add_argument("--samples-per-identity", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--supcon-weight", type=float, default=0.25)
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--alpha-grid-size", type=int, default=21)
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
    all_records = train_records + eval_records
    all_paths = [record.path for record in all_records]

    crops, crop_metadata = crop_periocular_records(
        all_records,
        crop_width=args.crop_width,
        crop_height=args.crop_height,
        min_detection_confidence=args.min_detection_confidence,
    )
    crop_metadata.to_csv(args.out_dir / "periocular_crop_metadata.csv", index=False)

    specialist, specialist_identities, training_curve = train_periocular_specialist(
        train_records,
        crops,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        identities_per_batch=args.identities_per_batch,
        samples_per_identity=args.samples_per_identity,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        supcon_weight=args.supcon_weight,
        seed=args.seed,
        device=device,
    )
    training_curve.to_csv(args.out_dir / "periocular_specialist_training_curve.csv", index=False)
    torch.save(
        {
            "model_state_dict": specialist.state_dict(),
            "train_identities": specialist_identities,
            "args": vars(args),
        },
        args.out_dir / "periocular_specialist_head.pt",
    )

    periocular_embeddings = embed_periocular(
        specialist,
        crops,
        all_paths,
        device=device,
        batch_size=args.embed_batch_size,
    )
    facenet_embeddings = compute_facenet_full_embeddings(
        all_records,
        image_size=args.image_size,
        device=device,
        batch_size=args.embed_batch_size,
    )

    calibration_pairs = sample_pairs(train_records, args.train_pairs_per_case, args.seed + 101)
    eval_pairs = sample_pairs(eval_records, args.eval_pairs_per_case, args.seed + 202)
    calibration_components = component_scores(calibration_pairs, "calibration", facenet_embeddings, periocular_embeddings)
    eval_components = component_scores(eval_pairs, "eval", facenet_embeddings, periocular_embeddings)
    components = pd.concat([calibration_components, eval_components], ignore_index=True)
    components.to_csv(args.out_dir / "periocular_specialist_component_scores.csv", index=False)

    tuned_alpha, alpha_search = tune_alpha(calibration_components, grid_size=args.alpha_grid_size)
    alpha_search.to_csv(args.out_dir / "periocular_specialist_alpha_search.csv", index=False)
    scores = score_models(components, tuned_alpha=tuned_alpha)
    metrics = summarize(scores)

    scores.to_csv(args.out_dir / "periocular_specialist_pair_scores.csv", index=False)
    metrics.to_csv(args.out_dir / "periocular_specialist_results.csv", index=False)
    pd.DataFrame({"identity": train_ids, "split": "train"}).to_csv(args.out_dir / "train_identities.csv", index=False)
    pd.DataFrame({"identity": eval_ids, "split": "eval"}).to_csv(args.out_dir / "eval_identities.csv", index=False)
    print(metrics.to_string(index=False))
    write_conclusion(metrics, args.out_dir, tuned_alpha=tuned_alpha)


if __name__ == "__main__":
    main()
