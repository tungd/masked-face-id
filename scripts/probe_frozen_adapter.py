#!/usr/bin/env python3
"""Probe a frozen-recognizer adapter for masked face verification.

The probe freezes a FaceNet recognizer, trains a small linear adapter that maps
masked-image embeddings toward unmasked identity templates on calibration
identities, then evaluates on held-out identities.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CASES = ["masked-masked", "masked-unmasked", "unmasked-unmasked"]


@dataclass(frozen=True)
class FaceRecord:
    identity: str
    condition: str
    path: Path


@dataclass(frozen=True)
class Pair:
    case: str
    label: int
    left: Path
    right: Path
    left_id: str
    right_id: str
    left_condition: str
    right_condition: str


def normalize_condition(name: str) -> str | None:
    text = name.lower().replace("-", "_").replace(" ", "_")
    masked_tokens = [
        "masked",
        "with_mask",
        "with_masks",
        "mask",
        "rmfrd",
        "smfrd",
        "afdb_masked_face_dataset",
        "masked_face_dataset",
    ]
    unmasked_tokens = [
        "unmasked",
        "without_mask",
        "without_masks",
        "no_mask",
        "no_masks",
        "non_mask",
        "non_masked",
        "nomask",
        "common",
        "normal",
        "holistic",
        "afdb_face_dataset",
        "face_dataset",
    ]
    if any(token in text for token in masked_tokens):
        return "masked"
    if any(token in text for token in unmasked_tokens):
        return "unmasked"
    return None


def image_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def discover_records(root: Path) -> list[FaceRecord]:
    records: list[FaceRecord] = []
    for condition_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        condition = normalize_condition(condition_dir.name)
        if not condition:
            continue
        identity_dirs = [p for p in condition_dir.iterdir() if p.is_dir()]
        if identity_dirs:
            for identity_dir in sorted(identity_dirs):
                records.extend(FaceRecord(identity_dir.name, condition, img) for img in image_files(identity_dir))
        else:
            for img in image_files(condition_dir):
                identity = img.stem.split("_")[0].split("-")[0]
                records.append(FaceRecord(identity, condition, img))
    return records


def group_records(records: Sequence[FaceRecord]) -> dict[str, dict[str, list[FaceRecord]]]:
    grouped: dict[str, dict[str, list[FaceRecord]]] = {}
    for record in records:
        grouped.setdefault(record.identity, {}).setdefault(record.condition, []).append(record)
    return grouped


def select_records(
    root: Path,
    train_identities: int,
    eval_identities: int,
    max_images_per_condition: int,
    seed: int,
) -> tuple[list[FaceRecord], list[FaceRecord], list[str], list[str]]:
    records = discover_records(root)
    grouped = group_records(records)
    valid_ids = [ident for ident, groups in grouped.items() if groups.get("masked") and groups.get("unmasked")]
    rng = random.Random(seed)
    rng.shuffle(valid_ids)
    needed = train_identities + eval_identities
    if len(valid_ids) < needed:
        raise ValueError(f"Need {needed} valid identities, found {len(valid_ids)}")
    train_ids = valid_ids[:train_identities]
    eval_ids = valid_ids[train_identities:needed]

    def keep(ids: Sequence[str]) -> list[FaceRecord]:
        subset: list[FaceRecord] = []
        for ident in ids:
            for condition in ["masked", "unmasked"]:
                images = list(grouped[ident][condition])
                rng.shuffle(images)
                subset.extend(images[:max_images_per_condition])
        return subset

    return keep(train_ids), keep(eval_ids), train_ids, eval_ids


def lower_blackout(face_tensor: torch.Tensor) -> torch.Tensor:
    _, h, _ = face_tensor.shape
    out = face_tensor.clone()
    out[:, int(h * 0.55) :, :] = 0.0
    return out


class FrozenEmbedder:
    def __init__(self, image_size: int, device: str):
        self.device = device
        self.mtcnn = MTCNN(image_size=image_size, margin=16, post_process=True, device=device)
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    @torch.inference_mode()
    def embed(self, path: Path) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            img = Image.open(path).convert("RGB")
            face = self.mtcnn(img)
        except Exception:
            return None
        if face is None:
            return None
        faces = torch.stack([face, lower_blackout(face)]).to(self.device)
        emb = self.model(faces).detach().cpu().numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.maximum(norms, 1e-12)


def compute_embeddings(records: Sequence[FaceRecord], image_size: int, device: str) -> dict[Path, dict[str, np.ndarray]]:
    embedder = FrozenEmbedder(image_size=image_size, device=device)
    embeddings: dict[Path, dict[str, np.ndarray]] = {}
    failures = 0
    for record in tqdm(records, desc="Embedding images"):
        result = embedder.embed(record.path)
        if result is None:
            failures += 1
            continue
        full, blackout = result
        embeddings[record.path] = {"full": full, "blackout": blackout}
    print({"embedded": len(embeddings), "failures": failures})
    return embeddings


def mean_normalized(vectors: Sequence[np.ndarray]) -> np.ndarray | None:
    if not vectors:
        return None
    mean = np.mean(np.stack(vectors), axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm else mean


def build_templates(records: Sequence[FaceRecord], embeddings: dict[Path, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    by_id: dict[str, list[np.ndarray]] = {}
    for record in records:
        if record.condition != "unmasked" or record.path not in embeddings:
            continue
        by_id.setdefault(record.identity, []).append(embeddings[record.path]["full"])
    templates = {}
    for ident, vectors in by_id.items():
        template = mean_normalized(vectors)
        if template is not None:
            templates[ident] = template
    return templates


def train_ridge_adapter(
    records: Sequence[FaceRecord],
    embeddings: dict[Path, dict[str, np.ndarray]],
    templates: dict[str, np.ndarray],
    source: str,
    ridge: float,
) -> np.ndarray:
    xs = []
    ys = []
    for record in records:
        if record.condition != "masked" or record.path not in embeddings or record.identity not in templates:
            continue
        xs.append(embeddings[record.path][source])
        ys.append(templates[record.identity])
    if not xs:
        raise ValueError(f"No masked training embeddings for source={source}")
    x = np.stack(xs)
    y = np.stack(ys)
    x_aug = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    regularizer = ridge * np.eye(x_aug.shape[1])
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + regularizer, x_aug.T @ y)
    return weights


def apply_adapter(vector: np.ndarray, weights: np.ndarray) -> np.ndarray:
    projected = np.concatenate([vector, np.ones(1)]) @ weights
    norm = np.linalg.norm(projected)
    return projected / norm if norm else projected


def records_by_identity_condition(records: Sequence[FaceRecord]) -> dict[str, dict[str, list[FaceRecord]]]:
    return group_records(records)


def sample_pairs(records: Sequence[FaceRecord], pairs_per_case: int, seed: int) -> list[Pair]:
    rng = random.Random(seed)
    grouped = records_by_identity_condition(records)
    pairs: list[Pair] = []
    for case in CASES:
        left_condition, right_condition = case.split("-")
        ids = [ident for ident, groups in grouped.items() if groups.get(left_condition) and groups.get(right_condition)]
        positives = []
        negatives = []
        attempts = 0
        while len(positives) < pairs_per_case // 2 and attempts < pairs_per_case * 50:
            attempts += 1
            ident = rng.choice(ids)
            left = rng.choice(grouped[ident][left_condition])
            right = rng.choice(grouped[ident][right_condition])
            if left.path != right.path:
                positives.append(Pair(case, 1, left.path, right.path, ident, ident, left_condition, right_condition))
        attempts = 0
        while len(negatives) < pairs_per_case // 2 and attempts < pairs_per_case * 50:
            attempts += 1
            left_id, right_id = rng.sample(ids, 2)
            left = rng.choice(grouped[left_id][left_condition])
            right = rng.choice(grouped[right_id][right_condition])
            negatives.append(Pair(case, 0, left.path, right.path, left_id, right_id, left_condition, right_condition))
        pairs.extend(positives + negatives)
    rng.shuffle(pairs)
    return pairs


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def pair_vector(
    path: Path,
    condition: str,
    model: str,
    embeddings: dict[Path, dict[str, np.ndarray]],
    full_weights: np.ndarray,
    blackout_weights: np.ndarray,
) -> np.ndarray:
    emb = embeddings[path]
    if model == "baseline_full":
        return emb["full"]
    if model == "gated_blackout_all_mask_pairs":
        return emb["full"] if condition == "unmasked" else emb["blackout"]
    if model == "adapter_full_masked_only":
        return apply_adapter(emb["full"], full_weights) if condition == "masked" else emb["full"]
    if model == "adapter_blackout_masked_only":
        return apply_adapter(emb["blackout"], blackout_weights) if condition == "masked" else emb["full"]
    raise KeyError(model)


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


def evaluate(
    pairs: Sequence[Pair],
    embeddings: dict[Path, dict[str, np.ndarray]],
    full_weights: np.ndarray,
    blackout_weights: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = [
        "baseline_full",
        "gated_blackout_all_mask_pairs",
        "adapter_full_masked_only",
        "adapter_blackout_masked_only",
    ]
    score_rows = []
    for model in models:
        skipped = 0
        for pair in pairs:
            if pair.left not in embeddings or pair.right not in embeddings:
                skipped += 1
                continue
            left = pair_vector(pair.left, pair.left_condition, model, embeddings, full_weights, blackout_weights)
            right = pair_vector(pair.right, pair.right_condition, model, embeddings, full_weights, blackout_weights)
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
    metrics = pd.DataFrame(metric_rows).sort_values(["case", "model"])
    return scores, metrics


def write_conclusion(metrics: pd.DataFrame, out_dir: Path) -> None:
    def auc(model: str, case: str) -> float:
        row = metrics[(metrics["model"] == model) & (metrics["case"] == case)]
        return float(row.iloc[0].roc_auc) if len(row) else math.nan

    baseline_mu = auc("baseline_full", "masked-unmasked")
    baseline_uu = auc("baseline_full", "unmasked-unmasked")
    candidates = ["gated_blackout_all_mask_pairs", "adapter_full_masked_only", "adapter_blackout_masked_only"]
    best = max(candidates, key=lambda model: auc(model, "masked-unmasked"))
    best_mu = auc(best, "masked-unmasked")
    best_uu = auc(best, "unmasked-unmasked")
    gain = best_mu - baseline_mu
    regression = baseline_uu - best_uu
    verdict = "PROMISING" if gain > 0 and regression <= 0.03 else "NOT YET PROMISING"
    text = f"""# Frozen Adapter Probe Conclusion

Recommendation: {verdict}

- Best method on masked-unmasked: {best}
- Masked-unmasked ROC-AUC baseline: {baseline_mu:.4f}
- Masked-unmasked ROC-AUC best method: {best_mu:.4f}
- Masked-unmasked gain vs baseline: {gain:.4f}
- Unmasked-unmasked ROC-AUC baseline: {baseline_uu:.4f}
- Unmasked-unmasked ROC-AUC best method: {best_uu:.4f}
- Unmasked regression vs baseline: {regression:.4f}

This probe trains a ridge-linear adapter on calibration identities only and
evaluates on held-out identities. FaceNet remains frozen.
"""
    (out_dir / "frozen_adapter_probe_conclusion.md").write_text(text)
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
    parser.add_argument("--ridge", type=float, default=10.0)
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
    embeddings = compute_embeddings(all_records, image_size=args.image_size, device=device)
    templates = build_templates(train_records, embeddings)
    full_weights = train_ridge_adapter(train_records, embeddings, templates, source="full", ridge=args.ridge)
    blackout_weights = train_ridge_adapter(train_records, embeddings, templates, source="blackout", ridge=args.ridge)
    pairs = sample_pairs(eval_records, args.pairs_per_case, args.seed)
    scores, metrics = evaluate(pairs, embeddings, full_weights, blackout_weights)

    scores.to_csv(args.out_dir / "frozen_adapter_probe_pair_scores.csv", index=False)
    metrics.to_csv(args.out_dir / "frozen_adapter_probe_results.csv", index=False)
    pd.DataFrame({"identity": train_ids, "split": "train"}).to_csv(args.out_dir / "train_identities.csv", index=False)
    pd.DataFrame({"identity": eval_ids, "split": "eval"}).to_csv(args.out_dir / "eval_identities.csv", index=False)
    print(metrics.to_string(index=False))
    write_conclusion(metrics, args.out_dir)


if __name__ == "__main__":
    main()
