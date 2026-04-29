#!/usr/bin/env python3
"""Evaluate dedicated mask-aware recognizer checkpoints on the RMFRD split."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from probe_frozen_adapter import FaceRecord, Pair, best_threshold, cosine, sample_pairs, select_records
from probe_occlusion_ensemble import compute_embeddings as compute_facenet_views


OFFICIAL_REPO = "https://github.com/fdbtrs/Masked-Face-Recognition-KD.git"

MODEL_SPECS = {
    "elasticface_arc_aug": {
        "filename": "ElasticFaceArcAug_backbone.pth",
        "gdrive_id": "13-07SbnYGrdeXaX9uNc0TEY1xlkXzU5A",
    },
    "maskinv_hg": {
        "filename": "MaskInvHG_backbone.pth",
        "gdrive_id": "1fYJDn0MD7h5EpXPjWVT-rumzcLQjhqIA",
    },
    "maskinv_lg": {
        "filename": "MaskInvLG_backbone.pth",
        "gdrive_id": "1v8_plIehqFrFXGYToAiTZIggMtbF1BSu",
    },
}


def ensure_official_repo(repo_dir: Path) -> Path:
    if not (repo_dir / "backbones" / "iresnet.py").exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", OFFICIAL_REPO, str(repo_dir)], check=True)
    return repo_dir


def import_iresnet100(repo_dir: Path):
    sys.path.insert(0, str(repo_dir))
    from backbones import iresnet100  # type: ignore

    return iresnet100


def download_weight(model_name: str, weights_dir: Path, force: bool) -> Path:
    spec = MODEL_SPECS[model_name]
    weights_dir.mkdir(parents=True, exist_ok=True)
    out_path = weights_dir / spec["filename"]
    if out_path.exists() and not force:
        return out_path
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gdown",
            "--id",
            spec["gdrive_id"],
            "--output",
            str(out_path),
        ],
        check=True,
    )
    return out_path


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format at {path}")
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_maskaware_model(model_name: str, weight_path: Path, repo_dir: Path, device: str) -> torch.nn.Module:
    iresnet100 = import_iresnet100(repo_dir)
    model = iresnet100(num_features=512)
    model.load_state_dict(load_state_dict(weight_path), strict=True)
    model.eval().to(device)
    print({"loaded_model": model_name, "weight_path": str(weight_path)})
    return model


def align_arcface_faces(records: Sequence[FaceRecord], image_size: int, device: str) -> dict[Path, torch.Tensor]:
    mtcnn = MTCNN(image_size=image_size, margin=0, post_process=False, device=device)
    tensors: dict[Path, torch.Tensor] = {}
    failures = 0
    for record in tqdm(records, desc="Aligning 112x112 faces"):
        try:
            img = Image.open(record.path).convert("RGB")
            face = mtcnn(img)
        except Exception:
            face = None
        if face is None:
            failures += 1
            continue
        tensors[record.path] = face.detach().cpu()
    print({"arcface_aligned": len(tensors), "arcface_failures": failures})
    return tensors


@torch.inference_mode()
def embed_maskaware(
    model: torch.nn.Module,
    tensors: dict[Path, torch.Tensor],
    paths: Sequence[Path],
    device: str,
    batch_size: int,
    use_flip: bool,
    desc: str,
) -> dict[Path, np.ndarray]:
    embeddings: dict[Path, np.ndarray] = {}
    valid_paths = [path for path in paths if path in tensors]
    for start in tqdm(range(0, len(valid_paths), batch_size), desc=desc):
        batch_paths = valid_paths[start : start + batch_size]
        batch = torch.stack([tensors[path] for path in batch_paths]).to(device)
        batch = ((batch / 255.0) - 0.5) / 0.5
        out = model(batch)
        if use_flip:
            out = out + model(torch.flip(batch, dims=[3]))
        out = F.normalize(out, dim=-1).detach().cpu().numpy()
        for path, vector in zip(batch_paths, out, strict=True):
            embeddings[path] = vector
    return embeddings


def evaluate(
    pairs: Sequence[Pair],
    model_embeddings: dict[str, dict[Path, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for model_name, embeddings in model_embeddings.items():
        skipped = 0
        for pair in pairs:
            if pair.left not in embeddings or pair.right not in embeddings:
                skipped += 1
                continue
            rows.append(
                {
                    **asdict(pair),
                    "left": str(pair.left),
                    "right": str(pair.right),
                    "model": model_name,
                    "score": cosine(embeddings[pair.left], embeddings[pair.right]),
                    "skipped_for_model": skipped,
                }
            )
    scores = pd.DataFrame(rows)
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


def load_pair_head_reference(seed: int) -> dict[str, float] | None:
    candidates = {
        42: Path("artifacts/rmfrd_pair_verifier_head_probe/pair_verifier_head_results.csv"),
        7: Path("artifacts/rmfrd_pair_verifier_head_seed7/pair_verifier_head_results.csv"),
    }
    path = candidates.get(seed)
    if path is None or not path.exists():
        return None
    metrics = pd.read_csv(path)

    def auc(case: str) -> float:
        row = metrics[(metrics["model"] == "pair_head_masked_cases_only") & (metrics["case"] == case)]
        return float(row.iloc[0].roc_auc) if len(row) else math.nan

    return {"masked_unmasked": auc("masked-unmasked"), "unmasked_unmasked": auc("unmasked-unmasked")}


def write_conclusion(metrics: pd.DataFrame, out_dir: Path, seed: int) -> None:
    def auc(model: str, case: str) -> float:
        row = metrics[(metrics["model"] == model) & (metrics["case"] == case)]
        return float(row.iloc[0].roc_auc) if len(row) else math.nan

    baseline_mu = auc("baseline_facenet_full", "masked-unmasked")
    baseline_uu = auc("baseline_facenet_full", "unmasked-unmasked")
    evaluated = sorted(model for model in metrics["model"].unique() if model != "baseline_facenet_full")
    best = max(evaluated, key=lambda model: (auc(model, "masked-unmasked"), auc(model, "unmasked-unmasked")))
    best_mu = auc(best, "masked-unmasked")
    best_uu = auc(best, "unmasked-unmasked")
    pair_head = load_pair_head_reference(seed)
    pair_head_lines = ""
    if pair_head is not None:
        pair_head_lines = f"""
- Pair-head masked-unmasked ROC-AUC on same seed artifact: {pair_head["masked_unmasked"]:.4f}
- Best mask-aware gap vs pair head: {best_mu - pair_head["masked_unmasked"]:.4f}
"""
    verdict = "PROMISING BASELINE" if best_mu > baseline_mu else "WEAKER THAN FACENET BASELINE"
    text = f"""# Dedicated Mask-Aware Baseline Conclusion

Recommendation: {verdict}

- Seed: {seed}
- Baseline FaceNet masked-unmasked ROC-AUC: {baseline_mu:.4f}
- Best dedicated checkpoint: {best}
- Best dedicated masked-unmasked ROC-AUC: {best_mu:.4f}
- Best gain vs FaceNet baseline: {best_mu - baseline_mu:.4f}
- Baseline FaceNet unmasked-unmasked ROC-AUC: {baseline_uu:.4f}
- Best dedicated unmasked-unmasked ROC-AUC: {best_uu:.4f}
- Best dedicated unmasked regression vs FaceNet: {baseline_uu - best_uu:.4f}{pair_head_lines}
This probe evaluates official MaskInv-family IResNet-100 checkpoints from
`fdbtrs/Masked-Face-Recognition-KD` as dedicated mask-aware baselines on the
same RMFRD identity split protocol.
"""
    (out_dir / "maskaware_baseline_conclusion.md").write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--official-repo-dir", type=Path, default=Path("/content/Masked-Face-Recognition-KD"))
    parser.add_argument("--weights-dir", type=Path, default=Path("/content/maskinv_models"))
    parser.add_argument("--models", default="maskinv_hg,maskinv_lg,elasticface_arc_aug")
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--train-identities", type=int, default=140)
    parser.add_argument("--eval-identities", type=int, default=80)
    parser.add_argument("--max-images-per-condition", type=int, default=8)
    parser.add_argument("--pairs-per-case", type=int, default=800)
    parser.add_argument("--facenet-image-size", type=int, default=160)
    parser.add_argument("--arcface-image-size", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-flip", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print({"device": device, "data_root": str(args.data_root)})

    selected_models = [part.strip() for part in args.models.split(",") if part.strip()]
    unknown = sorted(set(selected_models) - set(MODEL_SPECS))
    if unknown:
        raise SystemExit(f"Unknown --models entries: {unknown}. Valid: {sorted(MODEL_SPECS)}")

    train_records, eval_records, train_ids, eval_ids = select_records(
        args.data_root,
        args.train_identities,
        args.eval_identities,
        args.max_images_per_condition,
        args.seed,
    )
    del train_records
    print({"eval_identities": len(eval_ids), "eval_images": len(eval_records), "models": selected_models})

    pairs = sample_pairs(eval_records, args.pairs_per_case, args.seed)
    eval_paths = sorted({pair.left for pair in pairs} | {pair.right for pair in pairs})

    facenet_views = compute_facenet_views(eval_records, image_size=args.facenet_image_size, device=device)
    model_embeddings: dict[str, dict[Path, np.ndarray]] = {
        "baseline_facenet_full": {path: views["full"] for path, views in facenet_views.items()}
    }

    repo_dir = ensure_official_repo(args.official_repo_dir)
    arcface_tensors = align_arcface_faces(eval_records, image_size=args.arcface_image_size, device=device)
    for model_name in selected_models:
        weight_path = args.weights_dir / MODEL_SPECS[model_name]["filename"]
        if args.download_missing or args.force_download or not weight_path.exists():
            weight_path = download_weight(model_name, args.weights_dir, force=args.force_download)
        model = load_maskaware_model(model_name, weight_path, repo_dir, device=device)
        model_embeddings[model_name] = embed_maskaware(
            model,
            arcface_tensors,
            eval_paths,
            device=device,
            batch_size=args.batch_size,
            use_flip=not args.no_flip,
            desc=f"Embedding {model_name}",
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scores, metrics = evaluate(pairs, model_embeddings)
    scores.to_csv(args.out_dir / "maskaware_baseline_pair_scores.csv", index=False)
    metrics.to_csv(args.out_dir / "maskaware_baseline_results.csv", index=False)
    pd.DataFrame({"identity": eval_ids, "split": "eval"}).to_csv(args.out_dir / "eval_identities.csv", index=False)
    print(metrics.to_string(index=False))
    write_conclusion(metrics, args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
