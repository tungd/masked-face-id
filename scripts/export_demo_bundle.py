#!/usr/bin/env python3
"""Export a presentation-safe offline demo bundle.

The main robustness probe writes raw pair score rows in Colab. This script turns
that score CSV plus the corresponding dataset image root into a small static
bundle that can run without Colab, a GPU, or network access.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image, ImageDraw, ImageFilter, ImageOps


BASELINE_MODEL = "baseline_full"
PAIR_MODEL = "pair_head_full_all_features_masked_only"
VIEW_NAMES = ["full", "lower_blackout", "lower_blur", "upper_only", "eye_band"]
REQUIRED_COLUMNS = {"seed", "split", "model", "case", "label", "score", "left", "right"}
PAIR_META_COLUMNS = ["left_id", "right_id", "left_condition", "right_condition"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}


def read_scores(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [dict(row) for row in reader]
    missing = sorted(REQUIRED_COLUMNS - set(fieldnames))
    if missing:
        raise ValueError(f"Score CSV is missing required columns: {missing}")
    splits = {row["split"] for row in rows}
    if "calibration" not in splits or "eval" not in splits:
        raise ValueError("Score CSV must contain both calibration and eval rows")
    return rows, fieldnames


def row_seed(row: dict[str, str]) -> int:
    return int(float(row["seed"]))


def row_label(row: dict[str, Any]) -> int:
    return int(float(row["label"]))


def row_score(row: dict[str, str]) -> float:
    return float(row["score"])


def threshold_for_far(labels: list[int], scores: list[float], target_far: float) -> float:
    negatives = sorted(score for label, score in zip(labels, scores, strict=True) if label == 0)
    if not negatives:
        return float(max(scores))
    allowed_false_accepts = int(math.floor(target_far * len(negatives)))
    if allowed_false_accepts <= 0:
        return float(math.nextafter(negatives[-1], math.inf))
    index = max(0, len(negatives) - allowed_false_accepts)
    return float(negatives[index])


def build_thresholds(
    rows: list[dict[str, str]],
    seed: int,
    models: Iterable[str],
    target_far: float,
) -> dict[str, dict[str, float]]:
    wanted = set(models)
    grouped: dict[tuple[str, str], tuple[list[int], list[float]]] = {}
    for row in rows:
        if row["split"] != "calibration" or row_seed(row) != seed or row["model"] not in wanted:
            continue
        labels, scores = grouped.setdefault((row["model"], row["case"]), ([], []))
        labels.append(row_label(row))
        scores.append(row_score(row))

    thresholds: dict[str, dict[str, float]] = {model: {} for model in wanted}
    for (model, case), (labels, scores) in grouped.items():
        thresholds[model][case] = threshold_for_far(labels, scores, target_far)
    for model in wanted:
        if not thresholds[model]:
            raise ValueError(f"No calibration rows found for model={model!r}, seed={seed}")
    return thresholds


def join_key(row: dict[str, str], columns: list[str]) -> tuple[str, ...]:
    values: list[str] = []
    for column in columns:
        value = row.get(column, "")
        if column == "label":
            value = str(row_label(row))
        values.append(str(value))
    return tuple(values)


def merge_model_scores(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    seed: int,
    baseline_model: str,
    pair_model: str,
) -> list[dict[str, Any]]:
    eval_rows = [row for row in rows if row["split"] == "eval" and row_seed(row) == seed]
    if not eval_rows:
        raise ValueError(f"No eval rows found for seed={seed}")

    join_columns = ["seed", "split", "case", "label", "left", "right"]
    join_columns.extend(column for column in PAIR_META_COLUMNS if column in fieldnames)

    baseline: dict[tuple[str, ...], dict[str, str]] = {}
    pair_head: dict[tuple[str, ...], dict[str, str]] = {}
    for row in eval_rows:
        if row["model"] == baseline_model:
            baseline.setdefault(join_key(row, join_columns), row)
        elif row["model"] == pair_model:
            pair_head.setdefault(join_key(row, join_columns), row)

    if not baseline:
        raise ValueError(f"No eval rows found for baseline model={baseline_model!r}, seed={seed}")
    if not pair_head:
        raise ValueError(f"No eval rows found for pair-head model={pair_model!r}, seed={seed}")

    merged: list[dict[str, Any]] = []
    for key, base_row in baseline.items():
        head_row = pair_head.get(key)
        if head_row is None:
            continue
        combined: dict[str, Any] = {column: base_row.get(column, "") for column in join_columns}
        combined["label"] = row_label(base_row)
        combined["baseline_score"] = row_score(base_row)
        combined["pair_head_score"] = row_score(head_row)
        merged.append(combined)
    if not merged:
        raise ValueError("No eval pairs had both baseline and pair-head scores")
    return merged


def add_policy_columns(
    pairs: list[dict[str, Any]],
    thresholds: dict[str, dict[str, float]],
    target_far: float,
    baseline_model: str,
    pair_model: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in pairs:
        case = str(row["case"])
        baseline_threshold = thresholds[baseline_model][case]
        pair_threshold = thresholds[pair_model].get(case, baseline_threshold)
        use_baseline = case == "unmasked-unmasked"
        final_score = float(row["baseline_score"] if use_baseline else row["pair_head_score"])
        final_threshold = float(baseline_threshold if use_baseline else pair_threshold)
        enriched = dict(row)
        enriched.update(
            {
                "baseline_threshold": float(baseline_threshold),
                "pair_head_threshold": float(pair_threshold),
                "final_score": final_score,
                "final_threshold": final_threshold,
                "target_far": float(target_far),
                "baseline_decision": int(float(row["baseline_score"]) >= baseline_threshold),
                "pair_head_decision": int(float(row["pair_head_score"]) >= pair_threshold),
                "final_decision": int(final_score >= final_threshold),
                "final_model": baseline_model if use_baseline else pair_model,
            }
        )
        rows.append(enriched)
    return rows


def pair_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (str(row["left"]), str(row["right"]), str(row["case"]), int(row["label"]))


def decision_word(value: int) -> str:
    return "match" if int(value) == 1 else "non-match"


def select_one(
    candidates: list[dict[str, Any]],
    used: set[tuple[str, str, str, int]],
    predicate: Callable[[dict[str, Any]], bool],
    sort_key: Callable[[dict[str, Any]], float],
    role: str,
    title: str,
    notes: str,
) -> dict[str, Any] | None:
    eligible = [row for row in candidates if pair_key(row) not in used and predicate(row)]
    if not eligible:
        return None
    selected = sorted(eligible, key=sort_key, reverse=True)[0].copy()
    selected.update({"outcome_role": role, "title": title, "notes": notes})
    used.add(pair_key(selected))
    return selected


def select_examples(candidates: list[dict[str, Any]], requested: int) -> list[dict[str, Any]]:
    used: set[tuple[str, str, str, int]] = set()
    selected: list[dict[str, Any]] = []

    def add(row: dict[str, Any] | None) -> None:
        if row is not None:
            selected.append(row)

    add(
        select_one(
            candidates,
            used,
            lambda row: row["case"] == "masked-unmasked"
            and int(row["label"]) == 1
            and int(row["baseline_decision"]) == 0
            and int(row["final_decision"]) == 1,
            lambda row: (row["final_score"] - row["final_threshold"]) + (row["baseline_threshold"] - row["baseline_score"]),
            "masked_unmasked_recovered",
            "Masked-unmasked match recovered",
            "Baseline cosine falls below the selected threshold, while the pair head crosses it.",
        )
    )
    add(
        select_one(
            candidates,
            used,
            lambda row: row["case"] == "masked-masked" and int(row["label"]) == 1 and int(row["final_decision"]) == 1,
            lambda row: row["final_score"] - row["final_threshold"],
            "masked_masked_match",
            "Masked-masked genuine match",
            "Both images are masked, so the adapted pair-head score is the deployed policy score.",
        )
    )
    add(
        select_one(
            candidates,
            used,
            lambda row: row["case"] == "unmasked-unmasked" and int(row["label"]) == 1 and int(row["final_decision"]) == 1,
            lambda row: row["baseline_score"] - row["baseline_threshold"],
            "unmasked_bypass",
            "Unmasked-unmasked bypass",
            "No mask is involved, so the policy preserves the legacy FaceNet cosine path.",
        )
    )
    add(
        select_one(
            candidates,
            used,
            lambda row: int(row["label"]) == 0
            and int(row["baseline_decision"]) == 0
            and int(row["pair_head_decision"]) == 0
            and int(row["final_decision"]) == 0,
            lambda row: min(row["baseline_threshold"] - row["baseline_score"], row["final_threshold"] - row["final_score"]),
            "impostor_rejected",
            "Impostor pair rejected",
            "The baseline and adapted scores both remain below their selected thresholds.",
        )
    )
    add(
        select_one(
            candidates,
            used,
            lambda row: int(row["final_decision"]) != int(row["label"]),
            lambda row: -abs(row["final_score"] - row["final_threshold"]),
            "hard_failure",
            "Hard or failure case",
            "The demo keeps this case visible because verifier thresholds can still make mistakes.",
        )
    )

    while len(selected) < requested:
        filler = select_one(
            candidates,
            used,
            lambda _row: True,
            lambda row: -abs(row["final_score"] - row["final_threshold"]),
            "additional_representative",
            "Additional representative pair",
            "Extra curated pair selected near the operating threshold.",
        )
        if filler is None:
            break
        selected.append(filler)

    if not selected:
        raise ValueError("No demo examples could be selected")
    return selected[:requested]


def resolve_image(path_text: str, image_root: Path) -> Path:
    raw = Path(path_text)
    if raw.exists():
        return raw
    candidates = []
    if not raw.is_absolute():
        candidates.append(image_root / raw)
    parts = raw.parts
    start = 1 if raw.is_absolute() else 0
    for idx in range(start, len(parts)):
        suffix = Path(*parts[idx:])
        candidates.append(image_root / suffix)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    matches = list(image_root.rglob(raw.name)) if image_root.exists() and raw.name else []
    image_matches = [match for match in matches if match.suffix.lower() in IMAGE_EXTS]
    if len(image_matches) == 1:
        return image_matches[0]
    if len(image_matches) > 1:
        raise FileNotFoundError(f"Ambiguous image filename {raw.name!r}; found {len(image_matches)} matches under {image_root}")
    raise FileNotFoundError(f"Could not resolve image {path_text!r} under {image_root}")


def normalized_image(src: Path, dst: Path, size: int) -> Image.Image:
    with Image.open(src) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (244, 246, 249))
    offset = ((size - image.width) // 2, (size - image.height) // 2)
    canvas.paste(image, offset)
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst, quality=92)
    return canvas


def save_view(image: Image.Image, dst: Path, view: str) -> None:
    out = image.copy()
    width, height = out.size
    lower_y = int(height * 0.56)
    if view == "full":
        pass
    elif view == "lower_blackout":
        draw = ImageDraw.Draw(out)
        draw.rectangle([0, lower_y, width, height], fill=(18, 24, 33))
    elif view == "lower_blur":
        lower = out.crop((0, lower_y, width, height)).filter(ImageFilter.GaussianBlur(radius=18))
        out.paste(lower, (0, lower_y))
    elif view == "upper_only":
        crop_y = int(height * 0.62)
        masked = Image.new("RGB", out.size, (226, 232, 240))
        masked.paste(out.crop((0, 0, width, crop_y)), (0, 0))
        out = masked
    elif view == "eye_band":
        band_top = int(height * 0.28)
        band_bottom = int(height * 0.52)
        masked = Image.new("RGB", out.size, (31, 41, 55))
        masked.paste(out.crop((0, band_top, width, band_bottom)), (0, band_top))
        draw = ImageDraw.Draw(masked)
        draw.rectangle([0, band_top - 3, width, band_top], fill=(14, 165, 233))
        draw.rectangle([0, band_bottom, width, band_bottom + 3], fill=(14, 165, 233))
        out = masked
    else:
        raise KeyError(view)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst, quality=90)


def write_example_images(row: dict[str, Any], example_dir: Path, image_root: Path, image_size: int) -> dict[str, dict[str, Path]]:
    left_src = resolve_image(str(row["left"]), image_root)
    right_src = resolve_image(str(row["right"]), image_root)
    left_image = normalized_image(left_src, example_dir / "left.jpg", image_size)
    right_image = normalized_image(right_src, example_dir / "right.jpg", image_size)

    view_paths = {"left": {}, "right": {}}
    for view in VIEW_NAMES:
        left_dst = example_dir / "views_left" / f"{view}.jpg"
        right_dst = example_dir / "views_right" / f"{view}.jpg"
        save_view(left_image, left_dst, view)
        save_view(right_image, right_dst, view)
        view_paths["left"][view] = left_dst
        view_paths["right"][view] = right_dst
    return view_paths


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def make_demo_rows(
    selected: list[dict[str, Any]],
    assets_dir: Path,
    image_root: Path,
    image_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    examples_dir = assets_dir / "examples"
    if examples_dir.exists():
        shutil.rmtree(examples_dir)
    examples_dir.mkdir(parents=True, exist_ok=True)

    pair_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        example_id = f"example_{index:03d}"
        example_dir = examples_dir / example_id
        view_paths = write_example_images(row, example_dir, image_root, image_size)
        pair_rows.append(
            {
                "example_id": example_id,
                "title": row["title"],
                "outcome_role": row["outcome_role"],
                "case": row["case"],
                "ground_truth_label": int(row["label"]),
                "ground_truth": "genuine" if int(row["label"]) == 1 else "impostor",
                "left_id": row.get("left_id", ""),
                "right_id": row.get("right_id", ""),
                "left_condition": row.get("left_condition", ""),
                "right_condition": row.get("right_condition", ""),
                "left_image": rel(example_dir / "left.jpg", assets_dir),
                "right_image": rel(example_dir / "right.jpg", assets_dir),
                "left_source": str(row["left"]),
                "right_source": str(row["right"]),
                "notes": row["notes"],
                "views_left": {view: rel(view_paths["left"][view], assets_dir) for view in VIEW_NAMES},
                "views_right": {view: rel(view_paths["right"][view], assets_dir) for view in VIEW_NAMES},
            }
        )
        score_rows.append(
            {
                "example_id": example_id,
                "raw_facenet_cosine": row["baseline_score"],
                "pair_head_score": row["pair_head_score"],
                "final_policy_score": row["final_score"],
                "baseline_threshold": row["baseline_threshold"],
                "pair_head_threshold": row["pair_head_threshold"],
                "selected_threshold": row["final_threshold"],
                "target_far": row["target_far"],
                "baseline_decision": decision_word(row["baseline_decision"]),
                "pair_head_decision": decision_word(row["pair_head_decision"]),
                "predicted_decision": decision_word(row["final_decision"]),
                "final_model": row["final_model"],
            }
        )
    return pair_rows, score_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_demo_data(
    assets_dir: Path,
    pair_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    thresholds: dict[str, dict[str, float]],
    summary: dict[str, Any],
) -> None:
    pair_csv_fields = [
        "example_id",
        "title",
        "outcome_role",
        "case",
        "ground_truth_label",
        "ground_truth",
        "left_id",
        "right_id",
        "left_condition",
        "right_condition",
        "left_image",
        "right_image",
        "left_source",
        "right_source",
        "notes",
    ]
    score_csv_fields = [
        "example_id",
        "raw_facenet_cosine",
        "pair_head_score",
        "final_policy_score",
        "baseline_threshold",
        "pair_head_threshold",
        "selected_threshold",
        "target_far",
        "baseline_decision",
        "pair_head_decision",
        "predicted_decision",
        "final_model",
    ]
    write_csv(assets_dir / "demo_pairs.csv", pair_rows, pair_csv_fields)
    write_csv(assets_dir / "demo_scores.csv", score_rows, score_csv_fields)
    (assets_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (assets_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    scores_by_id = {row["example_id"]: row for row in score_rows}
    examples = [{**pair, **scores_by_id[pair["example_id"]]} for pair in pair_rows]
    payload = {"examples": examples, "thresholds": thresholds, "summary": summary}
    (assets_dir / "demo_data.js").write_text("window.DEMO_DATA = " + json.dumps(payload, indent=2) + ";\n")


def copy_static_app(out_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "demo" / "index.html"
    target = out_dir / "index.html"
    if source.exists() and source.resolve() != target.resolve():
        shutil.copy2(source, target)


def write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# Masked Face Verification Demo Bundle

Open `index.html` in a browser to run the curated verifier offline.

Bundle summary:

- examples: {summary["examples"]}
- seed: {summary["seed"]}
- target FAR: {summary["target_far"]}
- baseline model: `{summary["baseline_model"]}`
- adapted model: `{summary["pair_model"]}`

The app uses precomputed scores and copied image assets, so it does not require
Colab, a GPU, or network access during presentation.
"""
    (out_dir / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True, help="Raw pair_head_robustness_pair_scores.csv from Colab.")
    parser.add_argument("--image-root", type=Path, required=True, help="Dataset root used by the score CSV image paths.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output bundle directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-far", type=float, default=0.05)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--baseline-model", default=BASELINE_MODEL)
    parser.add_argument("--pair-model", default=PAIR_MODEL)
    parser.add_argument("--image-size", type=int, default=520)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional pair-head checkpoint to copy into assets/model/.")
    parser.add_argument("--standardizer", type=Path, default=None, help="Optional feature standardizer to copy into assets/model/.")
    parser.add_argument("--config", type=Path, default=None, help="Optional model/demo config to copy into assets/model/.")
    args = parser.parse_args()

    rows, fieldnames = read_scores(args.scores)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = args.out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    thresholds = build_thresholds(rows, args.seed, [args.baseline_model, args.pair_model], args.target_far)
    merged = merge_model_scores(rows, fieldnames, args.seed, args.baseline_model, args.pair_model)
    candidates = add_policy_columns(merged, thresholds, args.target_far, args.baseline_model, args.pair_model)
    selected = select_examples(candidates, args.examples)
    pair_rows, score_rows = make_demo_rows(selected, assets_dir, args.image_root, args.image_size)

    copied_model_files = []
    model_dir = assets_dir / "model"
    for optional in [args.checkpoint, args.standardizer, args.config]:
        if optional is None:
            continue
        model_dir.mkdir(parents=True, exist_ok=True)
        target = model_dir / optional.name
        shutil.copy2(optional, target)
        copied_model_files.append(rel(target, assets_dir))

    summary = {
        "source_scores": str(args.scores),
        "image_root": str(args.image_root),
        "seed": args.seed,
        "target_far": args.target_far,
        "examples": len(pair_rows),
        "baseline_model": args.baseline_model,
        "pair_model": args.pair_model,
        "model_files": copied_model_files,
        "selection_roles": [row["outcome_role"] for row in pair_rows],
    }
    write_demo_data(assets_dir, pair_rows, score_rows, thresholds, summary)
    copy_static_app(args.out_dir)
    write_readme(args.out_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
