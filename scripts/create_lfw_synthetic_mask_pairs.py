#!/usr/bin/env python3
"""Create an LFW synthetic masked/unmasked paired training root."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sklearn.datasets import fetch_lfw_people


MASK_COLORS = [
    (70, 150, 190),
    (93, 173, 226),
    (170, 183, 184),
    (230, 230, 230),
    (44, 62, 80),
    (125, 140, 142),
]


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip()).strip("_").lower() or "identity"


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        high = 255.0 if float(np.nanmax(arr)) > 1.5 else 1.0
        arr = np.clip(arr / high * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def synthetic_mask(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    top = rng.uniform(0.54, 0.61) * height
    bottom = rng.uniform(0.88, 0.96) * height
    left_top = rng.uniform(0.18, 0.25) * width
    right_top = rng.uniform(0.75, 0.82) * width
    left_bottom = rng.uniform(0.29, 0.36) * width
    right_bottom = rng.uniform(0.64, 0.71) * width
    color = rng.choice(MASK_COLORS)
    outline = tuple(max(0, channel - 35) for channel in color)

    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out, "RGBA")
    polygon = [
        (left_top, top),
        (right_top, top),
        (right_bottom, bottom),
        (left_bottom, bottom),
    ]
    draw.polygon(polygon, fill=(*color, 238), outline=(*outline, 255))

    # Pleats make the synthetic occlusion less like a single flat rectangle.
    for frac in [0.35, 0.50, 0.65]:
        y = top + (bottom - top) * frac
        draw.line(
            [(left_top + 0.04 * width, y), (right_top - 0.04 * width, y + rng.uniform(-2, 2))],
            fill=(*outline, 120),
            width=max(1, int(height * 0.01)),
        )
    return out


def save_pair(image: np.ndarray, out_root: Path, identity: str, index: int, rng: random.Random) -> None:
    pil = Image.fromarray(to_uint8_rgb(image)).convert("RGB")
    unmasked_path = out_root / "unmasked" / identity / f"{index:05d}.jpg"
    masked_path = out_root / "masked" / identity / f"{index:05d}.jpg"
    unmasked_path.parent.mkdir(parents=True, exist_ok=True)
    masked_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(unmasked_path, quality=94)
    synthetic_mask(pil, rng).save(masked_path, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("/content/datasets/normalized/lfw_synthetic_mask_pairs"))
    parser.add_argument("--data-home", type=Path, default=Path("/content/datasets/sklearn"))
    parser.add_argument("--min-faces-per-person", type=int, default=2)
    parser.add_argument("--max-identities", type=int, default=1200)
    parser.add_argument("--max-images-per-identity", type=int, default=8)
    parser.add_argument("--resize", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dataset = fetch_lfw_people(
        data_home=str(args.data_home),
        color=True,
        resize=args.resize,
        min_faces_per_person=args.min_faces_per_person,
        download_if_missing=True,
    )
    by_target: dict[int, list[int]] = {}
    for idx, target in enumerate(dataset.target):
        by_target.setdefault(int(target), []).append(idx)

    targets = [target for target, indices in by_target.items() if len(indices) >= args.min_faces_per_person]
    rng.shuffle(targets)
    targets = targets[: args.max_identities]
    args.out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for target in targets:
        identity = slugify(str(dataset.target_names[target]))
        indices = list(by_target[target])
        rng.shuffle(indices)
        indices = indices[: args.max_images_per_identity]
        for local_idx, image_idx in enumerate(indices):
            save_pair(dataset.images[image_idx], args.out_root, identity, local_idx, rng)
        summary_rows.append({"identity": identity, "images_per_condition": len(indices)})

    summary = {
        "out_root": str(args.out_root),
        "identities": len(summary_rows),
        "masked_images": sum(row["images_per_condition"] for row in summary_rows),
        "unmasked_images": sum(row["images_per_condition"] for row in summary_rows),
        "min_faces_per_person": args.min_faces_per_person,
        "max_images_per_identity": args.max_images_per_identity,
        "seed": args.seed,
    }
    (args.out_root / "synthetic_lfw_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
