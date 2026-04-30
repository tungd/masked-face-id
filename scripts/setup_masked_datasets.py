#!/usr/bin/env python3
"""Colab helpers for downloading and normalizing RMFRD/SMFRD-style datasets."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse
from urllib.request import urlretrieve


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}

RMFRD_GDRIVE_URL = "https://drive.google.com/open?id=1UlOk6EtiaXTHylRUx2mySgvJX9ycoeBp"
RMFRD_KAGGLE_SLUG = "muhammeddalkran/masked-facerecognition"
LFW_SMFRD_KAGGLE_SLUG = "muhammeddalkran/lfw-simulated-masked-face-dataset"


def normalize_condition(name: str) -> str | None:
    text = name.lower().replace("-", "_").replace(" ", "_")
    if any(token in text for token in ["masked", "with_mask", "with_masks", "mask", "rmfrd", "smfrd"]):
        if any(token in text for token in ["unmasked", "without_mask", "without_masks", "no_mask", "nomask"]):
            return "unmasked"
        return "masked"
    if any(
        token in text
        for token in [
            "unmasked",
            "without_mask",
            "without_masks",
            "no_mask",
            "non_mask",
            "nomask",
            "normal",
            "afdb_face_dataset",
            "face_dataset",
        ]
    ):
        return "unmasked"
    return None


def image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def extract_archive(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lower = archive.name.lower()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(out_dir)
        return
    if lower.endswith(".rar"):
        raise SystemExit(
            f"RAR archive detected: {archive}. Install unrar/patool or extract it manually, "
            f"then rerun this script with --summarize-only."
        )
    shutil.unpack_archive(str(archive), str(out_dir))


def run_command(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def download_gdown(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path
    run_command([sys.executable, "-m", "gdown", "--fuzzy", url, "--output", str(out_path)])
    return out_path


def download_url(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path
    urlretrieve(url, out_path)
    return out_path


def archive_path_from_url(url: str, fallback: str, archives_dir: Path) -> Path:
    name = Path(unquote(urlparse(url).path)).name
    return archives_dir / (name or fallback)


def download_kaggle(slug: str, out_dir: Path, unzip: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", slug, "-p", str(out_dir)]
    if unzip:
        cmd.append("--unzip")
    run_command(cmd)
    return out_dir


def identity_from_image(path: Path, root: Path) -> str:
    parent = path.parent
    if parent != root:
        return parent.name
    return path.stem.split("_")[0].split("-")[0]


def collect_by_identity(root: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in image_files(root):
        grouped.setdefault(identity_from_image(path, root), []).append(path)
    return grouped


def discover_condition_records(root: Path) -> dict[str, dict[str, list[Path]]]:
    records: dict[str, dict[str, list[Path]]] = {"masked": {}, "unmasked": {}}
    if not root.exists():
        return records
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        condition = normalize_condition(child.name)
        if not condition:
            continue
        grouped = collect_by_identity(child)
        for identity, paths in grouped.items():
            records[condition].setdefault(identity, []).extend(paths)
    return records


def summarize_root(root: Path) -> dict[str, object]:
    records = discover_condition_records(root)
    masked_ids = set(records["masked"])
    unmasked_ids = set(records["unmasked"])
    valid_ids = masked_ids & unmasked_ids
    return {
        "root": str(root),
        "exists": root.exists(),
        "masked_identities": len(masked_ids),
        "unmasked_identities": len(unmasked_ids),
        "valid_paired_identities": len(valid_ids),
        "masked_images": sum(len(paths) for paths in records["masked"].values()),
        "unmasked_images": sum(len(paths) for paths in records["unmasked"].values()),
    }


def candidate_roots(root: Path, max_depth: int) -> Iterable[Path]:
    yield root
    if not root.exists():
        return
    root_depth = len(root.parts)
    for path in sorted(p for p in root.rglob("*") if p.is_dir()):
        if len(path.parts) - root_depth <= max_depth:
            yield path


def summarize_candidates(root: Path, max_depth: int, min_valid_identities: int) -> list[dict[str, object]]:
    rows = []
    for candidate in candidate_roots(root, max_depth=max_depth):
        summary = summarize_root(candidate)
        if int(summary["valid_paired_identities"]) >= min_valid_identities:
            rows.append(summary)
    rows.sort(key=lambda row: int(row["valid_paired_identities"]), reverse=True)
    return rows


def safe_link(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def make_paired_view(
    masked_root: Path,
    unmasked_root: Path,
    out_root: Path,
    max_images_per_identity: int | None,
    copy: bool,
) -> dict[str, object]:
    masked = collect_by_identity(masked_root)
    unmasked = collect_by_identity(unmasked_root)
    identities = sorted(set(masked) & set(unmasked))
    if not identities:
        raise SystemExit(f"No overlapping identities between {masked_root} and {unmasked_root}")
    for condition, grouped in [("masked", masked), ("unmasked", unmasked)]:
        for identity in identities:
            paths = grouped[identity]
            if max_images_per_identity:
                paths = paths[:max_images_per_identity]
            for idx, src in enumerate(paths):
                dst = out_root / condition / identity / f"{idx:05d}{src.suffix.lower()}"
                safe_link(src, dst, copy=copy)
    return summarize_root(out_root)


def maybe_download_dataset(args: argparse.Namespace, name: str, source: str) -> None:
    if source == "none":
        return
    if name == "rmfrd":
        out_dir = args.root / "rmfrd"
        if source == "gdrive":
            archive = download_gdown(args.rmfrd_url, args.root / "archives" / "rmfrd.zip")
            extract_archive(archive, out_dir / "extracted")
        elif source == "url":
            archive = download_url(args.rmfrd_url, archive_path_from_url(args.rmfrd_url, "rmfrd_archive", args.root / "archives"))
            extract_archive(archive, out_dir / "extracted")
        elif source == "kaggle":
            download_kaggle(args.rmfrd_kaggle_slug, out_dir / "kaggle", unzip=True)
        elif source == "archive":
            if not args.rmfrd_archive:
                raise SystemExit("--rmfrd-archive is required when --rmfrd-source archive")
            extract_archive(args.rmfrd_archive, out_dir / "extracted")
        else:
            raise SystemExit(f"Unsupported RMFRD source: {source}")
        return

    if name == "smfrd":
        out_dir = args.root / "smfrd"
        if source == "kaggle":
            download_kaggle(args.smfrd_kaggle_slug, out_dir / "kaggle", unzip=True)
        elif source == "url":
            if not args.smfrd_url:
                raise SystemExit("--smfrd-url is required when --smfrd-source url")
            archive = download_url(args.smfrd_url, args.root / "archives" / "smfrd_archive")
            extract_archive(archive, out_dir / "extracted")
        elif source == "gdrive":
            if not args.smfrd_url:
                raise SystemExit("--smfrd-url is required when --smfrd-source gdrive")
            archive = download_gdown(args.smfrd_url, args.root / "archives" / "smfrd.zip")
            extract_archive(archive, out_dir / "extracted")
        elif source == "archive":
            if not args.smfrd_archive:
                raise SystemExit("--smfrd-archive is required when --smfrd-source archive")
            extract_archive(args.smfrd_archive, out_dir / "extracted")
        else:
            raise SystemExit(f"Unsupported SMFRD source: {source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/content/datasets"))
    parser.add_argument("--rmfrd-source", choices=["none", "gdrive", "kaggle", "url", "archive"], default="none")
    parser.add_argument("--rmfrd-url", default=RMFRD_GDRIVE_URL)
    parser.add_argument("--rmfrd-kaggle-slug", default=RMFRD_KAGGLE_SLUG)
    parser.add_argument("--rmfrd-archive", type=Path, default=None)
    parser.add_argument("--smfrd-source", choices=["none", "kaggle", "gdrive", "url", "archive"], default="none")
    parser.add_argument("--smfrd-url", default="")
    parser.add_argument("--smfrd-kaggle-slug", default=LFW_SMFRD_KAGGLE_SLUG)
    parser.add_argument("--smfrd-archive", type=Path, default=None)
    parser.add_argument("--make-paired-view", action="store_true")
    parser.add_argument("--masked-root", type=Path, default=None)
    parser.add_argument("--unmasked-root", type=Path, default=None)
    parser.add_argument("--paired-view-root", type=Path, default=Path("/content/datasets/normalized/smfrd_paired"))
    parser.add_argument("--max-view-images-per-identity", type=int, default=0)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks for paired views.")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--summary-depth", type=int, default=4)
    parser.add_argument("--min-valid-identities", type=int, default=1)
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        maybe_download_dataset(args, "rmfrd", args.rmfrd_source)
        maybe_download_dataset(args, "smfrd", args.smfrd_source)

    paired_summary = None
    if args.make_paired_view:
        if not args.masked_root or not args.unmasked_root:
            raise SystemExit("--masked-root and --unmasked-root are required with --make-paired-view")
        paired_summary = make_paired_view(
            args.masked_root,
            args.unmasked_root,
            args.paired_view_root,
            max_images_per_identity=args.max_view_images_per_identity or None,
            copy=args.copy,
        )

    summaries = summarize_candidates(args.root, max_depth=args.summary_depth, min_valid_identities=args.min_valid_identities)
    print(json.dumps({"paired_view": paired_summary, "candidate_roots": summaries[:20]}, indent=2))


if __name__ == "__main__":
    main()
