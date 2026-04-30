#!/usr/bin/env python3
"""Scan an RMFD-style root for all masked/unmasked paired identities."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from setup_masked_datasets import IMAGE_EXTS, normalize_condition


def image_count(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def collect_identity_dirs(condition_root: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    identity_dirs = [path for path in condition_root.iterdir() if path.is_dir()]
    if identity_dirs:
        for identity_dir in sorted(identity_dirs):
            rows[identity_dir.name] = {
                "dir_exists": True,
                "images": image_count(identity_dir),
                "path": str(identity_dir),
            }
        return rows

    grouped: dict[str, int] = {}
    for path in sorted(p for p in condition_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        identity = path.stem.split("_")[0].split("-")[0]
        grouped[identity] = grouped.get(identity, 0) + 1
    for identity, count in grouped.items():
        rows[identity] = {
            "dir_exists": False,
            "images": count,
            "path": str(condition_root),
        }
    return rows


def condition_roots(root: Path) -> dict[str, list[Path]]:
    roots = {"masked": [], "unmasked": []}
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        condition = normalize_condition(child.name)
        if condition:
            roots[condition].append(child)
    return roots


def scan(root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    roots = condition_roots(root)
    grouped: dict[str, dict[str, dict[str, object]]] = {"masked": {}, "unmasked": {}}
    for condition, paths in roots.items():
        for condition_root in paths:
            for identity, row in collect_identity_dirs(condition_root).items():
                current = grouped[condition].setdefault(
                    identity,
                    {"dir_exists": False, "images": 0, "paths": []},
                )
                current["dir_exists"] = bool(current["dir_exists"]) or bool(row["dir_exists"])
                current["images"] = int(current["images"]) + int(row["images"])
                current["paths"].append(row["path"])

    identities = sorted(set(grouped["masked"]) | set(grouped["unmasked"]))
    rows = []
    for identity in identities:
        masked = grouped["masked"].get(identity, {"dir_exists": False, "images": 0, "paths": []})
        unmasked = grouped["unmasked"].get(identity, {"dir_exists": False, "images": 0, "paths": []})
        rows.append(
            {
                "identity": identity,
                "masked_dir_exists": bool(masked["dir_exists"]),
                "unmasked_dir_exists": bool(unmasked["dir_exists"]),
                "masked_images": int(masked["images"]),
                "unmasked_images": int(unmasked["images"]),
                "has_both_dirs": bool(masked["dir_exists"]) and bool(unmasked["dir_exists"]),
                "has_both_nonempty": int(masked["images"]) > 0 and int(unmasked["images"]) > 0,
                "masked_paths": ";".join(masked["paths"]),
                "unmasked_paths": ";".join(unmasked["paths"]),
            }
        )
    identity_rows = len(rows)
    masked_identity_dirs = sum(1 for row in rows if row["masked_dir_exists"])
    unmasked_identity_dirs = sum(1 for row in rows if row["unmasked_dir_exists"])
    identities_with_both_dirs = sum(1 for row in rows if row["has_both_dirs"])
    masked_nonempty_identities = sum(1 for row in rows if int(row["masked_images"]) > 0)
    unmasked_nonempty_identities = sum(1 for row in rows if int(row["unmasked_images"]) > 0)
    usable_paired_identities = sum(1 for row in rows if row["has_both_nonempty"])
    empty_masked_dirs = sum(1 for row in rows if row["masked_dir_exists"] and int(row["masked_images"]) == 0)
    empty_unmasked_dirs = sum(1 for row in rows if row["unmasked_dir_exists"] and int(row["unmasked_images"]) == 0)
    summary = {
        "root": str(root),
        "condition_roots": {key: [str(path) for path in value] for key, value in roots.items()},
        "identity_rows": identity_rows,
        "masked_identity_dirs": masked_identity_dirs,
        "unmasked_identity_dirs": unmasked_identity_dirs,
        "identities_with_both_dirs": identities_with_both_dirs,
        "masked_nonempty_identities": masked_nonempty_identities,
        "unmasked_nonempty_identities": unmasked_nonempty_identities,
        "usable_paired_identities": usable_paired_identities,
        "masked_images": sum(int(row["masked_images"]) for row in rows),
        "unmasked_images": sum(int(row["unmasked_images"]) for row in rows),
        "empty_masked_dirs": empty_masked_dirs,
        "empty_unmasked_dirs": empty_unmasked_dirs,
    }
    return summary, rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(summary: dict[str, object], out_dir: Path) -> None:
    text = f"""# RMFD Paired Identity Scan

This artifact scans the full RMFD archive for identities that have both masked
and unmasked images.

- Data root: `{summary["root"]}`
- Masked condition roots: `{summary["condition_roots"]["masked"]}`
- Unmasked condition roots: `{summary["condition_roots"]["unmasked"]}`
- Identity rows seen across either condition: {summary["identity_rows"]}
- Masked identity directories: {summary["masked_identity_dirs"]}
- Unmasked identity directories: {summary["unmasked_identity_dirs"]}
- Identities with both directories, including empty dirs: {summary["identities_with_both_dirs"]}
- Usable paired identities with at least one image in both conditions: {summary["usable_paired_identities"]}
- Masked nonempty identities: {summary["masked_nonempty_identities"]}
- Unmasked nonempty identities: {summary["unmasked_nonempty_identities"]}
- Masked images: {summary["masked_images"]}
- Unmasked images: {summary["unmasked_images"]}
- Empty masked identity dirs: {summary["empty_masked_dirs"]}
- Empty unmasked identity dirs: {summary["empty_unmasked_dirs"]}

Interpretation: the archive exposes more identity directory overlaps than usable
paired identities because some masked identity directories are empty. The
usable count is the defensible count for training/evaluation.
"""
    (out_dir / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary, rows = scan(args.data_root)
    write_csv(rows, args.out_dir / "paired_identity_counts.csv")
    (args.out_dir / "paired_identity_scan_summary.json").write_text(json.dumps(summary, indent=2))
    write_readme(summary, args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
