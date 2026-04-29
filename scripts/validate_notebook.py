#!/usr/bin/env python3
"""Static validation for the Colab spike notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "validation_spike.ipynb"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    if notebook.get("nbformat") != 4:
        raise SystemExit(f"Unexpected notebook format: {notebook.get('nbformat')}")

    code = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    compile(code, str(NOTEBOOK), "exec")
    print(f"validated {NOTEBOOK}")


if __name__ == "__main__":
    main()
