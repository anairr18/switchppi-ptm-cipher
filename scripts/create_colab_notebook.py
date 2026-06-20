from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "colab" / "ptmppi_shield_gpu_baselines_colab.py"
OUT = ROOT / "colab" / "ptmppi_shield_gpu_baselines_colab.ipynb"


def parse_percent_cells(text: str) -> list[dict]:
    cells = []
    current_type = "code"
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines, current_type
        if not current_lines:
            return
        source = "\n".join(current_lines).rstrip() + "\n"
        if current_type == "markdown":
            cleaned = []
            for line in source.splitlines():
                cleaned.append(line[2:] if line.startswith("# ") else line)
            cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in cleaned]})
        else:
            cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in source.splitlines()]})
        current_lines = []

    for raw in text.splitlines():
        if raw.startswith("# %%"):
            flush()
            current_type = "markdown" if "[markdown]" in raw else "code"
            continue
        current_lines.append(raw)
    flush()
    return cells


def main() -> None:
    notebook = {
        "cells": parse_percent_cells(SRC.read_text(encoding="utf-8")),
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
