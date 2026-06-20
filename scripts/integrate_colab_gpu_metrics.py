from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES_V3 = ROOT / "results_v3" / "tables"
DEFAULT_COLAB_METRICS = TABLES_V3 / "shield_gpu_baseline_metrics_colab.tsv"
DEFAULT_COLAB_NOTES = TABLES_V3 / "shield_gpu_baseline_reproducibility_notes.tsv"


def main() -> None:
    if not DEFAULT_COLAB_METRICS.exists():
        raise FileNotFoundError(
            f"Missing {DEFAULT_COLAB_METRICS}. Copy it from the downloaded Colab output zip first."
        )
    local = pd.read_csv(TABLES_V3 / "shield_model_metrics_v3.tsv", sep="\t")
    gpu = pd.read_csv(DEFAULT_COLAB_METRICS, sep="\t")
    gpu["source"] = "colab_gpu"
    local["source"] = local.get("source", "local_cpu")
    merged = pd.concat([local, gpu], ignore_index=True, sort=False)
    merged.to_csv(TABLES_V3 / "shield_model_metrics_v3_with_gpu.tsv", sep="\t", index=False)
    if DEFAULT_COLAB_NOTES.exists():
        notes = pd.read_csv(DEFAULT_COLAB_NOTES, sep="\t")
        notes.to_csv(TABLES_V3 / "baseline_reproducibility_notes_v3.tsv", sep="\t", index=False)
    print(f"Wrote {TABLES_V3 / 'shield_model_metrics_v3_with_gpu.tsv'}")


if __name__ == "__main__":
    main()
