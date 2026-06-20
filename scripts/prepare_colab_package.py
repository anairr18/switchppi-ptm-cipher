from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
COLAB = ROOT / "colab"
OUT = COLAB / "ptmppi_shield_colab_inputs.zip"

FILES = [
    ("results_v3/tables/event_table_v3.tsv", "event_table_v3.tsv"),
    ("results_v3/tables/interface_split_audit_v3.tsv", "interface_split_audit_v3.tsv"),
    ("results_v3/tables/shield_model_metrics_v3.tsv", "shield_model_metrics_v3.tsv"),
    ("results_v2/tables/external_validation_metrics_v2.tsv", "external_validation_metrics_v2.tsv"),
    ("validation/rrustemi2024_signed_external_validation.tsv", "rrustemi2024_signed_external_validation.tsv"),
    ("data/raw/uniprot_sequences.json", "uniprot_sequences.json"),
    ("results_v2/tables/deepphosppi_public_dataset_audit.tsv", "deepphosppi_public_dataset_audit.tsv"),
    ("results_v2/tables/external_method_reproducibility_inventory.tsv", "external_method_reproducibility_inventory.tsv"),
    ("results_v2/tables/structure_chain_uniprot_mapping_v2.tsv", "structure_chain_uniprot_mapping_v2.tsv"),
    ("results_v2/tables/structure_event_interface_mapping_v2.tsv", "structure_event_interface_mapping_v2.tsv"),
    ("results_v3/tables/ptm_cipher_input_manifest.tsv", "ptm_cipher_input_manifest.tsv"),
    ("scripts/ptm_cipher_model.py", "ptm_cipher_model.py"),
    ("docs/ptm_cipher_architecture.md", "ptm_cipher_architecture.md"),
]


def main() -> None:
    COLAB.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUT, "w", compression=ZIP_DEFLATED) as zf:
        for src, arcname in FILES:
            path = ROOT / src
            if path.exists():
                zf.write(path, arcname)
            else:
                print(f"Skipping missing file: {src}")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
