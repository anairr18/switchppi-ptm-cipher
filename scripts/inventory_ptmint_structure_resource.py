from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STRUCT = ROOT / "data" / "raw" / "ptmint_structure_information" / "Protein structure information"
TABLES = ROOT / "results_v2" / "tables"


def main() -> None:
    complex_interaction = STRUCT / "complex_interaction.csv"
    complex_interface = STRUCT / "complex_interface.csv"
    pdb_dir = STRUCT / "Complex structures"
    interactions = pd.read_csv(complex_interaction) if complex_interaction.exists() else pd.DataFrame()
    interfaces = pd.read_csv(complex_interface) if complex_interface.exists() else pd.DataFrame()
    pdbs = list(pdb_dir.glob("*.pdb")) if pdb_dir.exists() else []
    mapping_summary_path = TABLES / "structure_event_interface_summary_v2.tsv"
    has_mapping = False
    mapped_events = 0
    site_interface_events = 0
    if mapping_summary_path.exists():
        summary = pd.read_csv(mapping_summary_path, sep="\t").set_index("metric")["value"]
        mapped_events = int(float(summary.get("structure_supported_event_rows", 0)))
        site_interface_events = int(float(summary.get("site_at_interface_events", 0)))
        has_mapping = mapped_events > 0
    rows = [
        {
            "resource": "PTMint protein structure information",
            "downloaded": bool(STRUCT.exists()),
            "pdb_files": len(pdbs),
            "complex_interaction_rows": len(interactions),
            "complex_interface_rows": len(interfaces),
            "unique_complexes_in_interactions": interactions["Complex"].nunique() if "Complex" in interactions else 0,
            "unique_complexes_in_interfaces": interfaces["Complex"].nunique() if "Complex" in interfaces else 0,
            "has_uniprot_or_event_mapping": has_mapping,
            "mapped_event_rows": mapped_events,
            "site_at_interface_event_rows": site_interface_events,
            "usable_for_current_interface_shield": has_mapping,
            "reason_not_used": "" if has_mapping else "Archive exposes complex IDs/contact residues but no discovered UniProt/event mapping in CSV/PDB headers; using it for interface-cold splits would risk fabricated mapping.",
            "next_step": "Add interface-similarity clusters with Foldseek/TM-align/contact-Jaccard on the mapped structures; current mapping supports interface localization but not cold-interface similarity shielding.",
        }
    ]
    out = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / "structure_resource_inventory_v2.tsv", sep="\t", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
