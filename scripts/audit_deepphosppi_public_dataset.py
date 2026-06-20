from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEEP = ROOT / "external_methods" / "DeepPhosPPI" / "Datasets" / "DatasetB"
TABLES = ROOT / "results_v2" / "tables"
FIGURES = ROOT / "results_v2" / "figures"


def norm_gene(value: object) -> str:
    return str(value or "").strip().upper()


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_deepphosppi_table() -> pd.DataFrame:
    rows_a = load_pickle(DEEP / "TASK2_TPhos_dataB_protein_list_all_AA.pkl")
    rows_b = load_pickle(DEEP / "TASK2_TPhos_dataB_protein_list_all_BB.pkl")
    train_idx = set(load_pickle(DEEP / "TASK2_all_train_samples.pkl"))
    test_idx = set(load_pickle(DEEP / "TASK2_all_test_samples.pkl"))

    records = []
    for idx, (row_a, row_b) in enumerate(zip(rows_a, rows_b)):
        id_a, gene_a, seq_a, pos_raw, residue, label = row_a
        id_b, gene_b, seq_b = row_b
        raw_1based_residue = seq_a[pos_raw - 1] if 1 <= pos_raw <= len(seq_a) else ""
        plus1_residue = seq_a[pos_raw] if 0 <= pos_raw < len(seq_a) else ""
        records.append(
            {
                "deepphosppi_idx": idx,
                "modified_gene": norm_gene(gene_a),
                "partner_gene": norm_gene(gene_b),
                "ptm_type": "Phos",
                "residue": str(residue).upper(),
                "position_raw": int(pos_raw),
                "position_plus1": int(pos_raw) + 1,
                "effect_label_deepphosppi": "enhance" if int(label) == 1 else "inhibit",
                "deepphosppi_label_numeric": int(label),
                "split_deepphosppi": "train" if idx in train_idx else ("test" if idx in test_idx else "unused"),
                "raw_position_residue_matches": raw_1based_residue == str(residue).upper(),
                "plus1_position_residue_matches": plus1_residue == str(residue).upper(),
                "modified_sequence_length": len(seq_a),
                "partner_sequence_length": len(seq_b),
                "modified_id": id_a,
                "partner_id": id_b,
            }
        )
    return pd.DataFrame(records)


def build_event_lookup() -> pd.DataFrame:
    event = pd.read_csv(TABLES / "event_table_v2.tsv", sep="\t")
    event = event[event["ptm_type"].eq("Phos")].copy()
    event["modified_gene_norm"] = event["modified_gene"].map(norm_gene)
    event["partner_gene_norm"] = event["partner_gene"].map(norm_gene)
    event["effect_label_event"] = event["effect_label"]
    return event[
        [
            "event_id",
            "modified_gene_norm",
            "partner_gene_norm",
            "ptm_type",
            "residue",
            "position",
            "effect_label_event",
            "pmid",
            "publication_year",
        ]
    ]


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    deep = build_deepphosppi_table()
    event = build_event_lookup()

    exact_raw = deep.merge(
        event,
        left_on=["modified_gene", "partner_gene", "ptm_type", "residue", "position_raw"],
        right_on=["modified_gene_norm", "partner_gene_norm", "ptm_type", "residue", "position"],
        how="left",
        suffixes=("", "_event"),
    )
    exact_plus1 = deep.merge(
        event,
        left_on=["modified_gene", "partner_gene", "ptm_type", "residue", "position_plus1"],
        right_on=["modified_gene_norm", "partner_gene_norm", "ptm_type", "residue", "position"],
        how="left",
        suffixes=("", "_event"),
    )

    overlap = exact_plus1.copy()
    overlap["matches_ptmint_v2_plus1_key"] = overlap["event_id"].notna()
    overlap["label_agrees_with_ptmint_v2"] = overlap["effect_label_deepphosppi"].eq(overlap["effect_label_event"])
    overlap["raw_key_matches_ptmint_v2"] = exact_raw["event_id"].notna()

    summary = pd.DataFrame(
        [
            {
                "metric": "deepphosppi_public_rows",
                "value": len(deep),
                "interpretation": "Rows in public DeepPhosPPI DatasetB protein list.",
            },
            {
                "metric": "raw_position_residue_match_rate",
                "value": deep["raw_position_residue_matches"].mean(),
                "interpretation": "Residue agreement if the released position is treated as one-based.",
            },
            {
                "metric": "plus1_position_residue_match_rate",
                "value": deep["plus1_position_residue_matches"].mean(),
                "interpretation": "Residue agreement if the released position is corrected by +1.",
            },
            {
                "metric": "raw_key_overlap_with_current_event_table",
                "value": exact_raw["event_id"].notna().sum(),
                "interpretation": "Exact gene/partner/site/effect-key overlap without position correction.",
            },
            {
                "metric": "plus1_key_overlap_with_current_event_table",
                "value": overlap["matches_ptmint_v2_plus1_key"].sum(),
                "interpretation": "Exact gene/partner/site-key overlap after +1 site correction.",
            },
            {
                "metric": "plus1_overlap_label_agreement_rate",
                "value": overlap.loc[overlap["matches_ptmint_v2_plus1_key"], "label_agrees_with_ptmint_v2"].mean(),
                "interpretation": "Agreement between DeepPhosPPI labels and the current normalized enhance/inhibit labels on overlapping +1-corrected rows.",
            },
            {
                "metric": "deepphosppi_train_rows_overlapping_current_event_table",
                "value": overlap[
                    overlap["matches_ptmint_v2_plus1_key"] & overlap["split_deepphosppi"].eq("train")
                ].shape[0],
                "interpretation": "DeepPhosPPI public train rows that appear in the current event table after +1 correction.",
            },
            {
                "metric": "deepphosppi_test_rows_overlapping_current_event_table",
                "value": overlap[
                    overlap["matches_ptmint_v2_plus1_key"] & overlap["split_deepphosppi"].eq("test")
                ].shape[0],
                "interpretation": "DeepPhosPPI public test rows that appear in the current event table after +1 correction.",
            },
        ]
    )

    deep.to_csv(TABLES / "deepphosppi_public_dataset_rows.tsv", sep="\t", index=False)
    overlap.to_csv(TABLES / "deepphosppi_public_dataset_overlap.tsv", sep="\t", index=False)
    summary.to_csv(TABLES / "deepphosppi_public_dataset_audit.tsv", sep="\t", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(
        ["released\nposition", "+1 corrected\nposition"],
        [
            deep["raw_position_residue_matches"].mean(),
            deep["plus1_position_residue_matches"].mean(),
        ],
        color=["#E45756", "#54A24B"],
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Residue match rate")
    axes[0].set_title("Site-coordinate convention")

    overlap_counts = pd.Series(
        {
            "not in current\nevent table": len(overlap) - int(overlap["matches_ptmint_v2_plus1_key"].sum()),
            "overlaps current\nevent table": int(overlap["matches_ptmint_v2_plus1_key"].sum()),
        }
    )
    axes[1].bar(overlap_counts.index, overlap_counts.values, color=["#BAB0AC", "#4C78A8"])
    axes[1].set_ylabel("DeepPhosPPI public rows")
    axes[1].set_title("Dataset lineage overlap")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_v2_deepphosppi_lineage_audit.png", dpi=300)
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
