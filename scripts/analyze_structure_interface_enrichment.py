from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from map_ptmint_structure_events import AA3, STRUCT, TABLES, parse_pdb_chains


FIGURES = Path(__file__).resolve().parents[1] / "results_v2" / "figures"


def load_interface_sites() -> dict[tuple[str, str], set[int]]:
    interface = pd.read_csv(STRUCT / "complex_interface.csv")
    out: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in interface.itertuples(index=False):
        out[(str(row.Complex), str(row.Chain))].add(int(row.Site))
    return dict(out)


def same_residue_interface_rate(chain_rows: list[dict[str, object]], residue: str, interface_sites: set[int]) -> tuple[int, int, float]:
    residue = residue.upper()
    sites = [int(row["chain_site"]) for row in chain_rows if str(row["aa"]).upper() == residue]
    if not sites:
        return 0, 0, 0.0
    hits = sum(1 for site in sites if site in interface_sites)
    return len(sites), hits, hits / len(sites)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(TABLES / "structure_event_interface_mapping_v2.tsv", sep="\t")
    chains = pd.read_csv(TABLES / "structure_chain_uniprot_mapping_v2.tsv", sep="\t")
    if mapping.empty:
        raise SystemExit("No structure-event mapping available. Run map_ptmint_structure_events.py first.")

    interface_sites = load_interface_sites()
    chain_cache: dict[tuple[str, str], list[dict[str, object]]] = {}
    pdb_lookup = (
        chains[chains["mapped"].eq(True)]
        .drop_duplicates(["complex_id", "chain"])
        .set_index(["complex_id", "chain"])["pdb_file"]
        .to_dict()
    )

    rows = []
    for row in mapping.itertuples(index=False):
        key = (str(row.complex_id), str(row.modified_chain))
        if key not in chain_cache:
            chain_cache[key] = parse_pdb_chains(Path(pdb_lookup[key])).get(str(row.modified_chain), [])
        iface = interface_sites.get(key, set())
        residue_site_count, residue_interface_count, residue_null_rate = same_residue_interface_rate(
            chain_cache[key], str(row.residue), iface
        )
        expected_distance = None
        if iface and residue_site_count:
            same_residue_sites = [int(r["chain_site"]) for r in chain_cache[key] if str(r["aa"]).upper() == str(row.residue).upper()]
            expected_distance = sum(min(abs(site - iface_site) for iface_site in iface) for site in same_residue_sites) / len(same_residue_sites)
        rows.append(
            {
                "event_id": row.event_id,
                "complex_id": row.complex_id,
                "modified_chain": row.modified_chain,
                "effect_label": row.effect_label,
                "residue": row.residue,
                "site_at_interface": bool(row.site_at_interface),
                "distance_to_nearest_interface_site": row.distance_to_nearest_interface_site,
                "same_residue_sites_in_chain": residue_site_count,
                "same_residue_interface_sites_in_chain": residue_interface_count,
                "same_residue_interface_null_rate": residue_null_rate,
                "same_residue_expected_distance_to_interface": expected_distance,
            }
        )

    per_mapping = pd.DataFrame(rows)
    per_event = (
        per_mapping.groupby("event_id")
        .agg(
            effect_label=("effect_label", "first"),
            residue=("residue", "first"),
            site_at_interface=("site_at_interface", "max"),
            distance_to_nearest_interface_site=("distance_to_nearest_interface_site", "min"),
            same_residue_interface_null_rate=("same_residue_interface_null_rate", "mean"),
            same_residue_expected_distance_to_interface=("same_residue_expected_distance_to_interface", "mean"),
            structure_mappings=("complex_id", "nunique"),
        )
        .reset_index()
    )
    actual = float(per_event["site_at_interface"].mean())
    expected = float(per_event["same_residue_interface_null_rate"].mean())
    distance_actual = float(per_event["distance_to_nearest_interface_site"].dropna().mean())
    distance_expected = float(per_event["same_residue_expected_distance_to_interface"].dropna().mean())

    summary_rows = [
        {
            "scope": "all_structure_supported_events",
            "n_events": len(per_event),
            "actual_interface_fraction": actual,
            "same_residue_null_interface_fraction": expected,
            "interface_enrichment_ratio": actual / expected if expected else None,
            "actual_mean_distance_to_interface": distance_actual,
            "same_residue_null_mean_distance_to_interface": distance_expected,
            "distance_ratio_actual_over_null": distance_actual / distance_expected if distance_expected else None,
        }
    ]
    for label, group in per_event.groupby("effect_label"):
        actual_l = float(group["site_at_interface"].mean())
        expected_l = float(group["same_residue_interface_null_rate"].mean())
        distance_actual_l = float(group["distance_to_nearest_interface_site"].dropna().mean())
        distance_expected_l = float(group["same_residue_expected_distance_to_interface"].dropna().mean())
        summary_rows.append(
            {
                "scope": f"effect_{label}",
                "n_events": len(group),
                "actual_interface_fraction": actual_l,
                "same_residue_null_interface_fraction": expected_l,
                "interface_enrichment_ratio": actual_l / expected_l if expected_l else None,
                "actual_mean_distance_to_interface": distance_actual_l,
                "same_residue_null_mean_distance_to_interface": distance_expected_l,
                "distance_ratio_actual_over_null": distance_actual_l / distance_expected_l if distance_expected_l else None,
            }
        )
    summary = pd.DataFrame(summary_rows)

    per_mapping.to_csv(TABLES / "structure_interface_enrichment_by_mapping_v2.tsv", sep="\t", index=False)
    per_event.to_csv(TABLES / "structure_interface_enrichment_by_event_v2.tsv", sep="\t", index=False)
    summary.to_csv(TABLES / "structure_interface_enrichment_summary_v2.tsv", sep="\t", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot = summary.copy()
    plot["scope_label"] = plot["scope"].str.replace("effect_", "", regex=False).str.replace("_", " ")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(plot["scope_label"], plot["actual_interface_fraction"], label="PTM sites", color="#4C78A8")
    axes[0].scatter(plot["scope_label"], plot["same_residue_null_interface_fraction"], label="same-residue null", color="#E45756", zorder=3)
    axes[0].set_ylabel("Interface fraction")
    axes[0].set_title("Interface localization")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(frameon=False)

    axes[1].bar(plot["scope_label"], plot["interface_enrichment_ratio"], color="#54A24B")
    axes[1].axhline(1.0, color="#333333", linewidth=1)
    axes[1].set_ylabel("Fold enrichment")
    axes[1].set_title("Residue-matched enrichment")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_v2_structure_interface_enrichment.png", dpi=300)
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
