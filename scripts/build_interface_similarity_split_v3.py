from __future__ import annotations

import itertools
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from map_ptmint_structure_events import PDB_DIR, STRUCT, TABLES, parse_pdb_chains
from run_ptmppi_shield_v2 import SEED, assign_group_split


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v3"
TABLES_V3 = RESULTS / "tables"
FIGURES_V3 = RESULTS / "figures"

CONTACT_RE = re.compile(r"^([A-Za-z0-9_]+):([A-Z]{3})(-?\d+)")
JACCARD_THRESHOLD = 0.25


def complex_id_from_path(path: str | Path) -> str:
    stem = Path(path).stem
    match = re.search(r"(Complex\d+(?:_\d+)?)$", stem)
    return match.group(1) if match else stem


def parse_contact_site(value: object) -> tuple[str, str, int] | None:
    match = CONTACT_RE.match(str(value))
    if not match:
        return None
    return match.group(1), match.group(2), int(match.group(3))


def load_position_maps(chain_mapping: pd.DataFrame) -> dict[tuple[str, str, str], dict[int, int]]:
    """Return (complex, chain, uniprot) -> chain_site -> uniprot_position."""
    pos_maps: dict[tuple[str, str, str], dict[int, int]] = {}
    unique = chain_mapping[chain_mapping["mapped"].eq(True) & chain_mapping["mapping_unique"].eq(True)].drop_duplicates(
        ["complex_id", "chain", "uniprot"]
    )
    for row in unique.itertuples(index=False):
        chain_rows = parse_pdb_chains(Path(row.pdb_file)).get(str(row.chain), [])
        start = int(row.uniprot_start)
        pos_maps[(str(row.complex_id), str(row.chain), str(row.uniprot))] = {
            int(entry["chain_site"]): start + i for i, entry in enumerate(chain_rows)
        }
    return pos_maps


def load_contacts(chain_mapping: pd.DataFrame) -> pd.DataFrame:
    contacts = pd.read_csv(STRUCT / "complex_interaction.csv")
    mapped = chain_mapping[chain_mapping["mapped"].eq(True) & chain_mapping["mapping_unique"].eq(True)].copy()
    chain_to_uniprot = {
        (str(row.complex_id), str(row.chain)): str(row.uniprot)
        for row in mapped.itertuples(index=False)
    }
    pos_maps = load_position_maps(mapped)
    rows = []
    for rec in contacts.itertuples(index=False):
        complex_id = str(rec.Complex)
        left = parse_contact_site(rec.AA1)
        right = parse_contact_site(rec.AA2)
        if left is None or right is None:
            continue
        chain_a, aa_a, site_a = left
        chain_b, aa_b, site_b = right
        uniprot_a = chain_to_uniprot.get((complex_id, chain_a))
        uniprot_b = chain_to_uniprot.get((complex_id, chain_b))
        if not uniprot_a or not uniprot_b:
            continue
        upos_a = pos_maps.get((complex_id, chain_a, uniprot_a), {}).get(site_a)
        upos_b = pos_maps.get((complex_id, chain_b, uniprot_b), {}).get(site_b)
        if upos_a is None or upos_b is None:
            continue
        rows.append(
            {
                "complex_id": complex_id,
                "chain_a": chain_a,
                "chain_b": chain_b,
                "uniprot_a": uniprot_a,
                "uniprot_b": uniprot_b,
                "upos_a": int(upos_a),
                "upos_b": int(upos_b),
                "aa_a": aa_a,
                "aa_b": aa_b,
                "contact_type": str(rec.Type),
            }
        )
    return pd.DataFrame(rows)


def event_contact_signatures(event_mapping: pd.DataFrame, contacts: pd.DataFrame) -> pd.DataFrame:
    by_complex = {cid: group for cid, group in contacts.groupby("complex_id")}
    rows = []
    for rec in event_mapping.itertuples(index=False):
        partner_chains = {part.strip() for part in str(rec.partner_chains).split(",") if part.strip()}
        sub = by_complex.get(str(rec.complex_id), pd.DataFrame())
        tokens: set[str] = set()
        contact_count = 0
        if not sub.empty:
            for contact in sub.itertuples(index=False):
                forward = contact.chain_a == rec.modified_chain and contact.uniprot_a == rec.modified_uniprot and contact.uniprot_b == rec.partner_uniprot
                reverse = contact.chain_b == rec.modified_chain and contact.uniprot_b == rec.modified_uniprot and contact.uniprot_a == rec.partner_uniprot
                if forward and (not partner_chains or contact.chain_b in partner_chains):
                    mod_pos, partner_pos = int(contact.upos_a), int(contact.upos_b)
                elif reverse and (not partner_chains or contact.chain_a in partner_chains):
                    mod_pos, partner_pos = int(contact.upos_b), int(contact.upos_a)
                else:
                    continue
                contact_count += 1
                rel = mod_pos - int(rec.position)
                tokens.add(f"M:{rec.modified_uniprot}:{mod_pos}")
                tokens.add(f"P:{rec.partner_uniprot}:{partner_pos}")
                tokens.add(f"REL:{rel:+d}:{contact.contact_type}")
                tokens.add(f"AA:{rec.residue}:{contact.contact_type}")
        if not tokens and pd.notna(rec.distance_to_nearest_interface_site):
            tokens.add(f"IFACE_NEAR:{rec.modified_uniprot}:{int(rec.distance_to_nearest_interface_site)}")
        rows.append(
            {
                "event_id": rec.event_id,
                "complex_id": rec.complex_id,
                "modified_chain": rec.modified_chain,
                "signature_tokens": "|".join(sorted(tokens)),
                "signature_size": len(tokens),
                "mapped_contact_count": contact_count,
            }
        )
    return pd.DataFrame(rows)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def cluster_events(event_table: pd.DataFrame, signatures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_tokens: dict[str, set[str]] = defaultdict(set)
    event_contact_counts: dict[str, int] = defaultdict(int)
    event_complexes: dict[str, set[str]] = defaultdict(set)
    for rec in signatures.itertuples(index=False):
        event_tokens[str(rec.event_id)].update(token for token in str(rec.signature_tokens).split("|") if token)
        event_contact_counts[str(rec.event_id)] += int(rec.mapped_contact_count)
        event_complexes[str(rec.event_id)].add(str(rec.complex_id))

    graph = nx.Graph()
    graph.add_nodes_from(event_table["event_id"].astype(str))
    token_buckets: dict[str, list[str]] = defaultdict(list)
    for event_id, tokens in event_tokens.items():
        for token in tokens:
            token_buckets[token].append(event_id)

    comparisons = 0
    edges = []
    candidates: set[tuple[str, str]] = set()
    for members in token_buckets.values():
        if len(members) > 250:
            continue
        for a, b in itertools.combinations(sorted(set(members)), 2):
            candidates.add((a, b))
    for a, b in candidates:
        score = jaccard(event_tokens[a], event_tokens[b])
        comparisons += 1
        if score >= JACCARD_THRESHOLD:
            graph.add_edge(a, b, jaccard=score)
            edges.append({"event_id_a": a, "event_id_b": b, "contact_jaccard": score})

    cluster = {}
    for cid, component in enumerate(nx.connected_components(graph)):
        cluster_id = f"interface_cluster_{cid}"
        for event_id in component:
            cluster[event_id] = cluster_id

    annotation = event_table[["event_id", "modified_uniprot", "partner_uniprot"]].copy()
    annotation["event_id"] = annotation["event_id"].astype(str)
    annotation["interface_signature_size"] = annotation["event_id"].map(lambda x: len(event_tokens.get(x, set()))).fillna(0).astype(int)
    annotation["interface_contact_count"] = annotation["event_id"].map(lambda x: event_contact_counts.get(x, 0)).fillna(0).astype(int)
    annotation["structure_complexes"] = annotation["event_id"].map(lambda x: ",".join(sorted(event_complexes.get(x, set()))))
    annotation["interface_cluster_id"] = annotation["event_id"].map(cluster)
    fallback_pair = annotation.apply(
        lambda r: "unmapped_pair_" + "::".join(sorted([str(r["modified_uniprot"]), str(r["partner_uniprot"])])),
        axis=1,
    )
    annotation["interface_cluster_id"] = annotation["interface_cluster_id"].fillna(fallback_pair)
    edge_table = pd.DataFrame(edges)
    edge_table.attrs["comparisons"] = comparisons
    return annotation, edge_table


def audit_split(df: pd.DataFrame) -> pd.DataFrame:
    groups = {
        part: set(df.loc[df["S2b_cold_interface_split"].eq(part), "interface_cluster_id"].dropna().astype(str))
        for part in ["train", "valid", "test"]
    }
    return pd.DataFrame(
        [
            {
                "split_col": "S2b_cold_interface_split",
                "held_out_axis": "interface_cluster_id_contact_jaccard",
                "jaccard_threshold": JACCARD_THRESHOLD,
                "train_rows": int(df["S2b_cold_interface_split"].eq("train").sum()),
                "valid_rows": int(df["S2b_cold_interface_split"].eq("valid").sum()),
                "test_rows": int(df["S2b_cold_interface_split"].eq("test").sum()),
                "train_valid_overlap": len(groups["train"] & groups["valid"]),
                "train_test_overlap": len(groups["train"] & groups["test"]),
                "valid_test_overlap": len(groups["valid"] & groups["test"]),
                "mapped_events_with_contact_signature": int(df["interface_signature_size"].gt(0).sum()),
                "mapped_events_with_contacts": int(df["interface_contact_count"].gt(0).sum()),
                "interface_clusters": int(df["interface_cluster_id"].nunique()),
                "test_pos_rate": float(df.loc[df["S2b_cold_interface_split"].eq("test"), "label_binary"].mean()),
            }
        ]
    )


def write_figures(df: pd.DataFrame, edges: pd.DataFrame) -> None:
    FIGURES_V3.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    df["S2b_cold_interface_split"].value_counts().reindex(["train", "valid", "test"]).plot(kind="bar", ax=axes[0], color="#4C78A8")
    axes[0].set_title("Cold-interface split rows")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Events")
    if len(edges):
        axes[1].hist(edges["contact_jaccard"], bins=30, color="#54A24B")
    else:
        axes[1].hist([0.0], bins=1, color="#54A24B")
    axes[1].axvline(JACCARD_THRESHOLD, color="#E45756", linewidth=1)
    axes[1].set_title("Clustered contact-Jaccard edges")
    axes[1].set_xlabel("Jaccard")
    axes[1].set_ylabel("Event pairs")
    fig.tight_layout()
    fig.savefig(FIGURES_V3 / "figure_v3_cold_interface_split.png", dpi=300)
    plt.close(fig)


def main() -> None:
    TABLES_V3.mkdir(parents=True, exist_ok=True)
    FIGURES_V3.mkdir(parents=True, exist_ok=True)
    event = pd.read_csv(TABLES / "event_table_v2.tsv", sep="\t")
    event["event_id"] = event["event_id"].astype(str)
    event["label_binary"] = event["effect_label"].map({"inhibit": 0, "enhance": 1}).astype(int)
    chain_mapping = pd.read_csv(TABLES / "structure_chain_uniprot_mapping_v2.tsv", sep="\t")
    event_mapping = pd.read_csv(TABLES / "structure_event_interface_mapping_v2.tsv", sep="\t")
    event_mapping["event_id"] = event_mapping["event_id"].astype(str)

    contacts = load_contacts(chain_mapping)
    signatures = event_contact_signatures(event_mapping, contacts)
    annotation, edges = cluster_events(event, signatures)
    out = event.merge(annotation.drop(columns=["modified_uniprot", "partner_uniprot"]), on="event_id", how="left")
    out["interface_signature_size"] = out["interface_signature_size"].fillna(0).astype(int)
    out["interface_contact_count"] = out["interface_contact_count"].fillna(0).astype(int)
    out["interface_cluster_id"] = out["interface_cluster_id"].fillna(
        out.apply(lambda r: "unmapped_pair_" + "::".join(sorted([str(r["modified_uniprot"]), str(r["partner_uniprot"])])), axis=1)
    )
    out["S2b_cold_interface_split"] = assign_group_split(out, "interface_cluster_id", SEED + 22)
    audit = audit_split(out)

    out.to_csv(TABLES_V3 / "event_table_v3.tsv", sep="\t", index=False)
    signatures.to_csv(TABLES_V3 / "interface_contact_signatures_v3.tsv", sep="\t", index=False)
    annotation.to_csv(TABLES_V3 / "interface_cluster_annotations_v3.tsv", sep="\t", index=False)
    edges.to_csv(TABLES_V3 / "interface_contact_jaccard_edges_v3.tsv", sep="\t", index=False)
    audit.to_csv(TABLES_V3 / "interface_split_audit_v3.tsv", sep="\t", index=False)
    write_figures(out, edges)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
