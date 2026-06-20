from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from build_interface_similarity_split_v3 import load_contacts


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TABLES_V3 = ROOT / "results_v3" / "tables"

MAX_MOD_LEN = 384
MAX_PARTNER_LEN = 384

PTM_TO_STATE = {
    "Phos": "phospho",
    "Ac": "acetyl",
    "Me": "methyl",
    "Sumo": "sumoyl",
    "Ub": "ubiquityl",
    "Glyco": "glycosyl",
}


def clean_seq(seq: str) -> str:
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join(ch if ch in allowed else "X" for ch in str(seq).upper())


def crop_around_position(seq: str, position: int, max_len: int) -> tuple[str, int, int]:
    seq = clean_seq(seq)
    if len(seq) <= max_len:
        return seq, 0, int(position) - 1
    center = int(position) - 1
    start = max(0, min(center - max_len // 2, len(seq) - max_len))
    end = start + max_len
    return seq[start:end], start, center - start


def crop_partner(seq: str, contact_positions: set[int], max_len: int) -> tuple[str, int]:
    seq = clean_seq(seq)
    if len(seq) <= max_len:
        return seq, 0
    if contact_positions:
        center = int(sorted(contact_positions)[len(contact_positions) // 2]) - 1
    else:
        center = len(seq) // 2
    start = max(0, min(center - max_len // 2, len(seq) - max_len))
    return seq[start : start + max_len], start


def build_event_contact_pairs() -> dict[str, list[tuple[int, int]]]:
    chain_mapping = pd.read_csv(TABLES_V3.parent.parent / "results_v2" / "tables" / "structure_chain_uniprot_mapping_v2.tsv", sep="\t")
    event_mapping = pd.read_csv(TABLES_V3.parent.parent / "results_v2" / "tables" / "structure_event_interface_mapping_v2.tsv", sep="\t")
    contacts = load_contacts(chain_mapping)
    by_complex = {cid: group for cid, group in contacts.groupby("complex_id")}
    out: dict[str, set[tuple[int, int]]] = {}
    for rec in event_mapping.itertuples(index=False):
        partner_chains = {part.strip() for part in str(rec.partner_chains).split(",") if part.strip()}
        sub = by_complex.get(str(rec.complex_id), pd.DataFrame())
        if sub.empty:
            continue
        event_pairs = out.setdefault(str(rec.event_id), set())
        for contact in sub.itertuples(index=False):
            forward = (
                contact.chain_a == rec.modified_chain
                and contact.uniprot_a == rec.modified_uniprot
                and contact.uniprot_b == rec.partner_uniprot
            )
            reverse = (
                contact.chain_b == rec.modified_chain
                and contact.uniprot_b == rec.modified_uniprot
                and contact.uniprot_a == rec.partner_uniprot
            )
            if forward and (not partner_chains or contact.chain_b in partner_chains):
                event_pairs.add((int(contact.upos_a), int(contact.upos_b)))
            elif reverse and (not partner_chains or contact.chain_a in partner_chains):
                event_pairs.add((int(contact.upos_b), int(contact.upos_a)))
    return {key: sorted(value) for key, value in out.items()}


def remap_pairs(pairs: list[tuple[int, int]], mod_start: int, partner_start: int, mod_len: int, partner_len: int) -> str:
    out = []
    seen = set()
    for mod_pos, partner_pos in pairs:
        i = int(mod_pos) - 1 - mod_start
        j = int(partner_pos) - 1 - partner_start
        if 0 <= i < mod_len and 0 <= j < partner_len and (i, j) not in seen:
            seen.add((i, j))
            out.append(f"{i}:{j}")
    return ";".join(out)


def main() -> None:
    event = pd.read_csv(TABLES_V3 / "event_table_v3.tsv", sep="\t")
    seqs = json.loads((RAW / "uniprot_sequences.json").read_text(encoding="utf-8"))
    pairs_by_event = build_event_contact_pairs()
    rows = []
    for rec in event.itertuples(index=False):
        pairs = pairs_by_event.get(str(rec.event_id), [])
        partner_positions = {partner_pos for _, partner_pos in pairs}
        mod_seq, mod_start, ptm_index = crop_around_position(seqs[str(rec.modified_uniprot)], int(rec.position), MAX_MOD_LEN)
        partner_seq, partner_start = crop_partner(seqs[str(rec.partner_uniprot)], partner_positions, MAX_PARTNER_LEN)
        contact_pairs = remap_pairs(pairs, mod_start, partner_start, len(mod_seq), len(partner_seq))
        rows.append(
            {
                "event_id": rec.event_id,
                "modified_uniprot": rec.modified_uniprot,
                "partner_uniprot": rec.partner_uniprot,
                "modified_gene": rec.modified_gene,
                "partner_gene": rec.partner_gene,
                "ptm_type": rec.ptm_type,
                "ptm_state": PTM_TO_STATE.get(str(rec.ptm_type), "other"),
                "residue": rec.residue,
                "position": int(rec.position),
                "effect_label": rec.effect_label,
                "label_binary": 1 if rec.effect_label == "enhance" else 0,
                "mod_seq_crop": mod_seq,
                "partner_seq_crop": partner_seq,
                "mod_crop_start_0based": mod_start,
                "partner_crop_start_0based": partner_start,
                "ptm_index_crop_0based": ptm_index,
                "contact_pairs_crop": contact_pairs,
                "contact_pair_count": 0 if not contact_pairs else len(contact_pairs.split(";")),
                "assay_family": rec.assay_family,
                "pmid": rec.pmid,
                "topology_pair_community": rec.topology_pair_community,
                "interface_cluster_id": rec.interface_cluster_id,
                "S2b_cold_interface_split": rec.S2b_cold_interface_split,
                "S7_temporal_prospective_split": rec.S7_temporal_prospective_split,
                "S9_full_shield_split": rec.S9_full_shield_split,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_V3 / "ptm_cipher_input_manifest.tsv", sep="\t", index=False)
    print(
        out[["contact_pair_count", "label_binary"]]
        .agg({"contact_pair_count": ["count", "mean", "max"], "label_binary": ["mean", "sum"]})
        .to_string()
    )
    print(f"Wrote {TABLES_V3 / 'ptm_cipher_input_manifest.tsv'}")


if __name__ == "__main__":
    main()
