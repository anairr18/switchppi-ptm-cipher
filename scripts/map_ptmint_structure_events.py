from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STRUCT = ROOT / "data" / "raw" / "ptmint_structure_information" / "Protein structure information"
PDB_DIR = STRUCT / "Complex structures"
FASTA = ROOT / "data" / "raw" / "uniprot_ptmint_accessions.fasta"
TABLES = ROOT / "results_v2" / "tables"

AA3 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}


def load_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    acc: str | None = None
    parts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if acc:
                seqs[acc] = "".join(parts)
            match = re.search(r"\|([^|]+)\|", line)
            acc = match.group(1) if match else line[1:].split()[0]
            parts = []
        else:
            parts.append(line)
    if acc:
        seqs[acc] = "".join(parts)
    return seqs


def build_kmer_index(seqs: dict[str, str], k: int = 20) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for accession, sequence in seqs.items():
        if len(sequence) < k:
            continue
        for i in range(0, len(sequence) - k + 1):
            index[sequence[i : i + k]].add(accession)
    return dict(index)


def complex_id_from_path(path: Path) -> str:
    stem = path.stem
    match = re.search(r"(Complex\d+(?:_\d+)?)$", stem)
    return match.group(1) if match else stem


def parse_pdb_chains(path: Path) -> dict[str, list[dict[str, object]]]:
    chains: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            if atom != "CA":
                continue
            residue = line[17:20].strip()
            aa = AA3.get(residue, "X")
            chain = line[21].strip() or "_"
            resnum_text = line[22:26].strip()
            insertion = line[26].strip()
            key = (chain, resnum_text, insertion)
            if key in seen:
                continue
            seen.add(key)
            try:
                resnum = int(resnum_text)
            except ValueError:
                continue
            chains[chain].append({"chain_site": resnum, "aa": aa, "aa3": residue})
    return dict(chains)


def map_chain_sequence(
    chain_rows: list[dict[str, object]],
    seqs: dict[str, str],
    kmer_index: dict[str, set[str]],
    k: int = 20,
) -> list[dict[str, object]]:
    sequence = "".join(str(row["aa"]) for row in chain_rows)
    if len(sequence) < k or "X" in sequence:
        return []
    kmers = [sequence[:k], sequence[max(0, len(sequence) // 2 - k // 2) : max(0, len(sequence) // 2 - k // 2) + k], sequence[-k:]]
    candidate_sets = [kmer_index.get(kmer, set()) for kmer in kmers if len(kmer) == k]
    if not candidate_sets:
        return []
    candidates = set.intersection(*candidate_sets) if len(candidate_sets) > 1 else set(candidate_sets[0])
    matches = []
    for accession in candidates:
        uniprot_seq = seqs[accession]
        offset = uniprot_seq.find(sequence)
        if offset >= 0:
            matches.append(
                {
                    "uniprot": accession,
                    "uniprot_start": offset + 1,
                    "uniprot_end": offset + len(sequence),
                    "chain_length": len(sequence),
                    "sequence_match_type": "exact_contained",
                }
            )
    return matches


def load_interface_sites() -> dict[tuple[str, str], set[int]]:
    path = STRUCT / "complex_interface.csv"
    if not path.exists():
        return {}
    interface = pd.read_csv(path)
    out: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in interface.itertuples(index=False):
        out[(str(row.Complex), str(row.Chain))].add(int(row.Site))
    return dict(out)


def load_contact_sites() -> dict[tuple[str, str], set[int]]:
    path = STRUCT / "complex_interaction.csv"
    if not path.exists():
        return {}
    contacts = pd.read_csv(path)
    pattern = re.compile(r"^([A-Za-z0-9_]+):[A-Z]{3}(-?\d+)")
    out: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in contacts.itertuples(index=False):
        complex_id = str(row.Complex)
        for field in (str(row.AA1), str(row.AA2)):
            match = pattern.match(field)
            if match:
                out[(complex_id, match.group(1))].add(int(match.group(2)))
    return dict(out)


def build_chain_mapping(seqs: dict[str, str], kmer_index: dict[str, set[str]]) -> tuple[pd.DataFrame, dict[tuple[str, str], list[dict[str, object]]]]:
    records = []
    chain_cache: dict[tuple[str, str], list[dict[str, object]]] = {}
    for pdb in sorted(PDB_DIR.glob("*.pdb")):
        complex_id = complex_id_from_path(pdb)
        chains = parse_pdb_chains(pdb)
        for chain, rows in chains.items():
            chain_cache[(complex_id, chain)] = rows
            sequence = "".join(str(row["aa"]) for row in rows)
            matches = map_chain_sequence(rows, seqs, kmer_index)
            if not matches:
                records.append(
                    {
                        "complex_id": complex_id,
                        "pdb_file": str(pdb),
                        "chain": chain,
                        "chain_length": len(sequence),
                        "mapped": False,
                        "mapping_unique": False,
                        "uniprot": "",
                        "uniprot_start": "",
                        "uniprot_end": "",
                        "sequence_match_type": "no_exact_contained_match",
                    }
                )
                continue
            for match in matches:
                records.append(
                    {
                        "complex_id": complex_id,
                        "pdb_file": str(pdb),
                        "chain": chain,
                        "mapped": True,
                        "mapping_unique": len(matches) == 1,
                        **match,
                    }
                )
    return pd.DataFrame(records), chain_cache


def build_position_maps(
    chain_mapping: pd.DataFrame, chain_cache: dict[tuple[str, str], list[dict[str, object]]]
) -> dict[tuple[str, str, str], dict[int, int]]:
    maps: dict[tuple[str, str, str], dict[int, int]] = {}
    for row in chain_mapping[chain_mapping["mapped"].eq(True)].itertuples(index=False):
        chain_rows = chain_cache.get((str(row.complex_id), str(row.chain)), [])
        start = int(row.uniprot_start)
        maps[(str(row.complex_id), str(row.chain), str(row.uniprot))] = {
            start + i: int(entry["chain_site"]) for i, entry in enumerate(chain_rows)
        }
    return maps


def nearest_distance(value: int, sites: set[int]) -> int | None:
    if not sites:
        return None
    return min(abs(value - site) for site in sites)


def build_event_mapping(
    chain_mapping: pd.DataFrame,
    pos_maps: dict[tuple[str, str, str], dict[int, int]],
    interface_sites: dict[tuple[str, str], set[int]],
    contact_sites: dict[tuple[str, str], set[int]],
) -> pd.DataFrame:
    event = pd.read_csv(TABLES / "event_table_v2.tsv", sep="\t")
    mapped = chain_mapping[chain_mapping["mapped"].eq(True) & chain_mapping["mapping_unique"].eq(True)].copy()
    pair_index: dict[tuple[str, str], list[tuple[str, pd.DataFrame, pd.DataFrame]]] = defaultdict(list)
    for complex_id, group in mapped.groupby("complex_id"):
        for modified in sorted(set(group["uniprot"].astype(str))):
            mod_rows = group[group["uniprot"].eq(modified)]
            for partner in sorted(set(group["uniprot"].astype(str))):
                partner_rows = group[group["uniprot"].eq(partner)]
                if not mod_rows.empty and not partner_rows.empty:
                    pair_index[(modified, partner)].append((str(complex_id), mod_rows, partner_rows))
    records = []
    for row in event.itertuples(index=False):
        modified = str(row.modified_uniprot)
        partner = str(row.partner_uniprot)
        position = int(row.position)
        for complex_id, mod_rows, partner_rows in pair_index.get((modified, partner), []):
            for mod in mod_rows.itertuples(index=False):
                pos_map = pos_maps.get((str(mod.complex_id), str(mod.chain), str(mod.uniprot)), {})
                chain_site = pos_map.get(position)
                if chain_site is None:
                    continue
                iface = interface_sites.get((str(mod.complex_id), str(mod.chain)), set())
                contacts = contact_sites.get((str(mod.complex_id), str(mod.chain)), set())
                records.append(
                    {
                        "event_id": row.event_id,
                        "modified_uniprot": modified,
                        "partner_uniprot": partner,
                        "modified_gene": row.modified_gene,
                        "partner_gene": row.partner_gene,
                        "ptm_type": row.ptm_type,
                        "residue": row.residue,
                        "position": position,
                        "effect_label": row.effect_label,
                        "complex_id": complex_id,
                        "modified_chain": mod.chain,
                        "modified_chain_site": chain_site,
                        "partner_chains": ",".join(sorted(set(partner_rows["chain"].astype(str)))),
                        "site_at_interface": chain_site in iface,
                        "site_has_recorded_contact": chain_site in contacts,
                        "distance_to_nearest_interface_site": nearest_distance(chain_site, iface),
                        "distance_to_nearest_contact_site": nearest_distance(chain_site, contacts),
                    }
                )
    return pd.DataFrame(records)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    seqs = load_fasta(FASTA)
    kmer_index = build_kmer_index(seqs)
    chain_mapping, chain_cache = build_chain_mapping(seqs, kmer_index)
    interface_sites = load_interface_sites()
    contact_sites = load_contact_sites()
    pos_maps = build_position_maps(chain_mapping, chain_cache)
    event_mapping = build_event_mapping(chain_mapping, pos_maps, interface_sites, contact_sites)

    if len(event_mapping):
        summary = pd.DataFrame(
            [
                {
                    "metric": "structure_supported_event_rows",
                    "value": event_mapping["event_id"].nunique(),
                    "interpretation": "Current event rows with exact chain-to-UniProt structure support.",
                },
                {
                    "metric": "mapped_complexes_used",
                    "value": event_mapping["complex_id"].nunique(),
                    "interpretation": "PTMint structure complexes tied to at least one event by exact sequence mapping.",
                },
                {
                    "metric": "site_at_interface_events",
                    "value": event_mapping[event_mapping["site_at_interface"]]["event_id"].nunique(),
                    "interpretation": "Events whose modified site is explicitly listed in the interface residue table.",
                },
                {
                    "metric": "site_contact_events",
                    "value": event_mapping[event_mapping["site_has_recorded_contact"]]["event_id"].nunique(),
                    "interpretation": "Events whose modified site has a recorded cross-chain contact in the archive.",
                },
                {
                    "metric": "median_distance_to_nearest_interface_site",
                    "value": float(event_mapping["distance_to_nearest_interface_site"].dropna().median()),
                    "interpretation": "Median sequence-position distance from PTM site to nearest listed interface residue among mapped events.",
                },
            ]
        )
    else:
        summary = pd.DataFrame(
            [
                {
                    "metric": "structure_supported_event_rows",
                    "value": 0,
                    "interpretation": "No event rows could be mapped by exact chain-to-UniProt sequence containment.",
                }
            ]
        )

    chain_mapping.to_csv(TABLES / "structure_chain_uniprot_mapping_v2.tsv", sep="\t", index=False)
    event_mapping.to_csv(TABLES / "structure_event_interface_mapping_v2.tsv", sep="\t", index=False)
    summary.to_csv(TABLES / "structure_event_interface_summary_v2.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
