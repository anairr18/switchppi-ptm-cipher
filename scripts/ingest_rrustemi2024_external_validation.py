from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUPP = ROOT / "data" / "external" / "rrustemi2024_prisma" / "supplementary"
TABLES = ROOT / "results_v2" / "tables"
VALIDATION = ROOT / "validation"

SUPP1 = SUPP / "41467_2024_46794_MOESM4_ESM.xlsx"
SUPP2 = SUPP / "41467_2024_46794_MOESM5_ESM.xlsx"
BENCHMARK_KEYS = VALIDATION / "benchmark_validation_keys.tsv"

AA3 = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
}


def clean_accession(value: object) -> str:
    text = "" if value is None else str(value)
    first = re.split(r"[;,\s]+", text.strip())[0]
    return first.split("-")[0]


def parse_candidate(candidate: str) -> tuple[str, str, int, str] | tuple[None, None, None, None]:
    # Examples: ANK2_Thr3093Ile, APC_Ser2621Cys, EGFR_Tyr1092Phe
    match = re.match(r"^(.+)_([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$", str(candidate))
    if not match:
        return None, None, None, None
    gene, wt3, pos, var3 = match.groups()
    return gene, AA3.get(wt3, wt3), int(pos), AA3.get(var3, var3)


def infer_effect(row: pd.Series) -> tuple[str | None, str]:
    wt_phos = pd.to_numeric(row.get("Median.SILAC.ratio.wt_phos"), errors="coerce")
    phos_sig = str(row.get("LFQsignificantPhos", "")).strip() == "+"
    wt_sig = str(row.get("LFQsignificantWt", "")).strip() == "+"
    mut_sig = str(row.get("LFQsignificantMut", "")).strip() == "+"
    # In the paper, WT/PHOS < -1 means preferential interaction with phosphorylated peptide.
    if pd.notna(wt_phos) and wt_phos <= -1 and phos_sig:
        return "enhance", "phosphorylated peptide preferential binding; WT/PHOS log2 <= -1 and LFQsignificantPhos=+"
    if pd.notna(wt_phos) and wt_phos >= 1 and wt_sig:
        return "inhibit", "non-phosphorylated WT peptide preferential binding; WT/PHOS log2 >= 1 and LFQsignificantWt=+"
    if pd.notna(wt_phos) and wt_phos <= -1:
        return "enhance", "phosphorylated peptide higher than WT by SILAC; LFQ phos flag not set"
    if pd.notna(wt_phos) and wt_phos >= 1:
        return "inhibit", "WT peptide higher than phosphorylated by SILAC; LFQ WT flag not set"
    if phos_sig and not wt_sig and not mut_sig:
        return "enhance", "LFQsignificantPhos only"
    if wt_sig and not phos_sig:
        return "inhibit", "LFQsignificantWt without LFQsignificantPhos"
    return None, "ambiguous or mutation-dominant differential interaction"


def build_external_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_excel(SUPP1, sheet_name="MutationCandidates")
    candidate_map = {}
    for _, cand in candidates.iterrows():
        if pd.isna(cand.get("Gene Name")) or pd.isna(cand.get("WT_AA")) or pd.isna(cand.get("MUT_RSD#")):
            continue
        key = (str(cand["Gene Name"]), str(cand["WT_AA"]), int(cand["MUT_RSD#"]))
        candidate_map[key] = {
            "UniProt ID": cand.get("UniProt ID", ""),
            "Disease": cand.get("Disease", ""),
            "WT_sequence": cand.get("WT_sequence", ""),
        }
    interactions = pd.read_excel(SUPP2, sheet_name="Peptide-protein interaction")
    rows = []
    ambiguous = []
    for _, rec in interactions.iterrows():
        gene, residue, position, variant = parse_candidate(rec["Peptide Candidate"])
        if gene is None:
            ambiguous.append({"reason": "unparsed_candidate", **rec.to_dict()})
            continue
        effect, rationale = infer_effect(rec)
        if effect is None:
            ambiguous.append({"reason": rationale, **rec.to_dict()})
            continue
        candidate_meta = candidate_map.get((gene, residue, int(position)), {})
        modified_uniprot = candidate_meta.get("UniProt ID", "")
        partner_uniprot = clean_accession(rec["Majority.protein.IDs"])
        partner_gene = str(rec["Gene.name"]).split(";")[0]
        row_key = f"{modified_uniprot}|{partner_uniprot}|Phos|{residue}|{position}"
        site_key = f"{modified_uniprot}|Phos|{residue}|{position}"
        pair_key = "||".join(sorted([str(modified_uniprot), str(partner_uniprot)]))
        rows.append(
            {
                "external_source": "Rrustemi2024_PRISMA",
                "external_tier": "Tier1_signed_prospective",
                "modified_uniprot": modified_uniprot,
                "modified_gene": gene,
                "partner_uniprot": partner_uniprot,
                "partner_gene": partner_gene,
                "organism": "Human",
                "ptm_type": "Phos",
                "residue": residue,
                "position": position,
                "variant_residue": variant,
                "effect_label": effect,
                "evidence_rationale": rationale,
                "pmid": "",
                "doi": "10.1038/s41467-024-46794-8",
                "publication_date": "2024-04-11",
                "assay_family": "peptide_prisma_interactomics",
                "detection_method": "PRISMA peptide pulldown + LFQ/SILAC MS",
                "median_silac_ratio_wt_phos": rec.get("Median.SILAC.ratio.wt_phos"),
                "median_silac_ratio_phos_mut": rec.get("Median.SILAC.ratio.phos_mut"),
                "median_silac_ratio_wt_mut": rec.get("Median.SILAC.ratio.wt_mut"),
                "lfq_significant_wt": rec.get("LFQsignificantWt"),
                "lfq_significant_mut": rec.get("LFQsignificantMut"),
                "lfq_significant_phos": rec.get("LFQsignificantPhos"),
                "disease": candidate_meta.get("Disease", ""),
                "site_window_15mer": candidate_meta.get("WT_sequence", ""),
                "row_key": row_key,
                "site_key": site_key,
                "pair_key": pair_key,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(ambiguous)


def audit_overlap(external: pd.DataFrame) -> pd.DataFrame:
    benchmark = pd.read_csv(BENCHMARK_KEYS, sep="\t")
    bench_rows = set(benchmark["row_key"].astype(str))
    bench_sites = set(benchmark["site_key"].astype(str))
    bench_pairs = set(benchmark["pair_key"].astype(str))
    bench_pmids = set(benchmark["pmid"].astype(str))
    return pd.DataFrame(
        [
            {
                "external_source": "Rrustemi2024_PRISMA",
                "external_rows": len(external),
                "exact_row_overlap": int(external["row_key"].astype(str).isin(bench_rows).sum()),
                "site_overlap": int(external["site_key"].astype(str).isin(bench_sites).sum()),
                "pair_overlap": int(external["pair_key"].astype(str).isin(bench_pairs).sum()),
                "pmid_overlap": int(external["pmid"].astype(str).isin(bench_pmids).sum()),
                "post_cutoff_rows": int((external["publication_date"] >= "2022-01-01").sum()),
                "enhance_rows": int((external["effect_label"] == "enhance").sum()),
                "inhibit_rows": int((external["effect_label"] == "inhibit").sum()),
            }
        ]
    )


def main() -> None:
    VALIDATION.mkdir(parents=True, exist_ok=True)
    external, ambiguous = build_external_table()
    audit = audit_overlap(external)
    external.to_csv(VALIDATION / "rrustemi2024_signed_external_validation.tsv", sep="\t", index=False)
    ambiguous.to_csv(VALIDATION / "rrustemi2024_ambiguous_rows.tsv", sep="\t", index=False)
    audit.to_csv(VALIDATION / "external_validation_overlap_audit.tsv", sep="\t", index=False)
    audit.to_csv(TABLES / "external_validation_overlap_audit.tsv", sep="\t", index=False)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
