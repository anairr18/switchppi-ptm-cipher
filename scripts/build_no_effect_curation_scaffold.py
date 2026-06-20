from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"
TABLES_V3 = ROOT / "results_v3" / "tables"


NO_EFFECT_COLUMNS = [
    "modified_uniprot",
    "modified_gene",
    "partner_uniprot",
    "partner_gene",
    "organism",
    "ptm_type",
    "residue",
    "position",
    "effect_label",
    "assay_type",
    "perturbation",
    "pmid",
    "publication_date",
    "source",
    "evidence_sentence",
    "curator",
    "curation_confidence",
    "notes",
]


def main() -> None:
    VALIDATION.mkdir(parents=True, exist_ok=True)
    TABLES_V3.mkdir(parents=True, exist_ok=True)
    schema = pd.DataFrame(
        [
            {"column": col, "required": True, "description": desc}
            for col, desc in [
                ("modified_uniprot", "UniProt accession for the modified/tested protein."),
                ("modified_gene", "Gene symbol for the modified/tested protein."),
                ("partner_uniprot", "UniProt accession for the interaction partner."),
                ("partner_gene", "Gene symbol for the interaction partner."),
                ("organism", "Organism name."),
                ("ptm_type", "PTM type, e.g. Phos, Ac, Me, Ub."),
                ("residue", "One-letter residue code."),
                ("position", "1-based UniProt canonical position."),
                ("effect_label", "Must be no_effect for this table."),
                ("assay_type", "Assay that explicitly tested the PTM-site/PPI effect."),
                ("perturbation", "PTM mimic/blocking mutation, kinase perturbation, peptide modification, or direct modified peptide/protein assay."),
                ("pmid", "PubMed ID for the no-effect evidence."),
                ("publication_date", "Publication date, ideally YYYY-MM-DD."),
                ("source", "Database or manual literature source."),
                ("evidence_sentence", "Short quoted/paraphrased evidence that the interaction was tested and no effect was observed."),
                ("curator", "Curator initials or agent ID."),
                ("curation_confidence", "high, medium, or low."),
                ("notes", "Isoform, assay caveat, or ambiguity notes."),
            ]
        ]
    )
    queries = pd.DataFrame(
        [
            {
                "source": "IntAct FeatureTab / IMEx",
                "query": '"phosphorylation" "no effect" "protein interaction" site mutant',
                "acceptance_rule": "Only accept rows where a modified site or mutation is explicitly tested for interaction change.",
            },
            {
                "source": "BioGRID PTMREL / PTMTAB",
                "query": '"phosphorylation" "does not affect binding" "co-immunoprecipitation"',
                "acceptance_rule": "Require direct PPI assay evidence; do not infer no-effect from missing annotations.",
            },
            {
                "source": "Peptide array / pull-down literature",
                "query": '"phosphopeptide" "no binding" "14-3-3" OR SH2 OR WW',
                "acceptance_rule": "Accept only modified-vs-unmodified or mutant-vs-wild-type comparisons with a named partner.",
            },
            {
                "source": "AP-MS / proximity labeling perturbation studies",
                "query": '"site mutant" "interactome" "no significant change" phosphorylation',
                "acceptance_rule": "Require statistical no-change call for the named PPI under the perturbation.",
            },
            {
                "source": "PhosphoSitePlus licensed exports",
                "query": "Regulatory site interaction effects where effect is explicitly absent or unchanged.",
                "acceptance_rule": "Use only if licensing permits; keep provenance and evidence text.",
            },
        ]
    )
    empty = pd.DataFrame(columns=NO_EFFECT_COLUMNS)
    schema.to_csv(VALIDATION / "no_effect_evidence_schema.tsv", sep="\t", index=False)
    queries.to_csv(VALIDATION / "no_effect_source_queries.tsv", sep="\t", index=False)
    empty.to_csv(TABLES_V3 / "no_effect_evidence.tsv", sep="\t", index=False)
    note = """No-effect evidence policy:
The headline task remains binary signed regulation: enhance vs inhibit.
Unknown, untested, missing, or unlabeled PPIs must not be used as no_effect.
The project should add no_effect only when a source explicitly tested a PTM-site/PPI perturbation and reported no binding or no significant interaction change.
Until >=100 high-confidence no_effect rows are curated, NCS claims should not describe a three-class switch predictor.
"""
    (VALIDATION / "no_effect_scope_note.txt").write_text(note, encoding="utf-8")
    print(f"Wrote {TABLES_V3 / 'no_effect_evidence.tsv'} and validation/no_effect_* scaffolds")


if __name__ == "__main__":
    main()
