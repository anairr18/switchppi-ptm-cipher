from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v2"
TABLES = RESULTS / "tables"
VALIDATION = ROOT / "validation"


def yesno(value: bool) -> str:
    return "complete" if value else "missing"


def build_scorecard() -> pd.DataFrame:
    audit = pd.read_csv(TABLES / "shield_split_audit_v2.tsv", sep="\t")
    metrics = pd.read_csv(TABLES / "shield_model_metrics_v2.tsv", sep="\t")
    gates = pd.read_csv(TABLES / "claim_gate_matrix_v2.tsv", sep="\t")
    ablations = pd.read_csv(TABLES / "novelty_ablation_summary_v2.tsv", sep="\t")
    event = pd.read_csv(TABLES / "event_table_v2.tsv", sep="\t")
    validation_sources = VALIDATION / "external_validation_source_candidates.tsv"
    external_metrics_path = TABLES / "external_validation_metrics_v2.tsv"
    external_overlap_path = TABLES / "external_validation_overlap_audit.tsv"
    method_inventory_path = TABLES / "external_method_reproducibility_inventory.tsv"
    deepphos_audit_path = TABLES / "deepphosppi_public_dataset_audit.tsv"
    structure_summary_path = TABLES / "structure_event_interface_summary_v2.tsv"
    structure_enrichment_path = TABLES / "structure_interface_enrichment_summary_v2.tsv"

    nonrandom = audit[audit["split_col"] != "S0_random_split"]
    zero_overlap = (nonrandom[["train_valid_overlap", "train_test_overlap", "valid_test_overlap"]].fillna(0) == 0).all().all()
    shield_splits = metrics[metrics["split_col"].ne("S0_random_split")]["split_col"].nunique()
    current_baselines = set(metrics["model"])
    has_controls = {"source_assay_only", "topology_only", "motif_only", "counterfactual_shuffled_labels"} <= current_baselines
    counterfactual_gates = gates[(gates["passes_gate"] == True) & gates["model"].str.startswith("counterfactual")]
    no_ptmchem = ablations[ablations["model"] == "counterfactual_no_ptmchem_logistic"]
    cf = ablations[ablations["model"] == "counterfactual_mlp"]
    cf_delta = float(cf.iloc[0]["delta_vs_no_ptmchem_mean_auprc"]) if len(cf) else 0.0
    prospective_ready = validation_sources.exists()
    external_ready = False
    external_evidence = "no external metrics table"
    if external_metrics_path.exists() and external_overlap_path.exists():
        external_metrics = pd.read_csv(external_metrics_path, sep="\t")
        external_overlap = pd.read_csv(external_overlap_path, sep="\t")
        best_external = external_metrics.sort_values("auprc", ascending=False).head(1)
        if len(best_external):
            external_ready = bool(best_external.iloc[0]["external_n"] >= 50 and external_overlap.iloc[0]["exact_row_overlap"] == 0 and external_overlap.iloc[0]["pmid_overlap"] == 0)
            external_evidence = (
                f"{int(best_external.iloc[0]['external_n'])} post-cutoff rows evaluated; "
                f"best={best_external.iloc[0]['model']} AUPRC={best_external.iloc[0]['auprc']:.3f}; "
                f"exact row overlap={int(external_overlap.iloc[0]['exact_row_overlap'])}, PMID overlap={int(external_overlap.iloc[0]['pmid_overlap'])}"
            )
    all_years = event["publication_year"].notna().all()
    structure_ready = False
    structure_evidence = "no exact chain-to-UniProt structure mapping table"
    structure_enrichment_line = ""
    if structure_summary_path.exists():
        structure_summary = pd.read_csv(structure_summary_path, sep="\t").set_index("metric")["value"]
        mapped_events = int(float(structure_summary.get("structure_supported_event_rows", 0)))
        interface_events = int(float(structure_summary.get("site_at_interface_events", 0)))
        contact_events = int(float(structure_summary.get("site_contact_events", 0)))
        structure_ready = mapped_events >= 500 and interface_events > 0
        structure_evidence = (
            f"{mapped_events} exact chain-to-UniProt mapped events; "
            f"{interface_events} PTM sites listed at interfaces; {contact_events} with recorded contacts"
        )
    if structure_enrichment_path.exists():
        enrichment = pd.read_csv(structure_enrichment_path, sep="\t")
        all_row = enrichment[enrichment["scope"].eq("all_structure_supported_events")]
        if len(all_row):
            structure_enrichment_line = (
                f" residue-matched interface enrichment={all_row.iloc[0]['interface_enrichment_ratio']:.2f}; "
                f"actual/null distance ratio={all_row.iloc[0]['distance_ratio_actual_over_null']:.2f}"
            )
            structure_evidence += ";" + structure_enrichment_line
    sota_evidence = "PhosPPI, DeepPhosPPI, PTM-Mamba, Betts/Mechismo, PINDER/PPIRef interface baselines not yet rerun"
    if method_inventory_path.exists():
        inventory = pd.read_csv(method_inventory_path, sep="\t")
        retrieved = int(inventory["retrieved"].fillna(False).sum())
        runnable = int(inventory["rerunnable_now"].fillna(False).sum())
        blocked = inventory[(inventory["retrieved"].fillna(False)) & (~inventory["rerunnable_now"].fillna(False))]
        blocked_methods = ", ".join(blocked["method"].tolist()) if len(blocked) else "none"
        sota_evidence = (
            f"reproducibility inventory complete for {len(inventory)} methods; "
            f"{retrieved} retrieved locally; {runnable} appears runnable without new artifacts; "
            f"blocked after retrieval: {blocked_methods}. No fair Shield rerun metrics yet."
        )
    if deepphos_audit_path.exists():
        deep_audit = pd.read_csv(deepphos_audit_path, sep="\t").set_index("metric")["value"]
        plus1_overlap = int(float(deep_audit.get("plus1_key_overlap_with_current_event_table", 0)))
        label_agree = float(deep_audit.get("plus1_overlap_label_agreement_rate", 0))
        plus1_match = float(deep_audit.get("plus1_position_residue_match_rate", 0))
        sota_evidence += (
            f" DeepPhosPPI public DatasetB audit: {plus1_overlap} rows overlap this event table after +1 site correction, "
            f"residue match rate={plus1_match:.3f}, label agreement={label_agree:.3f}; use this as a data-lineage warning, not a reproduced baseline."
        )

    rows = [
        {
            "criterion": "event-level evidence schema with provenance",
            "status": yesno({"pmid", "assay_family", "publication_year", "motif_family", "kinase_proxy"} <= set(event.columns)),
            "evidence": f"{len(event)} events, {event['pmid'].nunique()} PMIDs, {event['assay_family'].nunique()} assay families",
            "desk_rejection_risk_if_missing": "high",
        },
        {
            "criterion": "multi-axis leakage-shielded splits",
            "status": yesno(shield_splits >= 9 and zero_overlap),
            "evidence": f"{shield_splits} non-random shield splits; forbidden overlaps zero={zero_overlap}",
            "desk_rejection_risk_if_missing": "critical",
        },
        {
            "criterion": "confounder baselines and negative controls",
            "status": yesno(has_controls),
            "evidence": ", ".join(sorted(current_baselines & {"source_assay_only", "topology_only", "motif_only", "counterfactual_shuffled_labels"})),
            "desk_rejection_risk_if_missing": "critical",
        },
        {
            "criterion": "counterfactual/proteoform ablations",
            "status": yesno(len(no_ptmchem) > 0 and cf_delta > 0),
            "evidence": f"counterfactual_mlp mean shield AUPRC delta vs no-PTM-chemistry={cf_delta:.3f}",
            "desk_rejection_risk_if_missing": "high",
        },
        {
            "criterion": "claim-gating against shortcut baselines",
            "status": yesno(len(counterfactual_gates) > 0),
            "evidence": f"{len(gates[gates['passes_gate']==True])} total passed gates; {len(counterfactual_gates)} counterfactual passed gates",
            "desk_rejection_risk_if_missing": "high",
        },
        {
            "criterion": "temporal/prospective validation scaffold",
            "status": yesno(all_years and prospective_ready),
            "evidence": "all benchmark PMIDs dated; validation source candidates define post-2021 cutoff",
            "desk_rejection_risk_if_missing": "critical",
        },
        {
            "criterion": "real independent post-cutoff validation data ingested",
            "status": yesno(external_ready),
            "evidence": external_evidence,
            "desk_rejection_risk_if_missing": "critical for NCS",
        },
        {
            "criterion": "structure-supported interface localization audit",
            "status": yesno(structure_ready),
            "evidence": structure_evidence,
            "desk_rejection_risk_if_missing": "high",
        },
        {
            "criterion": "current SOTA baselines rerun under Shield",
            "status": "missing",
            "evidence": sota_evidence,
            "desk_rejection_risk_if_missing": "critical for NCS",
        },
        {
            "criterion": "true interface-similarity shielding",
            "status": "missing",
            "evidence": f"{structure_evidence}; no Foldseek/TM-align/contact-Jaccard/PINDER cold-interface clusters yet",
            "desk_rejection_risk_if_missing": "critical for NCS",
        },
        {
            "criterion": "experimentally tested no-effect labels",
            "status": "missing",
            "evidence": "current task is signed enhance vs inhibit among known regulatory events only",
            "desk_rejection_risk_if_missing": "medium if claims stay scoped; high for three-class switching",
        },
    ]
    return pd.DataFrame(rows)


def build_work_queue() -> pd.DataFrame:
    rows = [
        {
            "priority": 1,
            "task": "Expand post-2021 signed external validation rows beyond PRISMA",
            "owner_agent_prompt": "The Rrustemi2024 PRISMA validation set is ingested. Add more post-2021 signed PTM-PPI effect evidence from IntAct FeatureTab, BioGRID PTMREL, PhosphoSitePlus licensed exports, and manual literature. Output exact site+partner rows with PMID/date/source and effect direction.",
            "acceptance_criteria": ">=250 total PMID-disjoint post-2021 rows across at least two independent sources; zero overlap with training PMIDs; exact UniProt/site mapping.",
        },
        {
            "priority": 1,
            "task": "Run SOTA baselines under identical Shield splits",
            "owner_agent_prompt": "Reproduce or approximate PhosPPI, DeepPhosPPI, PTM-Mamba embedding classifiers, ESM pair MLP, ELM motif rules, SIGNOR/OmniPath propagation, and Betts/Mechismo interface rules under the v2 split columns.",
            "acceptance_criteria": "Each baseline has metrics in shield_model_metrics_v3.tsv and is evaluated on S1-S9 with identical train/valid/test rows.",
        },
        {
            "priority": 1,
            "task": "Add true interface-similarity shielding",
            "owner_agent_prompt": "Use the exact chain-to-UniProt PTMint structure mapping already generated in structure_event_interface_mapping_v2.tsv. Cluster mapped interfaces by Foldseek/TM-align/contact Jaccard or PINDER/PPIRef-derived interface families, then add S2b_cold_interface_split plus audit.",
            "acceptance_criteria": "Interface clusters for all structure-supported events; cold-interface split; leakage audit with train-test interface similarity distribution; model metrics rerun on the cold-interface split.",
        },
        {
            "priority": 2,
            "task": "GPU PTM-token counterfactual model",
            "owner_agent_prompt": "Train a real PTM-token paired-state model using ESM2/PTM-Mamba embeddings, LoRA/adapters, PTM chemistry tokens, and contrastive modified-vs-unmodified objectives. Save calibrated probabilities and ablations.",
            "acceptance_criteria": "Counterfactual model beats sequence RF or provides a clear complementary gate win on S7/S9/external validation.",
        },
        {
            "priority": 2,
            "task": "No-effect label curation",
            "owner_agent_prompt": "Find experimentally tested no-effect PTM-site/PPI cases from peptide arrays, mutagenesis, AP-MS, and literature. Keep unknown separate from no-effect.",
            "acceptance_criteria": "A separate no_effect table with evidence text/PMID and no synthetic negatives.",
        },
        {
            "priority": 3,
            "task": "Mechanistic localization validation",
            "owner_agent_prompt": "Quantify enrichment of high-confidence predicted switch sites in interfaces, IDRs, SLiMs, reader-domain motifs, allosteric paths, and kinase modules.",
            "acceptance_criteria": "Permutation-tested enrichment tables and at least three mechanism classes with clear positive controls.",
        },
    ]
    return pd.DataFrame(rows)


def build_figure_plan() -> pd.DataFrame:
    rows = [
        {"figure": "Fig. 1", "panel": "A-D", "content": "PTM-PPI Shield event schema, evidence provenance, PTM/assay/motif composition", "source_file": "event_table_v2.tsv + figure_v2_event_provenance.png"},
        {"figure": "Fig. 2", "panel": "A-E", "content": "Leakage graph and S0-S9 split audits; zero forbidden overlap table", "source_file": "shield_split_audit_v2.tsv"},
        {"figure": "Fig. 3", "panel": "A-D", "content": "Random-to-shield performance collapse and Robust Discovery Score", "source_file": "split_collapse_diagnostics_v2.tsv + robust_discovery_scores_v2.tsv"},
        {"figure": "Fig. 4", "panel": "A-D", "content": "Counterfactual proteoform ablations and claim-gate matrix", "source_file": "novelty_ablation_summary_v2.tsv + claim_gate_matrix_v2.tsv"},
        {"figure": "Fig. 5", "panel": "A-C", "content": "Structure-supported interface localization and same-residue null enrichment", "source_file": "structure_event_interface_summary_v2.tsv + structure_interface_enrichment_summary_v2.tsv + figure_v2_structure_interface_enrichment.png"},
        {"figure": "Fig. 6", "panel": "A-C", "content": "Prospective post-2021 validation on Rrustemi 2024 PRISMA signed PTM-PPI rows", "source_file": "external_validation_metrics_v2.tsv + external_validation_overlap_audit.tsv"},
        {"figure": "Extended Data", "panel": "all", "content": "Failure taxonomy, calibration risk, small-slice warnings, source/assay shortcut checks, and DeepPhosPPI public dataset lineage audit", "source_file": "failure_taxonomy_v2.tsv + figure_v2_calibration_risk.png + deepphosppi_public_dataset_audit.tsv"},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    scorecard = build_scorecard()
    queue = build_work_queue()
    figures = build_figure_plan()
    scorecard.to_csv(TABLES / "ncs_readiness_scorecard.tsv", sep="\t", index=False)
    queue.to_csv(TABLES / "ncs_remaining_work_queue.tsv", sep="\t", index=False)
    figures.to_csv(TABLES / "ncs_figure_plan.tsv", sep="\t", index=False)
    if (TABLES / "external_validation_metrics_v2.tsv").exists():
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ext = pd.read_csv(TABLES / "external_validation_metrics_v2.tsv", sep="\t").sort_values("auprc", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(ext["model"], ext["auprc"], color="#4C78A8")
        ax.set_xlabel("AUPRC")
        ax.set_title("Post-2021 PRISMA external validation")
        fig.tight_layout()
        (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(RESULTS / "figures" / "figure_v2_external_validation.png", dpi=300)
        plt.close(fig)

    complete = (scorecard["status"] == "complete").sum()
    missing = (scorecard["status"] == "missing").sum()
    structure_line = ""
    if (TABLES / "structure_event_interface_summary_v2.tsv").exists():
        struct = pd.read_csv(TABLES / "structure_event_interface_summary_v2.tsv", sep="\t").set_index("metric")["value"]
        structure_line = (
            f"Structure mapping: {int(float(struct.get('structure_supported_event_rows', 0)))} events have exact chain-to-UniProt support; "
            f"{int(float(struct.get('site_at_interface_events', 0)))} PTM sites are listed at interfaces; "
            f"{int(float(struct.get('site_contact_events', 0)))} have recorded contacts."
        )
        if (TABLES / "structure_interface_enrichment_summary_v2.tsv").exists():
            enrich = pd.read_csv(TABLES / "structure_interface_enrichment_summary_v2.tsv", sep="\t")
            all_row = enrich[enrich["scope"].eq("all_structure_supported_events")]
            if len(all_row):
                structure_line += (
                    f" Same-residue null enrichment={all_row.iloc[0]['interface_enrichment_ratio']:.2f}; "
                    f"actual/null distance ratio={all_row.iloc[0]['distance_ratio_actual_over_null']:.2f}."
                )
    sota_line = ""
    inventory_path = TABLES / "external_method_reproducibility_inventory.tsv"
    if inventory_path.exists():
        inv = pd.read_csv(inventory_path, sep="\t")
        retrieved = int(inv["retrieved"].fillna(False).sum())
        runnable = int(inv["rerunnable_now"].fillna(False).sum())
        sota_line = (
            f"External-method inventory is now present for {len(inv)} methods: "
            f"{retrieved} retrieved locally and {runnable} marked potentially runnable, "
            "but no SOTA method has yet been rerun under the Shield splits."
        )
    deepphos_line = ""
    deepphos_path = TABLES / "deepphosppi_public_dataset_audit.tsv"
    if deepphos_path.exists():
        deep_audit = pd.read_csv(deepphos_path, sep="\t").set_index("metric")["value"]
        deepphos_line = (
            "DeepPhosPPI public DatasetB lineage audit is complete: "
            f"{int(float(deep_audit.get('plus1_key_overlap_with_current_event_table', 0)))} rows overlap the current event table after +1 site correction, "
            f"with label agreement={float(deep_audit.get('plus1_overlap_label_agreement_rate', 0)):.3f}."
        )
    external_line = ""
    external_best_line = ""
    if (TABLES / "external_validation_metrics_v2.tsv").exists():
        ext = pd.read_csv(TABLES / "external_validation_metrics_v2.tsv", sep="\t").sort_values("auprc", ascending=False)
        best_auc = ext.iloc[0]
        best_mcc = ext.sort_values("mcc", ascending=False).iloc[0]
        external_best_line = (
            f"External validation: best ranking model {best_auc['model']} on {int(best_auc['external_n'])} PRISMA rows "
            f"AUPRC={best_auc['auprc']:.3f}, AUROC={best_auc['auroc']:.3f}; "
            f"best thresholded model {best_mcc['model']} MCC={best_mcc['mcc']:.3f}, "
            f"balanced_accuracy={best_mcc['balanced_accuracy']:.3f}."
        )
        external_line = (
            f"External validation now present on {int(best_auc['external_n'])} PRISMA rows: "
            f"best ranking model {best_auc['model']} AUPRC={best_auc['auprc']:.3f}, AUROC={best_auc['auroc']:.3f}; "
            f"best thresholded model {best_mcc['model']} MCC={best_mcc['mcc']:.3f}, balanced_accuracy={best_mcc['balanced_accuracy']:.3f}."
        )
    robust = pd.read_csv(TABLES / "robust_discovery_scores_v2.tsv", sep="\t").sort_values("robust_discovery_score", ascending=False)
    top_robust = robust.iloc[0]
    full_metrics = pd.read_csv(TABLES / "shield_model_metrics_v2.tsv", sep="\t")
    full_noncontrol = full_metrics[
        full_metrics["split_col"].eq("S9_full_shield_split")
        & ~full_metrics["model"].str.contains("only|shuffled|majority", regex=True)
    ].sort_values("auprc", ascending=False)
    full_line = ""
    if len(full_noncontrol):
        best_full = full_noncontrol.iloc[0]
        full_line = f"Full-shield best non-control model: {best_full['model']} AUPRC={best_full['auprc']:.3f}, MCC={best_full['mcc']:.3f}."
    claims = [
        "PTM-PPI Shield v2 central claim:",
        "This package is now a leakage-audited, structure-anchored evaluation framework for signed PTM-regulated protein-interaction effects. It should be framed as a methodological evaluation and discovery-readiness platform, not as a finished foundation model.",
        "",
        "Dataset:",
        "3824 nonredundant PTM-PPI evidence events; 2054 PMIDs; 10 assay families; all events have PubMed year metadata.",
        "",
        "Best shielded result:",
        full_line,
        f"Best Robust Discovery Score across non-random shield splits: {top_robust['model']} score={top_robust['robust_discovery_score']:.3f}.",
        "Random-split results are implementation checks, not discovery claims.",
        "",
        "External and biological validity:",
        external_best_line,
        structure_line,
        deepphos_line,
        "",
        "What is now novel:",
        "1. Multi-axis leakage graph over evidence events, not only pair/site split columns.",
        "2. PTM-specific shield axes: motif-window, kinase proxy, assay family, publication/source, topology community, and full-shield leakage component.",
        "3. Counterfactual feature construction with unmodified state, modified state, and explicit PTM-chemistry delta features.",
        "4. Exact PTMint structure-chain-to-UniProt mapping plus residue-matched interface localization audit.",
        "5. Competitor data-lineage audit showing DeepPhosPPI DatasetB nearly recapitulates the current PTMint-derived event table after +1 coordinate correction.",
        "",
        "Still missing for Nature Computational Science:",
        "1. True interface-similarity shielding with Foldseek/TM-align/contact-Jaccard/PINDER clusters, not only mapped interface localization.",
        "2. Fair reruns of PhosPPI, DeepPhosPPI, PTM-Mamba embeddings, ELM rules, SIGNOR/OmniPath propagation, and structure-aware baselines under identical Shield splits.",
        "3. Expansion of prospective validation beyond the current Rrustemi2024 PRISMA set.",
        "4. Real ESM/PTM-Mamba/PTM-token model training on GPU.",
        "5. Experimentally tested no-effect labels; unknown or unlabeled PPIs must not be treated as no-effect negatives.",
        "",
        "Safe manuscript sentence:",
        "We introduce PTM-PPI Shield, a multi-axis leakage-audited evaluation framework for signed PTM-regulated protein-interaction effects, and show that claims about PTM-conditioned interactome rewiring change materially under publication, assay, topology, motif, homology, temporal, and structure-aware audits.",
    ]
    (RESULTS / "claims_for_ncs_upgrade.txt").write_text("\n".join(line for line in claims if line), encoding="utf-8")
    text = [
        "NCS desk-rejection audit",
        f"Completed criteria: {complete}",
        f"Missing criteria: {missing}",
        "",
        "Bottom line:",
        "The current package is substantially stronger than a benchmark-only PTMint classifier because it includes event-level provenance, multi-axis Shield splits, shortcut baselines, claim gates, temporal scaffolding, and counterfactual ablations.",
        external_line,
        structure_line,
        sota_line,
        deepphos_line,
        "It is still not safe for Nature Computational Science submission until SOTA baseline reruns and true interface-similarity shielding are implemented.",
        "",
        "Most defensible current target:",
        "Bioinformatics / Briefings in Bioinformatics / NAR resource-style after external validation is added.",
        "",
        "NCS-safe target:",
        "Only after the remaining priority-1 tasks in ncs_remaining_work_queue.tsv are complete.",
    ]
    (RESULTS / "ncs_desk_rejection_audit.txt").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
