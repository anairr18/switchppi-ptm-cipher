# PTM-PPI Shield v2 Validation Plan

Generated: 2026-06-20

Scope: validation and external-evidence design only. This plan does not require edits to core model or training scripts.

## Current Data Contract

Primary benchmark table: `results/tables/benchmark_dataset.tsv`.

Usable columns already present:

- Evidence identity: `modified_uniprot`, `modified_gene`, `partner_uniprot`, `partner_gene`, `organism`, `ptm_type`, `residue`, `position`, `pmid`, `source`
- Signed label: `effect_label` with current binary values `enhance` and `inhibit`
- Audit keys already emitted: `pair_key`, `site_key`
- Split metadata: `random_split`, `pair_disjoint_split`, `modified_protein_disjoint_split`, `site_disjoint_split`, `pmid_disjoint_split`
- Supporting metadata: `source_effect_label`, `detection_method`, `disease`, `colocalized`

Local benchmark facts checked from the table:

- Rows: 3824
- Unique PMIDs: 2054
- Unique sites: 2463
- Unique unordered pairs: 2350
- PTM types: Phos 3298, Ac 264, Me 104, Sumo 80, Ub 73, Glyco 5
- Labels: enhance 2646, inhibit 1178

PubMed ESummary date check on all 2054 PMIDs:

- Earliest `sortpubdate`: 1991-10-01
- Latest `sortpubdate`: 2021-12-01
- PMIDs dated 2022-01-01 or later: 0

Therefore the prospective literature cutoff is:

```text
prospective_cutoff_date = 2022-01-01
```

## Validation Tiers

### Tier 1: Signed Prospective Experimental Validation

Goal: evaluate the frozen v2 model on post-cutoff external records with direct signed evidence that a specific PTM at a specific site enhances or inhibits a specific PPI.

Required external normalized columns:

```text
modified_uniprot, partner_uniprot, organism, ptm_type, residue, position,
effect_label, pmid, publication_date, detection_method, source
```

Primary inclusion rules:

1. `publication_date >= 2022-01-01`.
2. `pmid` is not in the 2054-PMID benchmark set.
3. `source != PTMint`.
4. External row has a resolvable exact site:
   `modified_uniprot + ptm_type + residue + position`.
5. External row has a resolvable partner:
   `partner_uniprot`.
6. Label is directly signed from assay or curator language:
   `enhance` means increased or induced binding, association, recruitment, or complex formation.
   `inhibit` means reduced, disrupted, blocked, or lost binding, association, recruitment, or complex formation.
7. No threshold, hyperparameter, model family, feature selection, or calibration step is tuned on this external set.

Deduplication:

- `row_key = modified_uniprot|partner_uniprot|ptm_type|residue|position`
- `site_key = modified_uniprot|ptm_type|residue|position`
- `pair_key = min(modified_uniprot, partner_uniprot)||max(modified_uniprot, partner_uniprot)`
- Collapse exact duplicate source records by `row_key + effect_label + pmid`.
- If the same `row_key` has both labels across sources, exclude it from the primary metric table and report it in a conflict audit.

Primary metrics:

- AUROC and AUPRC for `enhance` vs `inhibit`
- Macro F1 and balanced accuracy at the frozen validation threshold
- Brier score and calibration intercept/slope
- Baseline AUPRC equal to external-set `enhance` prevalence
- 95 percent bootstrap confidence intervals clustered by PMID
- Secondary 95 percent bootstrap confidence intervals clustered by `site_key`

Minimum reporting bar:

- If fewer than 50 signed rows or fewer than 10 PMIDs pass filters, do not present headline performance.
- Report the set as a case-study or pilot external validation instead.

### Tier 2: Independent Source-Disjoint Validation

Goal: test whether the model generalizes outside PTMint curation even when the source includes older papers.

Inclusion rules:

1. `source != PTMint`.
2. `pmid` is not in the benchmark PMID set.
3. `row_key` is not in the benchmark `row_key` set.
4. The source provides direct signed PPI effect language.

This tier is not prospective unless the date rule also passes. It can support "independent-curation validation" but not "prospective validation."

### Tier 3: External Support / Rank Enrichment

Goal: use high-quality external PTM or PPI resources that are not signed enough for binary labels.

Allowed sources:

- BioGRID PTMTAB/PTMREL relationships
- IntAct FeatureTab or MITAB feature records
- OmniPath/SIGNOR/PhosphoSIGNOR PTM or causal-signaling records when they lack direct PPI-effect wording
- PhosphoSitePlus regulatory records where interaction effect is missing or ambiguous

Do not convert missing labels to negatives. Instead:

- Score model predictions for candidate rows.
- Measure top-k enrichment of external support among high-confidence predictions.
- Compare against degree-matched and PTM-type-matched candidate rows.
- Report as "external evidence enrichment", not signed accuracy.

## Prospective Cohort Construction Logic

1. Freeze current benchmark row keys and PMID set from `results/tables/benchmark_dataset.tsv`.
2. Query PMID dates using NCBI ESummary and store a `pmid_date_manifest.tsv`.
3. For each external source, normalize source-specific fields to `validation/external_evidence_schema.tsv`.
4. Canonicalize PTM names:

```text
phosphorylation -> Phos
acetylation -> Ac
ubiquitination, ubiquitylation -> Ub
methylation -> Me
sumoylation, SUMOylation -> Sumo
glycosylation -> Glyco
```

5. Canonicalize labels:

```text
induces, increases, enhances, recruits, stabilizes association -> enhance
decreases, disrupts, inhibits, prevents, abolishes, destabilizes association -> inhibit
```

6. Exclude rows with:

- Missing or non-numeric `position`
- Missing `residue`
- Missing `pmid`
- `publication_date < 2022-01-01`
- `pmid` in benchmark PMIDs
- Exact `row_key` in benchmark rows
- Conflicting labels after deduplication

7. Score external rows with the frozen v2 model.
8. Evaluate metrics without refitting or recalibration.
9. Report per-source and pooled results. The pooled primary result should use PMID-clustered bootstraps so one paper cannot dominate uncertainty.

## Source-Specific Use

### Rrustemi et al. 2024 PRISMA phosphosite PPI data

Use as the strongest immediately actionable prospective source.

Mapping:

- Bait peptide protein -> `modified_uniprot`
- Phosphosite residue and position -> `residue`, `position`
- Enriched prey protein -> `partner_uniprot`
- PTM type -> `Phos`
- Published article PMID/DOI -> `pmid`, `publication_date`
- Phosphorylated peptide versus unphosphorylated or mutant peptide differential binding -> `effect_label`

Label rule:

- `enhance` if phosphorylated peptide enriches a prey over non-phosphorylated/mutant peptide above the source FDR/effect-size cutoff.
- `inhibit` if phosphorylated peptide depletes or loses a prey relative to non-phosphorylated/mutant peptide above the source FDR/effect-size cutoff.

### PhosphoSitePlus Regulatory Sites

Use as signed independent-curation evidence when the regulatory record explicitly says modification regulates interaction and has an effect.

Mapping:

- Protein/site -> `modified_uniprot`, `ptm_type`, `residue`, `position`
- "Modification regulates interactions with" molecule -> `partner_uniprot`
- PSP effect text -> `effect_label`
- Curated publication -> `pmid`, `publication_date`

Use only records with interaction-specific effect wording. General "function", "activity", or "localization" regulatory annotations are not enough.

### BioGRID PTMTAB/PTMREL

Use primarily for Tier 3 external support and source-disjoint audits.

Mapping:

- PTMTAB: PTM site protein, position, type, residue, PMID, organism
- PTMREL: relationship partner, relationship text, PMID

Only promote to Tier 1 or Tier 2 if the `Relationship`/`Identity` text explicitly states a signed effect on binding or association. Otherwise keep as support-only.

### IntAct FeatureTab / IMEx

Use for curated feature-level interaction evidence.

Mapping:

- Feature affected protein -> `modified_uniprot`
- Feature range/original residue -> `residue`, `position`
- Feature type -> `ptm_type`
- Interaction participants -> `partner_uniprot`
- PubMedID -> `pmid`, `publication_date`

Only use as signed validation if feature annotation or interaction effect explicitly says the PTM increases or decreases interaction.

### eFIP Online

Use as a candidate-discovery source for manual curation.

Mapping:

- Substrate -> `modified_uniprot`
- Site -> `residue`, `position`
- Interacting protein -> `partner_uniprot`
- Phosphorylation-PPI relation and evidence sentence -> candidate `effect_label`
- Document ID -> `pmid`

Because eFIP is text-mined, require manual sentence review before inclusion in Tier 1 or Tier 2.

### OmniPath / SIGNOR / PhosphoSIGNOR

Use for mechanism consistency and Tier 3 support.

Mapping:

- Enzyme-substrate records can confirm site-level PTM but generally do not prove PPI effect.
- Causal interaction records with `binding` mechanism and signed effect can become Tier 2 if they map to exact current columns and have a PMID.

Do not use general activation/inhibition of protein activity as a proxy for PPI enhancement/inhibition.

### Targeted post-cutoff proteomics case studies

Use as case-study validations when exact site and partners can be mapped:

- YAP1 phosphorylation-linked complex profiling, 2023.
- 14-3-3theta S232 phospho-mutant AP-MS, 2025 ProteomeXchange PXD071112.

For phosphomutant/AP-MS designs:

- Bait -> `modified_uniprot`
- Differential prey -> `partner_uniprot`
- Mutated phosphosite -> `ptm_type`, `residue`, `position`
- Phosphomimetic or phosphorylated condition higher than non-phosphorylatable condition -> `enhance`
- Phosphomimetic or phosphorylated condition lower than non-phosphorylatable condition -> `inhibit`
- Use the source paper/dataset statistical cutoff; if absent, pre-register log2 fold change and adjusted P value before scoring.

## Claim Language

Allowed after Tier 1 passes:

- "Prospective, PMID-disjoint external validation on post-2021 literature..."
- "External signed validation after freezing the PTMint-derived benchmark..."

Allowed after Tier 2 only:

- "Independent source-disjoint validation..."

Allowed after Tier 3 only:

- "High-confidence predictions are enriched for external PTM/PPI support..."

Not allowed:

- "Experimentally validated predictions" unless the exact prediction was made before the experimental evidence and the validation set was fully post-freeze.
- "Clinical utility."
- "True no-effect negatives" from unlabeled external pairs.
- "State of the art" unless competing methods are rerun under identical temporal/source-disjoint rules.

## Key URLs

- PTMint source article: https://academic.oup.com/bioinformatics/article/39/1/btac823/6957085
- NCBI ESummary endpoint: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi
- Rrustemi et al. 2024 Nature Communications: https://www.nature.com/articles/s41467-024-46794-8
- PRISMA phosphoarray Zenodo record: https://doi.org/10.5281/zenodo.10790953
- Rrustemi/Selbach PRISMA code/data Zenodo record: https://doi.org/10.5281/zenodo.10786078
- BioGRID latest release: https://downloads.thebiogrid.org/BioGRID/Latest-Release/
- BioGRID PTMTAB/PTMREL format: https://wiki.thebiogrid.org/doku.php/biogrid_ptmtab_ptmrel
- PhosphoSitePlus: https://www.phosphosite.org/
- IntAct FTP/download: https://www.ebi.ac.uk/intact/download/ftp
- PSI-MI FeatureTab specification: https://www.psidev.info/psi-mi-tab-featuretab-exchange-format-for-molecular-interaction-features
- eFIP Online: https://research.bioinformatics.udel.edu/eFIPonline/
- OmniPath: https://omnipathdb.org/
- OmniPath interactions docs: https://r.omnipathdb.org/reference/omnipath-interactions.html
- PhosphoSIGNOR: https://signor.uniroma2.it/PhosphoSIGNOR/about/
- YAP1 phosphorylation-linked complex profiling: https://link.springer.com/article/10.15252/msb.202211024
- ProteomeXchange PXD071112: https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD071112
