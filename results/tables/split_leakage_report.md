# Split Leakage Report

Generated: 2026-06-20T17:58:02+00:00
Seed: 2026

## Input

Input table: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\ptmint_normalized.csv`
Rows: 4653
Columns: modified_uniprot, modified_gene, partner_uniprot, partner_gene, organism, ptm_type, residue, position, effect_label, pmid, detection_method, disease, source

## Column Resolution

| Entity | Column | Note |
| --- | --- | --- |
| protein_a | `modified_uniprot` | ok |
| protein_b | `partner_uniprot` | ok |
| modified_protein | `modified_uniprot` | ok |
| site | `position` | ok |
| pmid | `pmid` | ok |
| label | `effect_label` | ok |

## Split Audits

| Split | Status | Forbidden entity | Train | Validation | Test | Entity overlaps | Train/test overlaps | Audit prep |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| random | passed | none | 3257 | 465 | 931 | n/a | n/a | shuffled-label, degree-only |
| pair_disjoint | passed | unordered protein pair | 3257 | 465 | 931 | 0 | 0 | shuffled-label, degree-only |
| modified_protein_disjoint | passed | modified protein | 3257 | 465 | 931 | 0 | 0 | shuffled-label, degree-only |
| site_disjoint | passed | modified protein + site when available, otherwise site | 3257 | 465 | 931 | 0 | 0 | shuffled-label, degree-only |
| pmid_disjoint | passed | PMID | 3257 | 465 | 931 | 0 | 0 | shuffled-label, degree-only |

## Files

### random
- split: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\splits\random.csv`
- leakage_audit: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\splits\random_leakage_audit.json`
- shuffled_labels: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\splits\random_shuffled_labels.csv`
- degree_only_features: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\splits\random_degree_only_features.csv`

### pair_disjoint
- split: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\splits\pair_disjoint.csv`
- leakage_audit: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\splits\pair_disjoint_leakage_audit.json`
- shuffled_labels: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\splits\pair_disjoint_shuffled_labels.csv`
- degree_only_features: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\splits\pair_disjoint_degree_only_features.csv`

### modified_protein_disjoint
- split: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\splits\modified_protein_disjoint.csv`
- leakage_audit: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\splits\modified_protein_disjoint_leakage_audit.json`
- shuffled_labels: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\splits\modified_protein_disjoint_shuffled_labels.csv`
- degree_only_features: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\splits\modified_protein_disjoint_degree_only_features.csv`

### site_disjoint
- split: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\splits\site_disjoint.csv`
- leakage_audit: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\splits\site_disjoint_leakage_audit.json`
- shuffled_labels: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\splits\site_disjoint_shuffled_labels.csv`
- degree_only_features: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\splits\site_disjoint_degree_only_features.csv`

### pmid_disjoint
- split: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\splits\pmid_disjoint.csv`
- leakage_audit: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\splits\pmid_disjoint_leakage_audit.json`
- shuffled_labels: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\splits\pmid_disjoint_shuffled_labels.csv`
- degree_only_features: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\splits\pmid_disjoint_degree_only_features.csv`
