# PTM-PPI Shield v2 Public Source Feasibility

Generated: 2026-06-20T18:26:08+00:00

## Existing PTMint Context

- Normalized table: `C:\Users\Aadi Nair\switchppi_sprint\data\processed\ptmint_normalized.csv`
- Rows: 4653
- Unique UniProt accessions across modified and partner proteins: 1870
- Columns: `modified_uniprot, modified_gene, partner_uniprot, partner_gene, organism, ptm_type, residue, position, effect_label, pmid, detection_method, disease, source`

## Source Checks

| Source | Endpoint | Probe | Rows / Size | Columns Seen | Fit | Recommendation |
| --- | --- | --- | ---: | --- | --- | --- |
| PhosphoSIGNOR all phosphorylation/dephosphorylation data | [endpoint](https://signor.uniroma2.it/PhosphoSIGNOR/apis/v1/index.php?role=all&format=tsv&header=yes) | ok | 26830 | entityA, entityA_name, mechanism, effect, entityB, entityB_name, pmid, signor_id, ... (9 cols) | best primary v2 candidate for signed phosphosite causal evidence | Add first as a v2 source: parse *_phSer/*_phThr/*_phTyr nodes into modified_uniprot, residue, position, effect_label, pmid, and source. |
| PhosphoSIGNOR kinase subset | [endpoint](https://signor.uniroma2.it/PhosphoSIGNOR/apis/v1/index.php?role=kinaseALL&format=tsv&header=yes) | ok | 24692 | entityA, entityA_name, mechanism, effect, entityB, entityB_name, pmid, signor_id, ... (9 cols) | kinase-to-site context for v2 features and evidence provenance | Use after all-data import if kinase-specific features are useful. |
| OmniPath signed causal interactions | [endpoint](https://omnipathdb.org/interactions?format=tsv&datasets=omnipath&fields=sources,references,curation_effort) | ok | 85217 | source, target, is_directed, is_stimulation, is_inhibition, consensus_direction, consensus_stimulation, consensus_inhibition, ... (11 cols) | signed directed PPI/regulatory prior for existing PTMint pairs | Add as v2 auxiliary features: signed edge prior, source count, reference count, and train-only degree controls. |
| OmniPath enzyme-substrate PTM sites | [endpoint](https://omnipathdb.org/enzsub?format=tsv&fields=sources,references) | ok | 41506 | enzyme, substrate, residue_type, residue_offset, modification, sources, references | enzyme-substrate PTM site prior | Add as v2 site context: kinase/phosphatase evidence for modified_uniprot, residue, position, and PTM type. |
| ELM motif classes | [endpoint](http://elm.eu.org/elms/elms_index.tsv) | ok | 353 | Accession, ELMIdentifier, FunctionalSiteName, Description, Regex, Probability, #Instances, #Instances_in_PDB | motif/window annotations for PTM site features | Add as v2 feature metadata: MOD/LIG/DOC class regex matches around the PTMint site window. |
| ELM experimentally curated motif instances | [endpoint](http://elm.eu.org/instances.tsv) | ok | 100 | Accession, ELMType, ELMIdentifier, ProteinName, Primary_Acc, Accessions, Start, End, ... (13 cols) | curated motif instance overlap checks | Use only as a secondary v2 annotation unless the full all-instance export is confirmed stable; the no-query endpoint returns a small default page. |
| ELM interaction-domain mappings | [endpoint](http://elm.eu.org/interactiondomains.tsv?q=*) | ok | 409 | ELM identifier, Interaction Domain Id, Interaction Domain Description, Interaction Domain Name | motif-to-domain partner context | Add with ELM classes if motif-domain features are desired. |
| BioGRID PTM/PTMREL latest release | [endpoint](https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-PTMS-LATEST.ptm.zip) | ok | HEAD ok; size not advertised |  | public PTM site and PTM relationship evidence | Feasible but heavier: add after SIGNOR/OmniPath if v2 needs broad PTMREL relationship annotations. |
| BioGRID all interactions Tab 3 latest release | [endpoint](https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-ALL-LATEST.tab3.zip) | ok | HEAD ok; size not advertised |  | general PPI background, not PTM-site labels | Do not prioritize for today's source expansion unless broad PPI background is required. |

## Feasible Additions Today

1. PhosphoSIGNOR all-data TSV is the strongest new public source. It provides SIGNOR-scored, PMID-backed signed phosphorylation/dephosphorylation causal relationships and encodes modified residue/position in node IDs such as `Q9UKV8_phSer387`.
2. OmniPath interactions can be added as signed causal network priors for PTMint pairs. Keep it auxiliary because it is not site-specific.
3. OmniPath `enzsub` can add kinase/substrate/PTM-site context for `modified_uniprot + residue + position` rows. Keep it auxiliary because it lacks partner-effect labels.
4. ELM classes and interaction-domain mappings are easy feature additions for motif/window annotation. ELM all-instance export needs a more careful download path because the `q=*` probe timed out manually.
5. BioGRID PTMS is public and feasible, but it is a second-pass task because the ZIP is larger and PTMTAB/PTMREL need identifier normalization before they fit the PTMint schema.

## Not Good Primary Label Additions

- BioGRID all Tab 3 is a useful broad PPI background, but it is large and mostly not signed PTM-site effect evidence.
- OmniPath signed interactions should not be converted directly into PTM-site labels without site evidence.
- ELM motif classes are feature annotations, not positive/negative effect labels.

## Suggested v2 Integration Order

1. Build `ingest_phosphosignor_v2.py` to normalize PhosphoSIGNOR into a new `data/processed/v2_phosphosignor_normalized.csv` table.
2. Build `annotate_omnipath_v2.py` to generate pair-level signed-prior and site-level enzyme-substrate feature tables keyed to PTMint rows.
3. Build `annotate_elm_v2.py` for motif class regex hits on PTM site windows.
4. Add BioGRID PTMS only after the first three sources are stable.

## Caveats

- PhosphoSIGNOR all phosphorylation/dephosphorylation data: This is signed phosphorylation causal biology, not always a direct PTM-regulated PPI effect like PTMint.
- PhosphoSIGNOR kinase subset: Same schema family as PhosphoSIGNOR all; mostly auxiliary to the all-data endpoint.
- OmniPath signed causal interactions: No PTM site columns; do not use as PTM-site labels without another source.
- OmniPath enzyme-substrate PTM sites: Does not encode PTM-regulated partner PPI effect labels.
- ELM motif classes: Class definitions are not evidence rows and are not signed effects.
- ELM experimentally curated motif instances: The attempted all-query export was slow in manual probing; default endpoint is only a sample/page.
- ELM interaction-domain mappings: Does not include PTMint-style signed effect labels.
- BioGRID PTM/PTMREL latest release: Requires zip parsing and identifier normalization; PTMTAB uses BioGRID/Entrez/RefSeq/sequence fields rather than direct PTMint-normalized UniProt pairs.
- BioGRID all interactions Tab 3 latest release: Large download and mostly not signed PTM-site-specific effect evidence.
