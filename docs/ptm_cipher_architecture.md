# PTM-CIPHER Architecture

PTM-CIPHER stands for Counterfactual Interface Perturbation Hierarchical Encoder for Rewiring.

## Novelty Claim

PTM-CIPHER predicts signed PTM-caused PPI rewiring by computing a representation-level counterfactual delta between unmodified and modified proteoform states, propagating the PTM perturbation through a residue-contact graph, conditioning partner cross-attention on mapped interface contacts, and optionally using gradient-reversal adversarial heads to audit metadata shortcuts.

This is intentionally different from an ESM pair MLP, DeepPhosPPI-style window classifier, or PTM-Mamba embedding classifier. The model must not be described as architecturally novel unless the following are present:

1. A proteoform-state tokenization pathway where residue identity, structure state, and PTM state are embedded before the encoder.
2. A weight-shared dual-state encoder for unmodified and modified proteoforms.
3. A residue/contact perturbation path from the PTM site to interface residues.
4. A delta head that classifies modified-minus-unmodified interaction state.
5. For the performance path, a concat counterfactual head that can combine `I_modified`, `I_unmodified`, `delta`, `abs(delta)`, and frozen ESM2 site/local/protein features.
6. For the robustness path, shortcut-adversarial heads for assay/source/PMID/topology-style leakage variables.

## Inputs

- Modified protein sequence and partner sequence.
- PTM type, residue, and 1-based UniProt position.
- Unmodified PTM-state token sequence.
- Modified PTM-state token sequence with the PTM state set only at the modified site.
- Optional 3Di/Foldseek structure state sequence; fallback is `no_structure`.
- Optional frozen ESM2 token embeddings pooled at the PTM site, local PTM window, modified protein crop, and partner crop for the `ptm_cipher_esm2_fusion` performance model.
- Sparse contact/interface residue pairs mapped to cropped sequence coordinates.
- Shortcut labels are allowed only as adversarial targets, not as predictive features.

## Model Modules

- `FactoredProteoformEmbedding`: residue + 3Di + PTM-state factors plus interaction terms.
- Shared Transformer encoder: processes unmodified modified-protein state, modified modified-protein state, and unmodified partner state.
- Interface-conditioned cross-attention: adds contact-mask-derived partner context to cross-attention outputs.
- Dense contact perturbation propagation: injects the PTM delta at the PTM site and propagates over the residue-contact adjacency.
- Delta classifier: pools interface residues and predicts signed effect from `I_modified - I_unmodified`.
- Concat/fusion classifier: predicts from `I_modified`, `I_unmodified`, `delta`, `abs(delta)`, and projected frozen ESM2 features when challenging the ESM2 random forest baseline.
- Evidence head: emits Dirichlet evidence parameters for uncertainty-aware calibration.
- Gradient reversal adversaries: predict shortcut variables through GRL so the shared representation learns invariance.

## Current Implementation Status

The local implementation now has two paths:

- `ptm_cipher_full_3di`: strict real-3Di/contact/PTM-state ablation model for mechanism tests.
- `ptm_cipher_esm2_fusion`: no-adversary, concat-head performance model that fuses frozen ESM2 features with PTM-CIPHER interface/counterfactual representations.

Current repeated-seed results indicate that the adversarial objective can reduce raw AUROC/AUPRC, so the adversary should be reported as a robustness/control variant rather than the headline predictive model unless later runs reverse that finding.

## Required Ablations

- PTM-state axis removed.
- Counterfactual delta head removed.
- Interface cross-attention removed.
- Contact perturbation propagation removed.
- Gradient-reversal adversaries removed.
- Frozen ESM2 fusion removed.
- ESM2 pair MLP baseline.
- ESM2 random forest same-split baseline.
- PTM-Mamba embedding baseline if checkpoint access is available.

## Safe Manuscript Framing

Use: "We propose and evaluate PTM-CIPHER, a counterfactual proteoform-edge architecture designed to test whether explicit PTM state, interface contacts, and shortcut-invariant training improve leakage-resistant PTM-PPI rewiring prediction."

Do not use: "PTM-CIPHER is the first PTM-aware protein language model" or "PTM-CIPHER validates PTM rewiring mechanisms experimentally."
