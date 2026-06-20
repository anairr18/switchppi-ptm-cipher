# SwitchPPI PTM-CIPHER

Computational sprint repository for PTM-conditioned protein-protein interaction
rewiring, with strict leakage-aware splits and a real 3Di/Foldseek-token
PTM-CIPHER ablation workflow.

## Run The Strict 3Di Colab

Open the notebook directly:

https://colab.research.google.com/github/anairr18/switchppi-ptm-cipher/blob/main/colab/ptm_cipher_3di_ablation_colab.ipynb

The notebook defaults to GitHub-backed inputs:

- `colab/ptmppi_shield_colab_inputs.zip` from this repository
- `ptmint_protein_structure_information.zip` from the `colab-data-v1` GitHub Release

No manual Colab upload is required once the release asset exists.

## One-Cell Colab Setup Alternative

If opening from GitHub is inconvenient, start a blank GPU Colab and run:

```python
!git clone https://github.com/anairr18/switchppi-ptm-cipher.git
%cd /content/switchppi-ptm-cipher
```

Then open or copy cells from:

```text
colab/ptm_cipher_3di_ablation_colab.ipynb
```

## Strict No-Fallback Policy

The 3Di notebook excludes events without complete real 3Di coverage for both
cropped proteins. `NO_STRUCTURE_ID` is allowed only for padding or for the
explicit `no_3di` ablation.

Primary ablations:

- `ptm_cipher_full_3di`
- `no_3di`
- `no_ptm_state`
- `no_contacts`
- `no_adversary`
- `no_delta_head`

The architecture claim is only supported if `ptm_cipher_full_3di` beats the
ablations on the strict S2b cold-interface test set, ideally across repeated
seeds.

## Key Outputs

The 3Di notebook writes `ptm_cipher_3di_colab_outputs.zip` with:

- `ptm_cipher_3di_strict_manifest.tsv`
- `ptm_cipher_3di_drop_report.tsv`
- `ptm_cipher_3di_chain_failures.tsv`
- `ptm_cipher_3di_ablation_metrics_colab.tsv`
- `ptm_cipher_3di_ablation_history_colab.tsv`
- one best-state `.pt` file per ablation

## Repository Data Policy

The extracted PTMint structure directory and compressed structure archive are
not committed to git. The compressed structure archive is attached as a GitHub
Release asset so Colab can fetch it by URL.
