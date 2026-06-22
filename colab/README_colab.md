# PTM-PPI Shield Colab Compute

Use this only for the GPU baselines that cannot be run credibly on the local CPU session.

## Files

- `ptmppi_shield_colab_inputs.zip`: upload this when the notebook prompts.
- `ptmppi_shield_gpu_baselines_colab.ipynb`: open directly in Google Colab.
- `ptmppi_shield_gpu_baselines_colab.py`: same notebook as percent cells for copy/paste or script review.
- `ptm_cipher_prototype_colab.ipynb`: trains the PTM-CIPHER-lite prototype architecture on the S2b cold-interface split.
- `ptm_cipher_prototype_colab.py`: same PTM-CIPHER notebook as percent cells.
- `ptm_cipher_3di_ablation_colab.ipynb`: strict real-3Di/Foldseek-token ablation notebook for the publishable architecture test.
- `ptm_cipher_3di_ablation_colab.py`: same strict real-3Di notebook as percent cells.

The strict 3Di notebook also requires `data/raw/ptmint_protein_structure_information.zip`.
It is intentionally not bundled into `ptmppi_shield_colab_inputs.zip` because the
structure archive is large. Upload it in Colab, or copy it to Google Drive and set
`USE_DRIVE_FOR_STRUCTURE_ZIP = True` in the notebook.

## Runtime

Recommended: A100/L40S/L4/T4 GPU runtime. A T4 should run ESM2 t12 embeddings, but A100/L40S is better.

## Outputs

The notebook downloads `ptmppi_shield_colab_outputs.zip` containing:

- `shield_gpu_baseline_metrics_colab.tsv`
- `shield_gpu_baseline_reproducibility_notes.tsv`
- cached ESM2 embedding `.npy` files

Copy `shield_gpu_baseline_metrics_colab.tsv` into `results_v3/tables/` and rerun local readiness scripts.

The PTM-CIPHER prototype notebook downloads `ptm_cipher_colab_outputs.zip` containing:

- `ptm_cipher_lite_metrics_colab.tsv`
- `ptm_cipher_lite_best_state.pt`

Treat this as an architecture feasibility run, not a final NCS result, because it uses the no-structure fallback instead of Foldseek/3Di tokens.

The strict 3Di notebook downloads `ptm_cipher_3di_colab_outputs.zip` containing:

- `ptm_cipher_3di_strict_manifest.tsv`
- `ptm_cipher_3di_drop_report.tsv`
- `ptm_cipher_3di_chain_failures.tsv`
- `ptm_cipher_3di_ablation_metrics_colab.tsv`
- `ptm_cipher_3di_ablation_history_colab.tsv`
- `ptm_cipher_3di_ablation_predictions_colab.tsv`
- `ptm_cipher_3di_repeated_seed_summary_colab.tsv`
- `ptm_cipher_3di_seed_delta_summary_colab.tsv`
- `ptm_cipher_3di_paired_bootstrap_deltas_colab.tsv`
- `ptm_cipher_3di_claim_gate_colab.tsv`
- `ptm_cipher_3di_ensemble_metrics_colab.tsv`
- `ptm_cipher_3di_ensemble_predictions_colab.tsv`
- `ptm_cipher_3di_model_leaderboard_colab.tsv`
- `same_split_competitor_metrics_colab.tsv`
- `same_split_competitor_predictions_colab.tsv`
- `same_split_competitor_audit_colab.tsv`
- `ptm_cipher_esm2_fusion_metrics_colab.tsv`
- `ptm_cipher_esm2_fusion_predictions_colab.tsv`
- `ptm_cipher_esm2_fusion_summary_colab.tsv`
- `ptm_cipher_fusion_challenge_leaderboard_colab.tsv`
- `ptm_cipher_esm2_fusion_vs_rf_bootstrap_colab.tsv`
- `literature_metric_comparison_colab.tsv`
- one best-state `.pt` file per ablation

Use the strict 3Di metrics, not the prototype metrics, for any claim that the
architecture benefits from real structure tokens. The notebook excludes rows
without complete real 3Di coverage and fails if train/valid/test lack both labels.
By default, it runs the full-capability PTM-CIPHER configuration across five
seeds and all ablations; switch `FULL_CAPABILITY_MODE = False` only for debugging.

The ESM2 fusion challenge section is the performance path after the same-split
competitor reruns. It trains `ptm_cipher_esm2_fusion`, a no-adversary
counterfactual-concat PTM-CIPHER model with frozen ESM2 site/local/protein
features, then compares it directly against `esm2_same_split_random_forest` by
seed and paired bootstrap.

## Claim Policy

If PTM-Mamba checkpoints are not supplied in Drive, report PTM-Mamba as blocked, not reproduced.
If DeepPhosPPI encoded feature caches are absent, report the ESM2 MLP as a DeepPhosPPI-style approximation, not an official reproduction.
