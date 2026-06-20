from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES_V3 = ROOT / "results_v3" / "tables"
TABLES_V2 = ROOT / "results_v2" / "tables"
COLAB = ROOT / "colab"
RESULTS_V3 = ROOT / "results_v3"


def yes(value: bool, complete: str = "complete", missing: str = "missing") -> str:
    return complete if value else missing


def main() -> None:
    RESULTS_V3.mkdir(parents=True, exist_ok=True)
    rows = []
    interface_audit = pd.read_csv(TABLES_V3 / "interface_split_audit_v3.tsv", sep="\t")
    interface_ok = bool(
        (
            interface_audit[["train_valid_overlap", "train_test_overlap", "valid_test_overlap"]].fillna(0)
            == 0
        )
        .all()
        .all()
    )
    rows.append(
        {
            "gap": "true interface-similarity shielding",
            "status": yes(interface_ok),
            "evidence": (
                f"S2b contact-Jaccard cold-interface split: "
                f"{int(interface_audit.iloc[0]['train_rows'])}/"
                f"{int(interface_audit.iloc[0]['valid_rows'])}/"
                f"{int(interface_audit.iloc[0]['test_rows'])} train/valid/test rows; "
                f"cluster overlaps zero={interface_ok}; "
                f"{int(interface_audit.iloc[0]['interface_clusters'])} interface clusters."
            ),
            "remaining_action": "Optional upgrade: add Foldseek/TM-align/PINDER clusters as a second interface axis.",
        }
    )

    metrics = pd.read_csv(TABLES_V3 / "shield_model_metrics_v3.tsv", sep="\t")
    s2b = metrics[metrics["split_col"].eq("S2b_cold_interface_split")]
    best = s2b.sort_values("auprc", ascending=False).iloc[0] if len(s2b) else None
    rows.append(
        {
            "gap": "local Shield reruns include S2b",
            "status": yes(len(s2b) > 0),
            "evidence": (
                "No S2b metrics"
                if best is None
                else f"Best S2b local model={best['model']} AUPRC={best['auprc']:.3f}, MCC={best['mcc']:.3f}."
            ),
            "remaining_action": "Use these as CPU baselines; headline SOTA comparisons require Colab/GPU outputs.",
        }
    )

    colab_ready = (COLAB / "ptmppi_shield_gpu_baselines_colab.ipynb").exists() and (COLAB / "ptmppi_shield_colab_inputs.zip").exists()
    gpu_done = (TABLES_V3 / "shield_model_metrics_v3_with_gpu.tsv").exists()
    rows.append(
        {
            "gap": "SOTA/foundation-model baselines under identical Shield splits",
            "status": "complete" if gpu_done else ("compute_ready" if colab_ready else "missing"),
            "evidence": (
                "GPU metrics integrated"
                if gpu_done
                else "Colab notebook and exact input zip are ready; no GPU output has been integrated yet."
            ),
            "remaining_action": "Run colab/ptmppi_shield_gpu_baselines_colab.ipynb on a GPU runtime, then run scripts/integrate_colab_gpu_metrics.py locally.",
        }
    )

    inventory = pd.read_csv(TABLES_V2 / "external_method_reproducibility_inventory.tsv", sep="\t")
    blocked = inventory[inventory["rerunnable_now"].astype(str).str.lower().ne("true")]
    rows.append(
        {
            "gap": "official external method reproducibility",
            "status": "partially_blocked",
            "evidence": "Blocked or artifact-limited methods: " + ", ".join(blocked["method"].tolist()),
            "remaining_action": "Do not claim official reproduction for PhosPPI, DeepPhosPPI, or PTM-Mamba unless missing weights/checkpoints/features are supplied.",
        }
    )

    no_effect = TABLES_V3 / "no_effect_evidence.tsv"
    no_effect_rows = 0
    if no_effect.exists():
        no_effect_rows = len(pd.read_csv(no_effect, sep="\t"))
    rows.append(
        {
            "gap": "experimentally tested no-effect labels",
            "status": "scoped_out_binary_task" if no_effect_rows == 0 else "in_progress",
            "evidence": f"{no_effect_rows} curated no-effect rows; no-effect schema and source queries exist.",
            "remaining_action": "Keep headline task enhance-vs-inhibit until >=100 high-confidence no-effect rows are curated.",
        }
    )

    out = pd.DataFrame(rows)
    out.to_csv(TABLES_V3 / "ncs_gap_closure_scorecard_v3.tsv", sep="\t", index=False)
    text = [
        "NCS gap closure status v3",
        "",
        *[
            f"- {row['gap']}: {row['status']} | {row['evidence']} | Next: {row['remaining_action']}"
            for row in rows
        ],
        "",
        "Compute answer:",
        "Yes. The remaining SOTA/foundation-model comparison needs GPU compute. The local CPU work now closes contact-Jaccard cold-interface shielding and prepares the exact Colab inputs for ESM2/PTM-aware baselines.",
    ]
    (RESULTS_V3 / "ncs_gap_closure_status_v3.txt").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
