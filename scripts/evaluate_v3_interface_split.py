from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from run_ptmppi_shield_v2 import (
    RAW,
    SEED,
    build_static_feature_matrices,
    claim_gate_table,
    robust_discovery_scores,
    split_collapse_table,
    train_and_evaluate,
    window_around_site,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_V2 = ROOT / "results_v2"
TABLES_V2 = RESULTS_V2 / "tables"
RESULTS_V3 = ROOT / "results_v3"
TABLES_V3 = RESULTS_V3 / "tables"
FIGURES_V3 = RESULTS_V3 / "figures"


def load_sequences() -> dict[str, str]:
    return json.loads((RAW / "uniprot_sequences.json").read_text(encoding="utf-8"))


def add_sequences(df: pd.DataFrame) -> pd.DataFrame:
    sequences = load_sequences()
    df = df.copy()
    df["modified_sequence"] = df["modified_uniprot"].map(sequences)
    df["partner_sequence"] = df["partner_uniprot"].map(sequences)
    missing = df["modified_sequence"].isna() | df["partner_sequence"].isna()
    if missing.any():
        raise ValueError(f"Missing sequences for {int(missing.sum())} event rows")
    df["site_window_31"] = [
        window_around_site(seq, int(pos), 15)
        for seq, pos in zip(df["modified_sequence"], df["position"])
    ]
    df["label_binary"] = df["effect_label"].map({"inhibit": 0, "enhance": 1}).astype(int)
    return df


def write_v3_figures(metrics: pd.DataFrame, robust: pd.DataFrame) -> None:
    FIGURES_V3.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    s2b = metrics[metrics["split_col"].eq("S2b_cold_interface_split")].sort_values("auprc", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(s2b["model"], s2b["auprc"], color="#4C78A8")
    ax.set_xlabel("AUPRC")
    ax.set_title("S2b cold-interface model comparison")
    fig.tight_layout()
    fig.savefig(FIGURES_V3 / "figure_v3_s2b_model_comparison.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot = robust.sort_values("robust_discovery_score", ascending=True)
    ax.barh(plot["model"], plot["robust_discovery_score"], color="#9C755F")
    ax.set_xlabel("Robust Discovery Score")
    ax.set_title("V3 Robust Discovery Score including S2b")
    fig.tight_layout()
    fig.savefig(FIGURES_V3 / "figure_v3_robust_discovery_score.png", dpi=300)
    plt.close(fig)


def main() -> None:
    TABLES_V3.mkdir(parents=True, exist_ok=True)
    FIGURES_V3.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TABLES_V3 / "event_table_v3.tsv", sep="\t")
    df = add_sequences(df)
    old_split_cols = [
        col
        for col in df.columns
        if col.startswith("S") and col.endswith("_split") and col != "S2b_cold_interface_split"
    ]
    df_s2b = df.drop(columns=old_split_cols)
    matrices = build_static_feature_matrices(df)
    metrics_s2b, predictions = train_and_evaluate(df_s2b, matrices)
    metrics_v2 = pd.read_csv(TABLES_V2 / "shield_model_metrics_v2.tsv", sep="\t")
    metrics = pd.concat([metrics_v2, metrics_s2b], ignore_index=True)
    audit = pd.concat(
        [
            pd.read_csv(TABLES_V2 / "shield_split_audit_v2.tsv", sep="\t"),
            pd.read_csv(TABLES_V3 / "interface_split_audit_v3.tsv", sep="\t"),
        ],
        ignore_index=True,
    )
    robust = robust_discovery_scores(metrics, audit)
    collapse = split_collapse_table(metrics)
    gates = claim_gate_table(metrics)

    metrics.to_csv(TABLES_V3 / "shield_model_metrics_v3.tsv", sep="\t", index=False)
    robust.to_csv(TABLES_V3 / "robust_discovery_scores_v3.tsv", sep="\t", index=False)
    collapse.to_csv(TABLES_V3 / "split_collapse_diagnostics_v3.tsv", sep="\t", index=False)
    gates.to_csv(TABLES_V3 / "claim_gate_matrix_v3.tsv", sep="\t", index=False)
    if len(predictions):
        predictions.to_csv(TABLES_V3 / "full_shield_predictions_v3.tsv", sep="\t", index=False)
    write_v3_figures(metrics, robust)
    s2b = metrics[metrics["split_col"].eq("S2b_cold_interface_split")].sort_values("auprc", ascending=False)
    print(s2b[["split_col", "model", "test_n", "auprc", "auroc", "mcc", "balanced_accuracy", "ece"]].to_string(index=False))


if __name__ == "__main__":
    main()
