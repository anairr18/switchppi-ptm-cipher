from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from scipy import sparse
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent))
import run_ptmppi_shield_v2 as shield


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
VALIDATION = ROOT / "validation"
TABLES = ROOT / "results_v2" / "tables"
UNIPROT_CACHE = RAW / "uniprot_sequences.json"


def parse_fasta(text: str) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            parts = line[1:].split("|")
            current = parts[1].split("-")[0] if len(parts) >= 2 and parts[0] in {"sp", "tr"} else line[1:].split()[0].split("-")[0]
            seqs.setdefault(current, [])
        elif current:
            seqs[current].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def load_or_fetch_sequences(accessions: list[str]) -> dict[str, str]:
    cache = json.loads(UNIPROT_CACHE.read_text(encoding="utf-8")) if UNIPROT_CACHE.exists() else {}
    wanted = sorted({str(a).split("-")[0] for a in accessions if isinstance(a, str) and re.match(r"^[A-Z0-9]+(?:-[0-9]+)?$", str(a))})
    missing = [a for a in wanted if a not in cache]
    session = requests.Session()
    for i in range(0, len(missing), 100):
        chunk = missing[i : i + 100]
        query = "(" + " OR ".join(f"accession:{a}" for a in chunk) + ")"
        url = "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=" + quote(query)
        response = session.get(url, timeout=90)
        response.raise_for_status()
        cache.update(parse_fasta(response.text))
    UNIPROT_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    return cache


def prepare_external(event_train: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    accessions = list(event_train["modified_uniprot"]) + list(event_train["partner_uniprot"]) + list(external["modified_uniprot"]) + list(external["partner_uniprot"])
    seqs = load_or_fetch_sequences(accessions)
    ext = external.copy()
    ext["modified_sequence"] = ext["modified_uniprot"].map(seqs)
    ext["partner_sequence"] = ext["partner_uniprot"].map(seqs)
    ext = ext[ext["modified_sequence"].notna() & ext["partner_sequence"].notna()].copy()
    ext["site_window_31"] = [shield.window_around_site(s, int(p), 15) for s, p in zip(ext["modified_sequence"], ext["position"])]
    ext["motif_family"] = [shield.motif_family(row, row["site_window_31"]) for _, row in ext.iterrows()]
    ext["kinase_proxy"] = [shield.kinase_proxy(row, row["site_window_31"]) for _, row in ext.iterrows()]
    ext["label_binary"] = (ext["effect_label"] == "enhance").astype(int)
    # Fill columns expected by topology/features.
    ext["source"] = ext["external_source"]
    ext["publication_year"] = 2024
    ext["pair_key"] = ext["pair_key"].astype(str)
    ext["site_key"] = ext["site_key"].astype(str)
    ext["event_id"] = [f"rrustemi2024_{i:04d}" for i in range(len(ext))]
    return ext.reset_index(drop=True)


def best_threshold(y_valid: np.ndarray, prob_valid: np.ndarray) -> float:
    return shield.best_valid_threshold(y_valid, prob_valid)


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame]:
    event = pd.read_csv(TABLES / "event_table_v2.tsv", sep="\t")
    # Reattach sequences using v2 builder cache by rebuilding event table with full internals.
    full_event = shield.build_event_table()
    external = pd.read_csv(VALIDATION / "rrustemi2024_signed_external_validation.tsv", sep="\t")
    external = prepare_external(full_event, external)
    train = full_event.copy()
    train["split"] = np.where(train["publication_year"].astype(int) <= 2021, "train", "holdout")
    # Use source-publication split valid rows for thresholding, but train all dated benchmark rows <=2021.
    train_idx = np.where(train["split"].to_numpy() == "train")[0]
    valid_idx = np.where(train["S3_source_publication_split"].to_numpy() == "valid")[0] if "S3_source_publication_split" in train.columns else train_idx[: max(50, len(train_idx) // 10)]
    combined = pd.concat([train, external], ignore_index=True, sort=False)
    matrices = shield.build_static_feature_matrices(combined)
    y_train_all = combined["label_binary"].to_numpy(int)
    ext_idx = np.arange(len(train), len(combined))
    rows = []
    pred_tables = []
    specs = [
        ("class_prior", None),
        ("topology_only", "topology"),
        ("motif_only", "motif"),
        ("sequence_random_forest", "sequence_only"),
        ("counterfactual_mlp", "counterfactual"),
        ("counterfactual_logistic", "counterfactual"),
        ("ptm_delta_only", "ptm_delta_only"),
    ]
    topo_train = shield.train_topology_features(combined.iloc[train_idx], combined.iloc[train_idx])
    topo_valid = shield.train_topology_features(combined.iloc[train_idx], combined.iloc[valid_idx])
    topo_ext = shield.train_topology_features(combined.iloc[train_idx], combined.iloc[ext_idx])
    for model, family in specs:
        if model == "class_prior":
            prob_valid = np.full(len(valid_idx), y_train_all[train_idx].mean())
            prob_ext = np.full(len(ext_idx), y_train_all[train_idx].mean())
        elif family == "topology":
            prob_valid, prob_ext = shield.fit_predict(model, topo_train, y_train_all[train_idx], topo_valid, topo_ext, 991)
        else:
            base = matrices[family]
            if family == "counterfactual":
                x_train = sparse.hstack([base[train_idx], topo_train], format="csr")
                x_valid = sparse.hstack([base[valid_idx], topo_valid], format="csr")
                x_ext = sparse.hstack([base[ext_idx], topo_ext], format="csr")
            else:
                x_train, x_valid, x_ext = base[train_idx], base[valid_idx], base[ext_idx]
            prob_valid, prob_ext = shield.fit_predict(model, x_train, y_train_all[train_idx], x_valid, x_ext, 991)
        y_ext = y_train_all[ext_idx]
        threshold = best_threshold(y_train_all[valid_idx], prob_valid)
        pred = (prob_ext >= threshold).astype(int)
        rows.append(
            {
                "external_source": "Rrustemi2024_PRISMA",
                "model": model,
                "train_n": int(len(train_idx)),
                "external_n": int(len(ext_idx)),
                "external_pos_rate": float(y_ext.mean()),
                "threshold": float(threshold),
                "auprc": float(average_precision_score(y_ext, prob_ext)),
                "auroc": float(roc_auc_score(y_ext, prob_ext)) if len(np.unique(y_ext)) > 1 else float("nan"),
                "mcc": float(matthews_corrcoef(y_ext, pred)) if len(np.unique(pred)) > 1 else 0.0,
                "macro_f1": float(f1_score(y_ext, pred, average="macro", zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_ext, pred)),
            }
        )
        pred_df = external[["event_id", "modified_gene", "partner_gene", "ptm_type", "residue", "position", "effect_label", "doi"]].copy()
        pred_df["model"] = model
        pred_df["prob_enhance"] = prob_ext
        pred_tables.append(pred_df)
    return pd.DataFrame(rows), pd.concat(pred_tables, ignore_index=True)


def main() -> None:
    metrics, preds = evaluate()
    metrics.to_csv(TABLES / "external_validation_metrics_v2.tsv", sep="\t", index=False)
    preds.to_csv(TABLES / "external_validation_predictions_v2.tsv", sep="\t", index=False)
    print(metrics.sort_values("auprc", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
