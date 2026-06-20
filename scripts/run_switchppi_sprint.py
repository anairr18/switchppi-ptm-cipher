from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
MODELS = ROOT / "models"

PTMINT_URL = "https://ptmint.sjtu.edu.cn/data/PTM%20experimental%20evidence.csv"
PTMINT_RAW = RAW / "ptmint_experimental_evidence.csv"
UNIPROT_CACHE = RAW / "uniprot_sequences.json"
RUN_SUMMARY = TABLES / "run_summary.json"

AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA)
RNG_SEED = 1729


def ensure_dirs() -> None:
    for path in [RAW, PROCESSED, TABLES, FIGURES, MODELS]:
        path.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(f"[switchppi] {message}", flush=True)


def download_ptmint() -> None:
    if PTMINT_RAW.exists() and PTMINT_RAW.stat().st_size > 1000:
        log(f"PTMint raw CSV already exists: {PTMINT_RAW}")
        return
    log(f"Downloading PTMint experimental evidence from {PTMINT_URL}")
    response = requests.get(PTMINT_URL, timeout=90)
    response.raise_for_status()
    PTMINT_RAW.write_bytes(response.content)
    log(f"Wrote {PTMINT_RAW} ({PTMINT_RAW.stat().st_size:,} bytes)")


def parse_fasta(text: str) -> dict[str, str]:
    sequences: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            header = line[1:]
            parts = header.split("|")
            if len(parts) >= 2 and parts[0] in {"sp", "tr"}:
                current = parts[1].split("-")[0]
            else:
                current = header.split()[0].split("-")[0]
            sequences.setdefault(current, [])
        elif current:
            sequences[current].append(line.strip())
    return {key: "".join(value) for key, value in sequences.items()}


def fetch_uniprot_sequences(accessions: list[str], chunk_size: int = 80) -> dict[str, str]:
    cache: dict[str, str] = {}
    if UNIPROT_CACHE.exists():
        cache = json.loads(UNIPROT_CACHE.read_text(encoding="utf-8"))

    wanted = sorted({x.split("-")[0] for x in accessions if isinstance(x, str) and re.match(r"^[A-Z0-9]+(?:-[0-9]+)?$", x)})
    missing = [x for x in wanted if x not in cache]
    if not missing:
        log(f"Loaded {len(cache):,} UniProt sequences from cache")
        return cache

    log(f"Fetching {len(missing):,} missing UniProt sequences in chunks of {chunk_size}")
    session = requests.Session()
    for i in range(0, len(missing), chunk_size):
        chunk = missing[i : i + chunk_size]
        query = "(" + " OR ".join(f"accession:{acc}" for acc in chunk) + ")"
        url = "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=" + quote(query)
        for attempt in range(3):
            try:
                response = session.get(url, timeout=90)
                response.raise_for_status()
                parsed = parse_fasta(response.text)
                cache.update(parsed)
                break
            except Exception as exc:
                if attempt == 2:
                    log(f"WARNING: failed UniProt chunk {i // chunk_size + 1}: {exc}")
                else:
                    time.sleep(2 + attempt)
        log(f"  UniProt chunk {i // chunk_size + 1}/{math.ceil(len(missing) / chunk_size)}: cache={len(cache):,}")
    UNIPROT_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    return cache


def normalize_ptmint() -> tuple[pd.DataFrame, dict[str, int]]:
    download_ptmint()
    raw = pd.read_csv(PTMINT_RAW, encoding="utf-8-sig")
    raw.columns = [c.strip() for c in raw.columns]

    required = [
        "Organism",
        "Gene",
        "Uniprot",
        "PTM",
        "Site",
        "AA",
        "Sequence window(-5,+5)",
        "Int_uniprot",
        "Int_gene",
        "Effect",
        "Method",
        "Disease",
        "Co-localized",
        "PMID",
    ]
    missing_cols = [c for c in required if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"PTMint schema changed; missing columns: {missing_cols}")

    label_map = {
        "enhance": "enhance",
        "induce": "enhance",
        "inhibit": "inhibit",
    }
    df = pd.DataFrame(
        {
            "modified_uniprot": raw["Uniprot"].astype(str).str.strip().str.split("-").str[0],
            "modified_gene": raw["Gene"].astype(str).str.strip(),
            "partner_uniprot": raw["Int_uniprot"].astype(str).str.strip().str.split("-").str[0],
            "partner_gene": raw["Int_gene"].astype(str).str.strip(),
            "organism": raw["Organism"].astype(str).str.strip(),
            "ptm_type": raw["PTM"].astype(str).str.strip(),
            "residue": raw["AA"].astype(str).str.strip().str.upper(),
            "position": pd.to_numeric(raw["Site"], errors="coerce").astype("Int64"),
            "effect_label": raw["Effect"].astype(str).str.strip().str.lower().map(label_map),
            "source_effect_label": raw["Effect"].astype(str).str.strip(),
            "pmid": raw["PMID"].astype(str).str.strip(),
            "detection_method": raw["Method"].astype(str).str.strip().replace({"nan": ""}),
            "disease": raw["Disease"].astype(str).str.strip().replace({"nan": ""}),
            "colocalized": raw["Co-localized"].astype(str).str.strip().replace({"nan": ""}),
            "source": "PTMint",
            "source_window_5": raw["Sequence window(-5,+5)"].astype(str).str.strip(),
        }
    )

    before_label = len(df)
    df = df[df["effect_label"].isin(["enhance", "inhibit"])].copy()
    dropped_label = before_label - len(df)

    accessions = sorted(set(df["modified_uniprot"]) | set(df["partner_uniprot"]))
    sequences = fetch_uniprot_sequences(accessions)
    df["modified_sequence"] = df["modified_uniprot"].map(sequences)
    df["partner_sequence"] = df["partner_uniprot"].map(sequences)

    before_seq = len(df)
    df = df[df["modified_sequence"].notna() & df["partner_sequence"].notna()].copy()
    dropped_missing_sequence = before_seq - len(df)

    def residue_matches(row: pd.Series) -> bool:
        pos = row["position"]
        seq = row["modified_sequence"]
        residue = row["residue"]
        if pd.isna(pos) or not isinstance(seq, str) or residue not in AA_SET:
            return False
        idx = int(pos) - 1
        return 0 <= idx < len(seq) and seq[idx] == residue

    before_residue = len(df)
    df["residue_position_valid"] = df.apply(residue_matches, axis=1)
    invalid_rows = df[~df["residue_position_valid"]].copy()
    invalid_rows.to_csv(TABLES / "invalid_residue_rows.tsv", sep="\t", index=False)
    df = df[df["residue_position_valid"]].copy()
    dropped_invalid_residue = before_residue - len(df)

    before_dup = len(df)
    dedup_key = [
        "modified_uniprot",
        "partner_uniprot",
        "ptm_type",
        "residue",
        "position",
        "effect_label",
        "pmid",
    ]
    df = df.drop_duplicates(subset=dedup_key).copy()
    dropped_duplicates = before_dup - len(df)

    df["pair_key"] = df.apply(lambda r: "||".join(sorted([r["modified_uniprot"], r["partner_uniprot"]])), axis=1)
    df["site_key"] = (
        df["modified_uniprot"]
        + "|"
        + df["ptm_type"]
        + "|"
        + df["residue"]
        + "|"
        + df["position"].astype(str)
    )
    df["directional_key"] = (
        df["modified_uniprot"]
        + "->"
        + df["partner_uniprot"]
        + "|"
        + df["site_key"]
    )
    df["label_binary"] = (df["effect_label"] == "enhance").astype(int)

    stats = {
        "raw_rows": int(len(raw)),
        "dropped_unknown_label": int(dropped_label),
        "dropped_missing_sequence": int(dropped_missing_sequence),
        "dropped_invalid_residue": int(dropped_invalid_residue),
        "dropped_duplicates": int(dropped_duplicates),
        "benchmark_rows": int(len(df)),
        "unique_modified_proteins": int(df["modified_uniprot"].nunique()),
        "unique_partner_proteins": int(df["partner_uniprot"].nunique()),
        "unique_unordered_pairs": int(df["pair_key"].nunique()),
        "unique_sites": int(df["site_key"].nunique()),
        "unique_pmids": int(df["pmid"].nunique()),
    }

    return df.reset_index(drop=True), stats


def assign_random_split(n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    values = np.array(["train"] * n, dtype=object)
    order = rng.permutation(n)
    n_train = int(n * 0.70)
    n_valid = int(n * 0.15)
    values[order[n_train : n_train + n_valid]] = "valid"
    values[order[n_train + n_valid :]] = "test"
    return values.tolist()


def assign_group_split(df: pd.DataFrame, group_col: str, seed: int) -> list[str]:
    rng = random.Random(seed)
    groups = list(df.groupby(group_col).size().items())
    rng.shuffle(groups)
    groups.sort(key=lambda x: (x[1], rng.random()), reverse=True)
    totals = {"train": 0, "valid": 0, "test": 0}
    targets = {"train": len(df) * 0.70, "valid": len(df) * 0.15, "test": len(df) * 0.15}
    assignment: dict[str, str] = {}
    for group, size in groups:
        deficits = {k: targets[k] - totals[k] for k in targets}
        split = max(deficits, key=deficits.get)
        assignment[group] = split
        totals[split] += size
    return df[group_col].map(assignment).tolist()


def add_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["random_split"] = assign_random_split(len(df), RNG_SEED)
    df["pair_disjoint_split"] = assign_group_split(df, "pair_key", RNG_SEED + 1)
    df["modified_protein_disjoint_split"] = assign_group_split(df, "modified_uniprot", RNG_SEED + 2)
    df["site_disjoint_split"] = assign_group_split(df, "site_key", RNG_SEED + 3)
    df["pmid_disjoint_split"] = assign_group_split(df, "pmid", RNG_SEED + 4)

    audit_rows: list[dict[str, object]] = []
    group_by_split = {
        "pair_disjoint_split": "pair_key",
        "modified_protein_disjoint_split": "modified_uniprot",
        "site_disjoint_split": "site_key",
        "pmid_disjoint_split": "pmid",
    }
    for split_col, group_col in group_by_split.items():
        groups = {
            split: set(df.loc[df[split_col] == split, group_col].dropna().astype(str))
            for split in ["train", "valid", "test"]
        }
        audit_rows.append(
            {
                "split_col": split_col,
                "group_col": group_col,
                "train_rows": int((df[split_col] == "train").sum()),
                "valid_rows": int((df[split_col] == "valid").sum()),
                "test_rows": int((df[split_col] == "test").sum()),
                "train_valid_overlap": len(groups["train"] & groups["valid"]),
                "train_test_overlap": len(groups["train"] & groups["test"]),
                "valid_test_overlap": len(groups["valid"] & groups["test"]),
            }
        )
    audit = pd.DataFrame(audit_rows)
    return df, audit


def safe_log1p(x: float) -> float:
    return math.log1p(max(float(x), 0.0))


def aa_composition(seq: str) -> np.ndarray:
    seq = "".join([a for a in str(seq).upper() if a in AA_SET])
    if not seq:
        return np.zeros(len(AA), dtype=np.float32)
    counts = np.array([seq.count(a) for a in AA], dtype=np.float32)
    return counts / counts.sum()


def physicochemical_features(seq: str) -> np.ndarray:
    seq = "".join([a for a in str(seq).upper() if a in AA_SET])
    if not seq:
        return np.zeros(9, dtype=np.float32)
    length = len(seq)
    hydrophobic = set("AILMFWVY")
    charged = set("DEKRH")
    acidic = set("DE")
    basic = set("KRH")
    polar = set("STNQCY")
    pro_gly = set("PG")
    aromatic = set("FWY")
    cysteine = set("C")
    return np.array(
        [
            safe_log1p(length),
            sum(a in hydrophobic for a in seq) / length,
            sum(a in charged for a in seq) / length,
            sum(a in acidic for a in seq) / length,
            sum(a in basic for a in seq) / length,
            sum(a in polar for a in seq) / length,
            sum(a in pro_gly for a in seq) / length,
            sum(a in aromatic for a in seq) / length,
            sum(a in cysteine for a in seq) / length,
        ],
        dtype=np.float32,
    )


def window_around_site(seq: str, position: int, flank: int = 15) -> str:
    idx = int(position) - 1
    start = max(0, idx - flank)
    end = min(len(seq), idx + flank + 1)
    return seq[start:end]


def hashed_kmers(seq: str, dims: int, k: int = 3, salt: str = "") -> np.ndarray:
    seq = "".join([a for a in str(seq).upper() if a in AA_SET])
    values = np.zeros(dims, dtype=np.float32)
    if len(seq) < k:
        return values
    for i in range(len(seq) - k + 1):
        token = salt + seq[i : i + k]
        digest = hashlib.md5(token.encode("ascii")).hexdigest()
        idx = int(digest[:8], 16) % dims
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        values[idx] += sign
    norm = np.linalg.norm(values)
    if norm > 0:
        values /= norm
    return values


def categorical_matrix(df: pd.DataFrame, columns: list[str]) -> tuple[sparse.csr_matrix, list[str]]:
    frames = []
    names: list[str] = []
    for col in columns:
        dummies = pd.get_dummies(df[col].fillna("missing").astype(str), prefix=col, dtype=np.float32)
        frames.append(dummies)
        names.extend(dummies.columns.tolist())
    if not frames:
        return sparse.csr_matrix((len(df), 0)), []
    mat = sparse.csr_matrix(pd.concat(frames, axis=1).to_numpy(dtype=np.float32))
    return mat, names


@dataclass
class FeaturePack:
    matrices: dict[str, sparse.csr_matrix]
    names: dict[str, list[str]]


def build_feature_pack(df: pd.DataFrame) -> FeaturePack:
    log("Building deterministic sequence and mechanism features")
    windows = [
        window_around_site(row.modified_sequence, int(row.position), flank=15)
        for row in df.itertuples(index=False)
    ]
    df = df.copy()
    df["site_window_31"] = windows
    df["position_norm"] = df["position"].astype(float) / df["modified_sequence"].str.len().astype(float)
    df["self_interaction"] = (df["modified_uniprot"] == df["partner_uniprot"]).astype(float)
    df["has_disease"] = df["disease"].fillna("").astype(str).str.len().gt(0).astype(float)
    df["has_colocalization"] = df["colocalized"].fillna("").astype(str).str.len().gt(0).astype(float)

    site_numeric = np.vstack(
        [
            np.concatenate(
                [
                    aa_composition(windows[i]),
                    physicochemical_features(windows[i]),
                    np.array([df.iloc[i]["position_norm"]], dtype=np.float32),
                ]
            )
            for i in range(len(df))
        ]
    )
    site_hash = np.vstack([hashed_kmers(w, dims=128, k=2, salt="site") for w in windows])
    mod_hash = np.vstack([hashed_kmers(seq, dims=192, k=3, salt="mod") for seq in df["modified_sequence"]])
    partner_hash = np.vstack([hashed_kmers(seq, dims=192, k=3, salt="partner") for seq in df["partner_sequence"]])
    mod_comp = np.vstack([np.concatenate([aa_composition(seq), physicochemical_features(seq)]) for seq in df["modified_sequence"]])
    partner_comp = np.vstack([np.concatenate([aa_composition(seq), physicochemical_features(seq)]) for seq in df["partner_sequence"]])
    cat, cat_names = categorical_matrix(df, ["ptm_type", "residue", "organism"])
    simple_numeric = sparse.csr_matrix(
        df[["position_norm", "self_interaction", "has_disease", "has_colocalization"]].to_numpy(dtype=np.float32)
    )

    site = sparse.csr_matrix(np.hstack([site_numeric, site_hash]))
    modified = sparse.csr_matrix(np.hstack([mod_comp, mod_hash]))
    partner = sparse.csr_matrix(np.hstack([partner_comp, partner_hash]))
    mechanism = sparse.hstack([cat, simple_numeric, site[:, :30]], format="csr")
    pair = sparse.hstack([modified, partner], format="csr")
    combined = sparse.hstack([mechanism, site, pair], format="csr")

    names = {
        "site": [f"site_{i}" for i in range(site.shape[1])],
        "modified": [f"modified_{i}" for i in range(modified.shape[1])],
        "partner": [f"partner_{i}" for i in range(partner.shape[1])],
        "pair": [f"pair_{i}" for i in range(pair.shape[1])],
        "mechanism": cat_names + ["position_norm", "self_interaction", "has_disease", "has_colocalization"] + [f"site_summary_{i}" for i in range(30)],
        "combined": [f"combined_{i}" for i in range(combined.shape[1])],
    }
    return FeaturePack(
        matrices={
            "site": site,
            "modified": modified,
            "partner": partner,
            "pair": pair,
            "mechanism": mechanism,
            "combined": combined,
        },
        names=names,
    )


def train_only_degree_features(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> sparse.csr_matrix:
    mod_degree: dict[str, int] = {}
    partner_degree: dict[str, int] = {}
    protein_degree: dict[str, int] = {}
    site_count: dict[str, int] = {}
    pair_seen: set[str] = set()
    for row in train_df.itertuples(index=False):
        mod_degree[row.modified_uniprot] = mod_degree.get(row.modified_uniprot, 0) + 1
        partner_degree[row.partner_uniprot] = partner_degree.get(row.partner_uniprot, 0) + 1
        protein_degree[row.modified_uniprot] = protein_degree.get(row.modified_uniprot, 0) + 1
        protein_degree[row.partner_uniprot] = protein_degree.get(row.partner_uniprot, 0) + 1
        site_count[row.site_key] = site_count.get(row.site_key, 0) + 1
        pair_seen.add(row.pair_key)
    values = []
    for row in eval_df.itertuples(index=False):
        values.append(
            [
                safe_log1p(mod_degree.get(row.modified_uniprot, 0)),
                safe_log1p(partner_degree.get(row.partner_uniprot, 0)),
                safe_log1p(protein_degree.get(row.modified_uniprot, 0)),
                safe_log1p(protein_degree.get(row.partner_uniprot, 0)),
                safe_log1p(site_count.get(row.site_key, 0)),
                float(row.pair_key in pair_seen),
            ]
        )
    return sparse.csr_matrix(np.asarray(values, dtype=np.float32))


def best_threshold(y_valid: np.ndarray, prob_valid: np.ndarray) -> float:
    if len(set(y_valid.tolist())) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_valid, prob_valid)
    best = 0.5
    best_score = -1.0
    for t in thresholds:
        score = f1_score(y_valid, prob_valid >= t, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best = float(t)
    return best


def bootstrap_ci(y_true: np.ndarray, prob: np.ndarray, metric_fn: Callable[[np.ndarray, np.ndarray], float], seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"))
    for _ in range(200):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            vals.append(metric_fn(y_true[idx], prob[idx]))
        except Exception:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def metric_row(
    split_col: str,
    model_name: str,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    prob_valid: np.ndarray,
    prob_test: np.ndarray,
    feature_family: str,
    seed: int,
) -> dict[str, object]:
    threshold = best_threshold(y_valid, prob_valid)
    pred = (prob_test >= threshold).astype(int)
    row: dict[str, object] = {
        "split_col": split_col,
        "model": model_name,
        "feature_family": feature_family,
        "train_n": int(len(y_train)),
        "valid_n": int(len(y_valid)),
        "test_n": int(len(y_test)),
        "train_pos_rate": float(np.mean(y_train)),
        "valid_pos_rate": float(np.mean(y_valid)),
        "test_pos_rate": float(np.mean(y_test)),
        "threshold_from_valid": float(threshold),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
    }
    if len(np.unique(y_test)) >= 2:
        row["auroc"] = float(roc_auc_score(y_test, prob_test))
        row["auprc"] = float(average_precision_score(y_test, prob_test))
        auroc_lo, auroc_hi = bootstrap_ci(y_test, prob_test, roc_auc_score, seed)
        auprc_lo, auprc_hi = bootstrap_ci(y_test, prob_test, average_precision_score, seed + 1)
        row["auroc_ci_low"] = auroc_lo
        row["auroc_ci_high"] = auroc_hi
        row["auprc_ci_low"] = auprc_lo
        row["auprc_ci_high"] = auprc_hi
    else:
        for key in ["auroc", "auprc", "auroc_ci_low", "auroc_ci_high", "auprc_ci_low", "auprc_ci_high"]:
            row[key] = float("nan")
    return row


def fit_predict_model(
    model_name: str,
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    x_valid: sparse.csr_matrix,
    x_test: sparse.csr_matrix,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "majority_baseline":
        p = float(np.mean(y_train))
        return np.full(x_valid.shape[0], p), np.full(x_test.shape[0], p)

    if model_name in {"logistic", "site_window_logistic", "mechanism_logistic", "degree_only"}:
        clf = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=seed),
        )
    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    elif model_name == "sequence_mlp":
        clf = make_pipeline(
            StandardScaler(with_mean=False),
            MLPClassifier(
                hidden_layer_sizes=(96,),
                activation="relu",
                alpha=1e-3,
                learning_rate_init=1e-3,
                early_stopping=True,
                max_iter=120,
                random_state=seed,
            ),
        )
    else:
        raise ValueError(model_name)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(x_train, y_train)
    return clf.predict_proba(x_valid)[:, 1], clf.predict_proba(x_test)[:, 1]


def train_models(df: pd.DataFrame, features: FeaturePack) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_cols = [
        "random_split",
        "pair_disjoint_split",
        "modified_protein_disjoint_split",
        "site_disjoint_split",
        "pmid_disjoint_split",
    ]
    model_specs = [
        ("majority_baseline", "none", None),
        ("degree_only", "degree", None),
        ("logistic", "combined", "combined"),
        ("random_forest", "combined", "combined"),
        ("site_window_logistic", "site", "site"),
        ("sequence_mlp", "pair", "pair"),
        ("mechanism_logistic", "mechanism", "mechanism"),
    ]
    rows: list[dict[str, object]] = []
    prediction_tables: list[pd.DataFrame] = []

    y = df["label_binary"].to_numpy(dtype=int)
    for split_idx, split_col in enumerate(split_cols):
        log(f"Training models for {split_col}")
        train_idx = np.where(df[split_col].to_numpy() == "train")[0]
        valid_idx = np.where(df[split_col].to_numpy() == "valid")[0]
        test_idx = np.where(df[split_col].to_numpy() == "test")[0]
        if len(set(y[train_idx].tolist())) < 2 or len(set(y[test_idx].tolist())) < 2:
            log(f"  Skipping {split_col}: train or test lacks both classes")
            continue

        degree_train = train_only_degree_features(df.iloc[train_idx], df.iloc[train_idx])
        degree_valid = train_only_degree_features(df.iloc[train_idx], df.iloc[valid_idx])
        degree_test = train_only_degree_features(df.iloc[train_idx], df.iloc[test_idx])

        for spec_idx, (model_name, family, feature_key) in enumerate(model_specs):
            if family == "none":
                x_train = sparse.csr_matrix((len(train_idx), 1))
                x_valid = sparse.csr_matrix((len(valid_idx), 1))
                x_test = sparse.csr_matrix((len(test_idx), 1))
            elif family == "degree":
                x_train, x_valid, x_test = degree_train, degree_valid, degree_test
            else:
                assert feature_key is not None
                base = features.matrices[feature_key]
                if feature_key == "combined":
                    x_train = sparse.hstack([base[train_idx], degree_train], format="csr")
                    x_valid = sparse.hstack([base[valid_idx], degree_valid], format="csr")
                    x_test = sparse.hstack([base[test_idx], degree_test], format="csr")
                else:
                    x_train = base[train_idx]
                    x_valid = base[valid_idx]
                    x_test = base[test_idx]

            prob_valid, prob_test = fit_predict_model(
                model_name=model_name,
                x_train=x_train,
                y_train=y[train_idx],
                x_valid=x_valid,
                x_test=x_test,
                seed=RNG_SEED + split_idx * 100 + spec_idx,
            )
            rows.append(
                metric_row(
                    split_col,
                    model_name,
                    y[train_idx],
                    y[valid_idx],
                    y[test_idx],
                    prob_valid,
                    prob_test,
                    family,
                    RNG_SEED + split_idx * 1000 + spec_idx,
                )
            )
            if split_col == "site_disjoint_split" and model_name in {"logistic", "random_forest"}:
                pred_df = df.iloc[test_idx][
                    [
                        "modified_gene",
                        "modified_uniprot",
                        "partner_gene",
                        "partner_uniprot",
                        "ptm_type",
                        "residue",
                        "position",
                        "effect_label",
                        "disease",
                        "pmid",
                        "detection_method",
                    ]
                ].copy()
                pred_df["model"] = model_name
                pred_df["prob_enhance"] = prob_test
                pred_df["confidence"] = np.maximum(prob_test, 1.0 - prob_test)
                pred_df["predicted_label"] = np.where(prob_test >= 0.5, "enhance", "inhibit")
                prediction_tables.append(pred_df)

        # Shuffled-label negative control for the main combined logistic model.
        rng = np.random.default_rng(RNG_SEED + split_idx)
        y_shuffled = y[train_idx].copy()
        rng.shuffle(y_shuffled)
        base = features.matrices["combined"]
        x_train = sparse.hstack([base[train_idx], degree_train], format="csr")
        x_valid = sparse.hstack([base[valid_idx], degree_valid], format="csr")
        x_test = sparse.hstack([base[test_idx], degree_test], format="csr")
        prob_valid, prob_test = fit_predict_model(
            "logistic",
            x_train,
            y_shuffled,
            x_valid,
            x_test,
            seed=RNG_SEED + split_idx * 101,
        )
        rows.append(
            metric_row(
                split_col,
                "combined_shuffled_labels",
                y_shuffled,
                y[valid_idx],
                y[test_idx],
                prob_valid,
                prob_test,
                "negative_control",
                RNG_SEED + split_idx * 2000,
            )
        )

    metrics = pd.DataFrame(rows)

    ablation_rows = []
    target_split = "site_disjoint_split"
    train_idx = np.where(df[target_split].to_numpy() == "train")[0]
    valid_idx = np.where(df[target_split].to_numpy() == "valid")[0]
    test_idx = np.where(df[target_split].to_numpy() == "test")[0]
    degree_train = train_only_degree_features(df.iloc[train_idx], df.iloc[train_idx])
    degree_valid = train_only_degree_features(df.iloc[train_idx], df.iloc[valid_idx])
    degree_test = train_only_degree_features(df.iloc[train_idx], df.iloc[test_idx])
    ablations = {
        "site_only": features.matrices["site"],
        "modified_only": features.matrices["modified"],
        "partner_only": features.matrices["partner"],
        "pair_sequence_only": features.matrices["pair"],
        "mechanism_only": features.matrices["mechanism"],
        "combined_plus_degree": sparse.hstack([features.matrices["combined"], train_only_degree_features(df, df)], format="csr"),
    }
    for i, (name, mat) in enumerate(ablations.items()):
        if name == "combined_plus_degree":
            x_train = sparse.hstack([features.matrices["combined"][train_idx], degree_train], format="csr")
            x_valid = sparse.hstack([features.matrices["combined"][valid_idx], degree_valid], format="csr")
            x_test = sparse.hstack([features.matrices["combined"][test_idx], degree_test], format="csr")
        else:
            x_train, x_valid, x_test = mat[train_idx], mat[valid_idx], mat[test_idx]
        prob_valid, prob_test = fit_predict_model("logistic", x_train, y[train_idx], x_valid, x_test, RNG_SEED + i)
        row = metric_row(
            target_split,
            name,
            y[train_idx],
            y[valid_idx],
            y[test_idx],
            prob_valid,
            prob_test,
            "ablation",
            RNG_SEED + i,
        )
        ablation_rows.append(row)
    ablations_df = pd.DataFrame(ablation_rows)
    predictions = pd.concat(prediction_tables, ignore_index=True) if prediction_tables else pd.DataFrame()
    return metrics, ablations_df, predictions


def write_tables(df: pd.DataFrame, audit: pd.DataFrame, metrics: pd.DataFrame, ablations: pd.DataFrame, predictions: pd.DataFrame, stats: dict[str, int]) -> None:
    public_cols = [
        "modified_uniprot",
        "modified_gene",
        "partner_uniprot",
        "partner_gene",
        "organism",
        "ptm_type",
        "residue",
        "position",
        "effect_label",
        "source_effect_label",
        "pmid",
        "detection_method",
        "disease",
        "colocalized",
        "source",
        "pair_key",
        "site_key",
        "random_split",
        "pair_disjoint_split",
        "modified_protein_disjoint_split",
        "site_disjoint_split",
        "pmid_disjoint_split",
    ]
    df[public_cols].to_csv(TABLES / "benchmark_dataset.tsv", sep="\t", index=False)
    audit.to_csv(TABLES / "split_leakage_report.tsv", sep="\t", index=False)
    metrics.to_csv(TABLES / "model_metrics.tsv", sep="\t", index=False)
    ablations.to_csv(TABLES / "ablation_metrics.tsv", sep="\t", index=False)
    if predictions.empty:
        pd.DataFrame().to_csv(TABLES / "top_predictions.tsv", sep="\t", index=False)
    else:
        predictions.sort_values("confidence", ascending=False).head(250).to_csv(TABLES / "top_predictions.tsv", sep="\t", index=False)
    RUN_SUMMARY.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")


def make_figures(df: pd.DataFrame, metrics: pd.DataFrame, ablations: pd.DataFrame, predictions: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    label_counts = df["effect_label"].value_counts().rename_axis("label").reset_index(name="count")
    sns.barplot(data=label_counts, x="label", y="count", ax=axes[0], palette="Set2", hue="label", legend=False)
    axes[0].set_title("PTMint labels after QC")
    axes[0].set_xlabel("")
    ptm_counts = df["ptm_type"].value_counts().head(8).rename_axis("ptm_type").reset_index(name="count")
    sns.barplot(data=ptm_counts, x="count", y="ptm_type", ax=axes[1], palette="viridis", hue="ptm_type", legend=False)
    axes[1].set_title("PTM type composition")
    axes[1].set_xlabel("Rows")
    axes[1].set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure1_dataset_composition.png", dpi=300)
    plt.close(fig)

    main = metrics[metrics["model"].isin(["logistic", "random_forest", "degree_only", "combined_shuffled_labels"])].copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=main, x="split_col", y="auprc", hue="model", ax=ax)
    ax.set_title("Performance collapses should be read under leakage-resistant splits")
    ax.set_xlabel("")
    ax.set_ylabel("AUPRC")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure2_split_leakage_performance.png", dpi=300)
    plt.close(fig)

    strict = metrics[(metrics["split_col"] == "site_disjoint_split") & (~metrics["model"].eq("combined_shuffled_labels"))].copy()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    order = strict.sort_values("auprc", ascending=False)["model"].tolist()
    sns.barplot(data=strict, y="model", x="auprc", order=order, ax=ax, palette="mako", hue="model", legend=False)
    ax.set_title("Model comparison on site-disjoint test rows")
    ax.set_xlabel("AUPRC")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure3_site_disjoint_model_comparison.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = ablations.sort_values("auprc", ascending=False)["model"].tolist()
    sns.barplot(data=ablations, y="model", x="auprc", order=order, ax=ax, palette="rocket", hue="model", legend=False)
    ax.set_title("Ablation under site-disjoint split")
    ax.set_xlabel("AUPRC")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure4_ablation_site_disjoint.png", dpi=300)
    plt.close(fig)

    if not predictions.empty:
        case = predictions[
            predictions["modified_gene"].str.contains("YWHA|TP53|PIN1|GRB2|PLK1", case=False, na=False)
            | predictions["partner_gene"].str.contains("YWHA|TP53|PIN1|GRB2|PLK1", case=False, na=False)
            | predictions["disease"].str.contains("cancer|carcinoma|tumor|leukemia|lymphoma", case=False, na=False)
        ].copy()
        if len(case) < 10:
            case = predictions.sort_values("confidence", ascending=False).head(60).copy()
        else:
            case = case.sort_values("confidence", ascending=False).head(80)
        graph = nx.Graph()
        for row in case.itertuples(index=False):
            graph.add_edge(row.modified_gene, row.partner_gene, label=f"{row.ptm_type}-{row.residue}{row.position}")
        fig, ax = plt.subplots(figsize=(10, 8))
        if graph.number_of_edges() > 0:
            pos = nx.spring_layout(graph, seed=RNG_SEED, k=0.7)
            degrees = dict(graph.degree())
            sizes = [80 + 45 * degrees[n] for n in graph.nodes]
            nx.draw_networkx_edges(graph, pos, alpha=0.35, width=1.0, ax=ax)
            nx.draw_networkx_nodes(graph, pos, node_size=sizes, node_color="#4C78A8", alpha=0.9, ax=ax)
            nx.draw_networkx_labels(graph, pos, font_size=7, ax=ax)
        ax.set_title("High-confidence held-out PTM-switch case-study network")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(FIGURES / "figure5_case_study_network.png", dpi=300)
        plt.close(fig)


def choose_claim(metrics: pd.DataFrame, ablations: pd.DataFrame, stats: dict[str, int]) -> str:
    site = metrics[metrics["split_col"] == "site_disjoint_split"].copy()
    site_main = site[site["model"].isin(["logistic", "random_forest", "mechanism_logistic", "sequence_mlp", "site_window_logistic"])].copy()
    best = site_main.sort_values("auprc", ascending=False).head(1)
    best_model = "NA" if best.empty else str(best.iloc[0]["model"])
    best_auprc = float("nan") if best.empty else float(best.iloc[0]["auprc"])
    best_auroc = float("nan") if best.empty else float(best.iloc[0]["auroc"])
    degree = site[site["model"] == "degree_only"]
    shuffle = site[site["model"] == "combined_shuffled_labels"]
    degree_auprc = float("nan") if degree.empty else float(degree.iloc[0]["auprc"])
    shuffle_auprc = float("nan") if shuffle.empty else float(shuffle.iloc[0]["auprc"])
    random_best = metrics[
        (metrics["split_col"] == "random_split")
        & metrics["model"].isin(["logistic", "random_forest", "mechanism_logistic", "sequence_mlp", "site_window_logistic"])
    ].sort_values("auprc", ascending=False)
    random_auprc = float("nan") if random_best.empty else float(random_best.iloc[0]["auprc"])
    ab_best = ablations.sort_values("auprc", ascending=False).head(1)
    ablation_phrase = "NA" if ab_best.empty else f"{ab_best.iloc[0]['model']} AUPRC={ab_best.iloc[0]['auprc']:.3f}"

    if not math.isnan(best_auprc) and not math.isnan(degree_auprc) and best_auprc > degree_auprc + 0.03:
        central = (
            "A nonredundant, leakage-audited benchmark for signed PTM-regulated PPI effects shows that strict "
            "site-disjoint evaluation remains learnable, but hub/degree baselines are strong and must be reported "
            "as first-class controls. The model result should be framed as a robust baseline improvement, not as "
            "a solved predictor."
        )
    else:
        central = (
            "PTM-conditioned PPI prediction is strongly split-sensitive; the primary contribution is a "
            "nonredundant leakage-audited benchmark and cautious evidence about which baseline signals survive "
            "strict evaluation."
        )

    return f"""Final central claim:
{central}

Strongest result:
After QC, the benchmark contains {stats['benchmark_rows']} PTM-regulated PPI evidence rows, {stats['unique_sites']} unique PTM sites, {stats['unique_unordered_pairs']} unordered protein pairs, and {stats['unique_pmids']} PMIDs. On the site-disjoint split, the best non-control model is {best_model} with AUPRC={best_auprc:.3f} and AUROC={best_auroc:.3f}. The random-split best AUPRC is {random_auprc:.3f}, which should be shown only as a leakage-sensitivity contrast.

Validity controls:
The site-disjoint degree-only baseline has AUPRC={degree_auprc:.3f}. The shuffled-label control has AUPRC={shuffle_auprc:.3f}. The best site-disjoint ablation is {ablation_phrase}. Split leakage audits are saved in results/tables/split_leakage_report.tsv and should be reported before random-split results.

Weakest result / limitation:
This sprint used PTMint as the primary positive evidence source and did not establish true no-effect negatives or independent experimental validation. ESM2 embeddings were not used in the completed run unless the runtime reported GPU/model availability; deterministic sequence hashes and mechanism features were used as the fast fallback. Claims must avoid clinical, causal, or experimentally validated language.

Novelty positioning:
Do not claim the first PTM-PPI database or first phosphorylation-PPI predictor. PTMint, PhosPPI, DeepPhosPPI, PTMcode, SIGNOR/OmniPath, BioGRID PTM, and ELM already cover adjacent territory. The safest novelty claim is the strict, leakage-audited signed PTM-PPI effect benchmark with explicit degree/shuffled-label controls and reusable baselines.

Exact journal fit recommendation:
Best immediate fit is Briefings in Bioinformatics if framed as a leakage-resistant benchmark plus method/resource for PTM-conditioned interactome learning. Nucleic Acids Research becomes stronger if the dataset/resource is hardened into a public web/server or database-style resource. Nature Communications or Nature Computational Science requires an external validation set, ideally perturbation AP-MS/proximity labeling or phosphomimetic/non-phosphorylatable mutant interaction evidence.

Unsafe claims to avoid:
Do not claim discovery of novel interactions, clinical utility, causal mechanisms, or state-of-the-art superiority unless all competing methods are re-run under identical strict splits. Do not treat unlabeled pairs as no-effect negatives.

Next validation needed for Nature-level submission:
Freeze the benchmark and model, then test on independent post-training evidence or collaborator data: kinase/phosphatase perturbation phosphoproteomics paired with AP-MS/proximity labeling, phosphomimetic/non-phosphorylatable mutant PPI assays, or time-split PTM-PPI literature released after the benchmark cutoff.
"""


def try_record_esm_status() -> dict[str, object]:
    status = {
        "torch_available": False,
        "cuda_available": False,
        "esm_model_attempted": False,
        "esm_model_loaded": False,
        "embedding_mode_used": "deterministic_hashed_sequence_features",
    }
    try:
        import torch

        status["torch_available"] = True
        status["cuda_available"] = bool(torch.cuda.is_available())
        if os.environ.get("SWITCHPPI_TRY_ESM", "0") == "1":
            status["esm_model_attempted"] = True
            from transformers import AutoTokenizer, EsmModel

            AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
            EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
            status["esm_model_loaded"] = True
            status["embedding_mode_used"] = "esm2_t12_35M_available_but_not_used_in_fast_sprint"
    except Exception as exc:
        status["esm_error"] = str(exc)
    return status


def main() -> None:
    ensure_dirs()
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    esm_status = try_record_esm_status()
    df, stats = normalize_ptmint()
    df, audit = add_splits(df)
    features = build_feature_pack(df)
    metrics, ablations, predictions = train_models(df, features)
    stats.update(esm_status)
    write_tables(df, audit, metrics, ablations, predictions, stats)
    make_figures(df, metrics, ablations, predictions)
    claims = choose_claim(metrics, ablations, stats)
    (ROOT / "results" / "claims_for_paper.txt").write_text(claims, encoding="utf-8")
    log(f"Done. Tables: {TABLES}")
    log(f"Done. Figures: {FIGURES}")
    log(f"Claims: {ROOT / 'results' / 'claims_for_paper.txt'}")


if __name__ == "__main__":
    main()
