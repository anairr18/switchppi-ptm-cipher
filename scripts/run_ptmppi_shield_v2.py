from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable

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
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
TABLES_V1 = ROOT / "results" / "tables"
RESULTS = ROOT / "results_v2"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "models_v2"

BENCHMARK_V1 = TABLES_V1 / "benchmark_dataset.tsv"
UNIPROT_CACHE = RAW / "uniprot_sequences.json"
PUBMED_CACHE = RAW / "pubmed_year_cache.json"

AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA)
SEED = 4242

PTM_CHEMISTRY = {
    "Phos": {"mass": 79.966, "charge": -2.0, "hbond": 2.0, "hydrophobicity": -1.0, "volume": 1.25, "reversible": 1.0},
    "Ac": {"mass": 42.011, "charge": -1.0, "hbond": -0.5, "hydrophobicity": 0.35, "volume": 0.65, "reversible": 1.0},
    "Me": {"mass": 14.016, "charge": 0.0, "hbond": -0.2, "hydrophobicity": 0.45, "volume": 0.35, "reversible": 1.0},
    "Ub": {"mass": 8564.0, "charge": 0.0, "hbond": 3.0, "hydrophobicity": -0.2, "volume": 8.0, "reversible": 1.0},
    "Sumo": {"mass": 11000.0, "charge": 0.0, "hbond": 3.0, "hydrophobicity": -0.2, "volume": 8.5, "reversible": 1.0},
    "Glyco": {"mass": 203.0, "charge": 0.0, "hbond": 4.0, "hydrophobicity": -1.0, "volume": 2.0, "reversible": 1.0},
}


def ensure_dirs() -> None:
    for path in [TABLES, FIGURES, MODELS]:
        path.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(f"[ptmppi-shield-v2] {message}", flush=True)


def load_sequences() -> dict[str, str]:
    if not UNIPROT_CACHE.exists():
        raise FileNotFoundError(f"Missing UniProt cache from v1 run: {UNIPROT_CACHE}")
    return json.loads(UNIPROT_CACHE.read_text(encoding="utf-8"))


def window_around_site(seq: str, position: int, flank: int = 15) -> str:
    idx = int(position) - 1
    start = max(0, idx - flank)
    end = min(len(seq), idx + flank + 1)
    return seq[start:end]


def parse_pub_year(pubdate: str) -> int | None:
    if not pubdate:
        return None
    match = re.search(r"(19|20)\d{2}", str(pubdate))
    if not match:
        return None
    return int(match.group(0))


def fetch_pubmed_years(pmids: list[str]) -> dict[str, int | None]:
    cache: dict[str, int | None] = {}
    if PUBMED_CACHE.exists():
        cache = json.loads(PUBMED_CACHE.read_text(encoding="utf-8"))
    missing = [p for p in sorted(set(map(str, pmids))) if p and p.lower() != "nan" and p not in cache]
    if not missing:
        return cache

    log(f"Fetching PubMed years for {len(missing):,} PMIDs")
    session = requests.Session()
    for i in range(0, len(missing), 180):
        chunk = missing[i : i + 180]
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params = {"db": "pubmed", "id": ",".join(chunk), "retmode": "json"}
        try:
            response = session.get(url, params=params, timeout=45)
            response.raise_for_status()
            data = response.json().get("result", {})
            for pmid in chunk:
                entry = data.get(pmid, {})
                cache[pmid] = parse_pub_year(entry.get("pubdate", "")) if entry else None
        except Exception as exc:
            log(f"WARNING: PubMed chunk failed: {exc}")
            for pmid in chunk:
                cache.setdefault(pmid, None)
        time.sleep(0.34)
    PUBMED_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    return cache


def assay_family(method: str) -> str:
    m = str(method).lower()
    if re.search(r"co-?ip|immunoprecip", m):
        return "co_ip"
    if re.search(r"pull.?down|pulldown|affinity chromatography|tap", m):
        return "pull_down"
    if re.search(r"spr|surface plasmon|bli|biolayer|itc|calorimetry|mst|thermophoresis|fluorescence polarization|anisotropy|nmr", m):
        return "biophysical"
    if re.search(r"two.hybrid|2-hybrid|y2h", m):
        return "two_hybrid"
    if re.search(r"mass spectrometry|lc-ms|ms/ms|proteomic|sds-page", m):
        return "proteomics"
    if re.search(r"microarray|array|chip", m):
        return "array"
    if re.search(r"structure|crystal|cryo|pdb|x-ray|docking|molecular dynamics|\\bmd\\b", m):
        return "structure_or_simulation"
    if re.search(r"colocal|microscop|immunofluorescence|confocal", m):
        return "localization"
    if re.search(r"western|blot|overlay", m):
        return "blot_overlay"
    return "other"


def motif_family(row: pd.Series, window: str) -> str:
    center = len(window) // 2
    partner = str(row.get("partner_gene", "")).upper()
    mod_gene = str(row.get("modified_gene", "")).upper()
    ptm = row["ptm_type"]
    residue = row["residue"]
    seq = window.upper()
    plus1 = seq[center + 1] if center + 1 < len(seq) else ""
    upstream = seq[max(0, center - 5) : center]
    downstream = seq[center + 1 : center + 7]
    if ptm == "Phos":
        if partner.startswith("YWHA") or mod_gene.startswith("YWHA"):
            return "reader_14_3_3_edge"
        if residue == "Y":
            return "phosphotyrosine_reader_candidate"
        if plus1 == "P":
            return "proline_directed_pin1_ww_candidate"
        if re.search(r"R..?$|R...?$", upstream):
            return "basophilic_phosphosite_candidate"
        if re.search(r"[DE].{0,3}$", upstream) or re.search(r"^[DE].{0,3}", downstream):
            return "acidic_phosphosite_candidate"
        return "other_phosphosite"
    if ptm == "Ac" and residue == "K":
        return "acetyl_lysine_reader_candidate"
    if ptm == "Me" and residue in {"K", "R"}:
        return "methyl_reader_candidate"
    if ptm in {"Ub", "Sumo"} and residue == "K":
        return "ubiquitin_like_lysine"
    if ptm == "Glyco":
        return "glycosylation_site"
    return "other_ptm"


def kinase_proxy(row: pd.Series, window: str) -> str:
    if row["ptm_type"] != "Phos":
        return f"non_phos_{row['ptm_type']}"
    center = len(window) // 2
    seq = window.upper()
    residue = row["residue"]
    plus1 = seq[center + 1] if center + 1 < len(seq) else ""
    upstream = seq[max(0, center - 5) : center]
    if residue == "Y":
        return "tyrosine_kinase_proxy"
    if plus1 == "P":
        return "proline_directed_kinase_proxy"
    if re.search(r"[RK].{0,3}$", upstream):
        return "basophilic_kinase_proxy"
    if re.search(r"[DE].{0,4}$", upstream):
        return "acidophilic_kinase_proxy"
    return "other_ser_thr_kinase_proxy"


def kmers(seq: str, k: int = 3) -> set[str]:
    seq = "".join(a for a in str(seq).upper() if a in AA_SET)
    if len(seq) < k:
        return set()
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def sketch(seq: str, n: int = 64, k: int = 3) -> tuple[int, ...]:
    vals = []
    for token in kmers(seq, k):
        vals.append(int(hashlib.md5(token.encode("ascii")).hexdigest()[:8], 16))
    vals.sort()
    return tuple(vals[:n])


def sketch_jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def build_sequence_clusters(sequences: dict[str, str], accessions: list[str], threshold: float = 0.72) -> dict[str, int]:
    accessions = sorted(set(accessions))
    sketches = {acc: sketch(sequences.get(acc, "")) for acc in accessions}
    graph = nx.Graph()
    graph.add_nodes_from(accessions)
    buckets: dict[int, list[str]] = defaultdict(list)
    for acc, sk in sketches.items():
        for value in sk[:8]:
            buckets[value].append(acc)
    candidates: set[tuple[str, str]] = set()
    for members in buckets.values():
        if len(members) > 200:
            continue
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                candidates.add((a, b) if a < b else (b, a))
    for a, b in candidates:
        if sketch_jaccard(sketches[a], sketches[b]) >= threshold:
            graph.add_edge(a, b)
    clusters = {}
    for cid, comp in enumerate(nx.connected_components(graph)):
        for acc in comp:
            clusters[acc] = cid
    return clusters


def topology_communities(df: pd.DataFrame) -> dict[str, int]:
    graph = nx.Graph()
    for row in df.itertuples(index=False):
        graph.add_edge(row.modified_uniprot, row.partner_uniprot)
    if graph.number_of_edges() == 0:
        return {}
    communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
    mapping: dict[str, int] = {}
    for cid, nodes in enumerate(communities):
        for node in nodes:
            mapping[node] = cid
    return mapping


def local_window_cluster_key(window: str, motif: str) -> str:
    digest = hashlib.md5("".join(sorted(kmers(window, 2))).encode("ascii")).hexdigest()[:8]
    return f"{motif}|{digest}"


def build_event_table() -> pd.DataFrame:
    if not BENCHMARK_V1.exists():
        raise FileNotFoundError(f"Run v1 first; missing {BENCHMARK_V1}")
    sequences = load_sequences()
    df = pd.read_csv(BENCHMARK_V1, sep="\t")
    df["modified_sequence"] = df["modified_uniprot"].map(sequences)
    df["partner_sequence"] = df["partner_uniprot"].map(sequences)
    df = df[df["modified_sequence"].notna() & df["partner_sequence"].notna()].copy()
    df["site_window_31"] = [
        window_around_site(seq, int(pos), 15) for seq, pos in zip(df["modified_sequence"], df["position"])
    ]
    df["assay_family"] = df["detection_method"].map(assay_family)
    years = fetch_pubmed_years(df["pmid"].astype(str).tolist())
    df["publication_year"] = df["pmid"].astype(str).map(years).astype("Int64")
    df["motif_family"] = [motif_family(row, row["site_window_31"]) for _, row in df.iterrows()]
    df["kinase_proxy"] = [kinase_proxy(row, row["site_window_31"]) for _, row in df.iterrows()]
    df["motif_cluster"] = [local_window_cluster_key(w, m) for w, m in zip(df["site_window_31"], df["motif_family"])]
    accessions = sorted(set(df["modified_uniprot"]) | set(df["partner_uniprot"]))
    clusters = build_sequence_clusters(sequences, accessions)
    df["modified_homology_cluster"] = df["modified_uniprot"].map(clusters).astype(str)
    df["partner_homology_cluster"] = df["partner_uniprot"].map(clusters).astype(str)
    df["homology_pair_cluster"] = [
        "||".join(sorted([a, b])) for a, b in zip(df["modified_homology_cluster"], df["partner_homology_cluster"])
    ]
    topo = topology_communities(df)
    df["modified_topology_community"] = df["modified_uniprot"].map(topo).fillna(-1).astype(int).astype(str)
    df["partner_topology_community"] = df["partner_uniprot"].map(topo).fillna(-1).astype(int).astype(str)
    df["topology_pair_community"] = [
        "||".join(sorted([a, b])) for a, b in zip(df["modified_topology_community"], df["partner_topology_community"])
    ]
    df["event_id"] = [f"event_{i:05d}" for i in range(len(df))]
    df["label_binary"] = (df["effect_label"] == "enhance").astype(int)
    return df.reset_index(drop=True)


def assign_random_split(n: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.array(["train"] * n, dtype=object)
    order = rng.permutation(n)
    n_train = int(0.70 * n)
    n_valid = int(0.15 * n)
    values[order[n_train : n_train + n_valid]] = "valid"
    values[order[n_train + n_valid :]] = "test"
    return values


def assign_group_split(df: pd.DataFrame, group_col: str, seed: int = SEED) -> np.ndarray:
    rng = random.Random(seed)
    groups = list(df.groupby(group_col, dropna=False).size().items())
    rng.shuffle(groups)
    groups.sort(key=lambda x: (x[1], rng.random()), reverse=True)
    targets = {"train": len(df) * 0.70, "valid": len(df) * 0.15, "test": len(df) * 0.15}
    totals = {"train": 0, "valid": 0, "test": 0}
    assignment = {}
    for group, size in groups:
        split = max(targets, key=lambda key: targets[key] - totals[key])
        assignment[group] = split
        totals[split] += int(size)
    return df[group_col].map(assignment).to_numpy(dtype=object)


def assign_temporal_split(df: pd.DataFrame) -> np.ndarray:
    years = df["publication_year"].dropna().astype(int)
    if years.nunique() < 6:
        return assign_group_split(df, "pmid", SEED + 7)
    train_cut = int(np.quantile(years, 0.70))
    valid_cut = int(np.quantile(years, 0.85))
    split = np.array(["train"] * len(df), dtype=object)
    split[df["publication_year"].astype("float").gt(train_cut).fillna(False).to_numpy()] = "valid"
    split[df["publication_year"].astype("float").gt(valid_cut).fillna(False).to_numpy()] = "test"
    # Put undated evidence in train so it cannot inflate prospective claims.
    split[df["publication_year"].isna().to_numpy()] = "train"
    return split


def build_full_shield_components(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    graph = nx.Graph()
    graph.add_nodes_from(df.index.tolist())
    leakage_axes = {
        "pair_key": "same_pair",
        "site_key": "same_site",
        "pmid": "same_publication",
        "homology_pair_cluster": "same_homology_pair",
        "motif_cluster": "same_motif_window_cluster",
        "topology_pair_community": "same_topology_community_pair",
    }
    axis_edge_counts = []
    for col, label in leakage_axes.items():
        count = 0
        for _, members in df.groupby(col, dropna=False).groups.items():
            members = list(members)
            if len(members) > 350:
                # Avoid one broad proxy swallowing the whole benchmark; audit it separately.
                continue
            for i, a in enumerate(members):
                for b in members[i + 1 :]:
                    graph.add_edge(int(a), int(b), axis=label)
                    count += 1
        axis_edge_counts.append({"axis": label, "edges_added": count})
    component_id = np.empty(len(df), dtype=object)
    for cid, comp in enumerate(nx.connected_components(graph)):
        for idx in comp:
            component_id[idx] = f"shield_component_{cid}"
    return component_id, pd.DataFrame(axis_edge_counts)


def add_shield_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    full_components, full_edges = build_full_shield_components(df)
    df["full_shield_component"] = full_components
    split_specs = {
        "S0_random_split": None,
        "S1_cold_homology_split": "homology_pair_cluster",
        "S2_cold_edge_split": "pair_key",
        "S3_source_publication_split": "pmid",
        "S4_assay_out_split": "assay_family",
        "S5_motif_family_split": "motif_cluster",
        "S6_kinase_proxy_split": "kinase_proxy",
        "S7_temporal_prospective_split": "publication_year",
        "S8_topology_shielded_split": "topology_pair_community",
        "S9_full_shield_split": "full_shield_component",
    }
    for i, (split_col, group_col) in enumerate(split_specs.items()):
        if split_col == "S0_random_split":
            df[split_col] = assign_random_split(len(df), SEED + i)
        elif split_col == "S7_temporal_prospective_split":
            df[split_col] = assign_temporal_split(df)
        else:
            assert group_col is not None
            df[split_col] = assign_group_split(df, group_col, SEED + i)

    audit_rows = []
    for split_col, group_col in split_specs.items():
        if split_col == "S0_random_split":
            audit_rows.append(
                {
                    "split_col": split_col,
                    "held_out_axis": "none",
                    "train_rows": int((df[split_col] == "train").sum()),
                    "valid_rows": int((df[split_col] == "valid").sum()),
                    "test_rows": int((df[split_col] == "test").sum()),
                    "train_valid_overlap": np.nan,
                    "train_test_overlap": np.nan,
                    "valid_test_overlap": np.nan,
                    "test_pos_rate": float(df.loc[df[split_col] == "test", "label_binary"].mean()),
                }
            )
            continue
        assert group_col is not None
        groups = {
            part: set(df.loc[df[split_col] == part, group_col].dropna().astype(str))
            for part in ["train", "valid", "test"]
        }
        audit_rows.append(
            {
                "split_col": split_col,
                "held_out_axis": group_col,
                "train_rows": int((df[split_col] == "train").sum()),
                "valid_rows": int((df[split_col] == "valid").sum()),
                "test_rows": int((df[split_col] == "test").sum()),
                "train_valid_overlap": len(groups["train"] & groups["valid"]),
                "train_test_overlap": len(groups["train"] & groups["test"]),
                "valid_test_overlap": len(groups["valid"] & groups["test"]),
                "test_pos_rate": float(df.loc[df[split_col] == "test", "label_binary"].mean()),
            }
        )
    return df, pd.DataFrame(audit_rows), full_edges


def aa_comp(seq: str) -> np.ndarray:
    seq = "".join(a for a in str(seq).upper() if a in AA_SET)
    if not seq:
        return np.zeros(len(AA), dtype=np.float32)
    return np.array([seq.count(a) for a in AA], dtype=np.float32) / len(seq)


def physchem(seq: str) -> np.ndarray:
    seq = "".join(a for a in str(seq).upper() if a in AA_SET)
    if not seq:
        return np.zeros(8, dtype=np.float32)
    groups = [
        set("AILMFWVY"),
        set("DEKRH"),
        set("DE"),
        set("KRH"),
        set("STNQCY"),
        set("PG"),
        set("FWY"),
    ]
    vals = [math.log1p(len(seq))]
    vals.extend(sum(a in g for a in seq) / len(seq) for g in groups)
    return np.asarray(vals, dtype=np.float32)


def hashed_kmers(seq: str, dims: int, k: int, salt: str) -> np.ndarray:
    seq = "".join(a for a in str(seq).upper() if a in AA_SET)
    out = np.zeros(dims, dtype=np.float32)
    if len(seq) < k:
        return out
    for i in range(len(seq) - k + 1):
        token = salt + seq[i : i + k]
        digest = hashlib.md5(token.encode("ascii")).hexdigest()
        idx = int(digest[:8], 16) % dims
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        out[idx] += sign
    norm = np.linalg.norm(out)
    return out / norm if norm > 0 else out


def ptm_chem_vector(ptm: str) -> np.ndarray:
    values = PTM_CHEMISTRY.get(ptm, {"mass": 0, "charge": 0, "hbond": 0, "hydrophobicity": 0, "volume": 0, "reversible": 0})
    return np.array(
        [
            math.log1p(abs(values["mass"])),
            values["charge"],
            values["hbond"],
            values["hydrophobicity"],
            values["volume"],
            values["reversible"],
        ],
        dtype=np.float32,
    )


def one_hot_frame(df: pd.DataFrame, cols: list[str]) -> sparse.csr_matrix:
    frames = [pd.get_dummies(df[col].fillna("missing").astype(str), prefix=col, dtype=np.float32) for col in cols]
    return sparse.csr_matrix(pd.concat(frames, axis=1).to_numpy(np.float32))


def train_topology_features(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> sparse.csr_matrix:
    graph = nx.Graph()
    for row in train_df.itertuples(index=False):
        graph.add_edge(row.modified_uniprot, row.partner_uniprot)
    vals = []
    for row in eval_df.itertuples(index=False):
        a, b = row.modified_uniprot, row.partner_uniprot
        deg_a = graph.degree(a) if graph.has_node(a) else 0
        deg_b = graph.degree(b) if graph.has_node(b) else 0
        common = len(list(nx.common_neighbors(graph, a, b))) if graph.has_node(a) and graph.has_node(b) else 0
        same_component = 0.0
        if graph.has_node(a) and graph.has_node(b):
            same_component = float(nx.has_path(graph, a, b))
        vals.append([math.log1p(deg_a), math.log1p(deg_b), math.log1p(common), same_component, float(graph.has_edge(a, b))])
    return sparse.csr_matrix(np.asarray(vals, dtype=np.float32))


def build_static_feature_matrices(df: pd.DataFrame) -> dict[str, sparse.csr_matrix]:
    log("Building v2 counterfactual/static feature matrices")
    position_norm = (df["position"].astype(float) / df["modified_sequence"].str.len().astype(float)).fillna(0).to_numpy(np.float32)[:, None]
    window_feats = np.vstack(
        [
            np.concatenate([aa_comp(w), physchem(w), hashed_kmers(w, 96, 2, "site")])
            for w in df["site_window_31"]
        ]
    )
    mod_protein = np.vstack([np.concatenate([aa_comp(s), physchem(s), hashed_kmers(s, 128, 3, "mod")]) for s in df["modified_sequence"]])
    partner_protein = np.vstack([np.concatenate([aa_comp(s), physchem(s), hashed_kmers(s, 128, 3, "partner")]) for s in df["partner_sequence"]])
    ptm_chem = np.vstack([ptm_chem_vector(p) for p in df["ptm_type"]])
    delta = np.hstack([ptm_chem, position_norm])
    unmodified_state = np.hstack([window_feats, mod_protein, partner_protein, position_norm])
    modified_state = np.hstack([window_feats, mod_protein, partner_protein, position_norm, ptm_chem])
    counterfactual = np.hstack([unmodified_state, modified_state, delta, modified_state[:, : min(64, modified_state.shape[1])] - unmodified_state[:, : min(64, unmodified_state.shape[1])]])
    counterfactual_no_ptmchem = np.hstack([unmodified_state, window_feats, mod_protein, partner_protein, position_norm])
    delta_only = np.hstack([ptm_chem, position_norm, window_feats[:, :34]])
    motif = sparse.hstack([one_hot_frame(df, ["ptm_type", "residue", "motif_family", "kinase_proxy"]), sparse.csr_matrix(window_feats[:, :34])], format="csr")
    source_assay = one_hot_frame(df, ["assay_family", "detection_method"])
    counterfactual_mat = sparse.hstack(
        [
            sparse.csr_matrix(counterfactual),
            one_hot_frame(df, ["ptm_type", "residue", "motif_family", "assay_family"]),
        ],
        format="csr",
    )
    sequence_only = sparse.csr_matrix(np.hstack([window_feats, mod_protein, partner_protein, position_norm]))
    return {
        "source_assay": source_assay,
        "motif": motif,
        "sequence_only": sequence_only,
        "counterfactual": counterfactual_mat,
        "counterfactual_no_ptmchem": sparse.hstack(
            [
                sparse.csr_matrix(counterfactual_no_ptmchem),
                one_hot_frame(df, ["residue", "motif_family", "assay_family"]),
            ],
            format="csr",
        ),
        "ptm_delta_only": sparse.hstack(
            [
                sparse.csr_matrix(delta_only),
                one_hot_frame(df, ["ptm_type", "residue"]),
            ],
            format="csr",
        ),
    }


def ece_score(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(prob[mask].mean() - y_true[mask].mean())
    return float(total)


def best_valid_threshold(y_valid: np.ndarray, prob_valid: np.ndarray) -> float:
    if len(y_valid) == 0 or len(np.unique(y_valid)) < 2:
        return 0.5
    candidates = np.unique(np.quantile(prob_valid, np.linspace(0.02, 0.98, 97)))
    best_t = 0.5
    best_score = -1.0
    for threshold in candidates:
        pred = (prob_valid >= threshold).astype(int)
        score = matthews_corrcoef(y_valid, pred)
        if score > best_score:
            best_score = float(score)
            best_t = float(threshold)
    return best_t


def precision_at_k(y_true: np.ndarray, prob: np.ndarray, k: int = 50) -> float:
    if len(y_true) == 0:
        return float("nan")
    k = min(k, len(y_true))
    order = np.argsort(prob)[::-1][:k]
    return float(precision_score(y_true[order], np.ones(k), zero_division=0))


def bootstrap_ci(y_true: np.ndarray, prob: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], float], seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y_true)
    for _ in range(150):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric(y_true[idx], prob[idx]))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def fit_predict(model_name: str, x_train: sparse.csr_matrix, y_train: np.ndarray, x_valid: sparse.csr_matrix, x_test: sparse.csr_matrix, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "class_prior":
        p = float(np.mean(y_train))
        return np.full(x_valid.shape[0], p), np.full(x_test.shape[0], p)
    if model_name.endswith("logistic") or model_name in {"topology_only", "source_assay_only", "motif_only", "ptm_delta_only"}:
        clf = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=2500, class_weight="balanced", solver="liblinear", random_state=seed),
        )
    elif model_name == "sequence_random_forest":
        clf = RandomForestClassifier(n_estimators=350, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    elif model_name == "counterfactual_mlp":
        clf = make_pipeline(
            StandardScaler(with_mean=False),
            MLPClassifier(hidden_layer_sizes=(128, 48), alpha=2e-3, early_stopping=True, max_iter=160, random_state=seed),
        )
    else:
        raise ValueError(model_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(x_train, y_train)
    return clf.predict_proba(x_valid)[:, 1], clf.predict_proba(x_test)[:, 1]


def metric_row(
    split_col: str,
    model: str,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    prob_valid: np.ndarray,
    prob: np.ndarray,
    seed: int,
) -> dict[str, object]:
    threshold = best_valid_threshold(y_valid, prob_valid)
    pred = (prob >= threshold).astype(int)
    row = {
        "split_col": split_col,
        "model": model,
        "train_n": int(len(y_train)),
        "valid_n": int(len(y_valid)),
        "test_n": int(len(y_test)),
        "train_pos_rate": float(np.mean(y_train)),
        "valid_pos_rate": float(np.mean(y_valid)),
        "test_pos_rate": float(np.mean(y_test)),
        "valid_threshold_mcc": float(threshold),
        "auprc": float(average_precision_score(y_test, prob)) if len(np.unique(y_test)) > 1 else float("nan"),
        "auroc": float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) > 1 else float("nan"),
        "mcc": float(matthews_corrcoef(y_test, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "brier": float(brier_score_loss(y_test, prob)),
        "ece": ece_score(y_test, prob),
        "precision_at_50": precision_at_k(y_test, prob, 50),
    }
    if len(np.unique(y_test)) > 1:
        lo, hi = bootstrap_ci(y_test, prob, average_precision_score, seed)
        row["auprc_ci_low"] = lo
        row["auprc_ci_high"] = hi
    else:
        row["auprc_ci_low"] = float("nan")
        row["auprc_ci_high"] = float("nan")
    return row


def train_and_evaluate(df: pd.DataFrame, matrices: dict[str, sparse.csr_matrix]) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_cols = [c for c in df.columns if c.startswith("S") and c.endswith("_split")]
    specs = [
        ("class_prior", "prior"),
        ("source_assay_only", "source_assay"),
        ("topology_only", "topology"),
        ("motif_only", "motif"),
        ("sequence_logistic", "sequence_only"),
        ("sequence_random_forest", "sequence_only"),
        ("ptm_delta_only", "ptm_delta_only"),
        ("counterfactual_no_ptmchem_logistic", "counterfactual_no_ptmchem"),
        ("counterfactual_logistic", "counterfactual"),
        ("counterfactual_mlp", "counterfactual"),
    ]
    y = df["label_binary"].to_numpy(int)
    rows = []
    predictions = []
    for si, split_col in enumerate(split_cols):
        log(f"Evaluating {split_col}")
        train_idx = np.where(df[split_col].to_numpy() == "train")[0]
        valid_idx = np.where(df[split_col].to_numpy() == "valid")[0]
        test_idx = np.where(df[split_col].to_numpy() == "test")[0]
        if len(test_idx) < 25 or len(set(y[train_idx])) < 2 or len(set(y[test_idx])) < 2:
            log(f"  Skipping {split_col}: insufficient class diversity")
            continue
        topo_train = train_topology_features(df.iloc[train_idx], df.iloc[train_idx])
        topo_valid = train_topology_features(df.iloc[train_idx], df.iloc[valid_idx])
        topo_test = train_topology_features(df.iloc[train_idx], df.iloc[test_idx])
        for mi, (model_name, family) in enumerate(specs):
            if family == "prior":
                x_train = sparse.csr_matrix((len(train_idx), 1))
                x_valid = sparse.csr_matrix((len(valid_idx), 1))
                x_test = sparse.csr_matrix((len(test_idx), 1))
            elif family == "topology":
                x_train, x_valid, x_test = topo_train, topo_valid, topo_test
            else:
                base = matrices[family]
                if family in {"counterfactual", "counterfactual_no_ptmchem"}:
                    x_train = sparse.hstack([base[train_idx], topo_train], format="csr")
                    x_valid = sparse.hstack([base[valid_idx], topo_valid], format="csr")
                    x_test = sparse.hstack([base[test_idx], topo_test], format="csr")
                else:
                    x_train, x_valid, x_test = base[train_idx], base[valid_idx], base[test_idx]
            prob_valid, prob_test = fit_predict(model_name, x_train, y[train_idx], x_valid, x_test, SEED + si * 100 + mi)
            rows.append(metric_row(split_col, model_name, y[train_idx], y[valid_idx], y[test_idx], prob_valid, prob_test, SEED + si * 1000 + mi))
            if split_col == "S9_full_shield_split" and model_name in {"counterfactual_logistic", "counterfactual_mlp", "sequence_random_forest"}:
                pred = df.iloc[test_idx][
                    [
                        "event_id",
                        "modified_gene",
                        "partner_gene",
                        "ptm_type",
                        "residue",
                        "position",
                        "effect_label",
                        "assay_family",
                        "motif_family",
                        "publication_year",
                        "pmid",
                    ]
                ].copy()
                pred["model"] = model_name
                pred["prob_enhance"] = prob_test
                pred["confidence"] = np.maximum(prob_test, 1.0 - prob_test)
                predictions.append(pred)
        # Within-block shuffled control for the headline split.
        if split_col in {"S8_topology_shielded_split", "S9_full_shield_split"}:
            rng = np.random.default_rng(SEED + si)
            shuffled = y[train_idx].copy()
            rng.shuffle(shuffled)
            base = matrices["counterfactual"]
            x_train = sparse.hstack([base[train_idx], topo_train], format="csr")
            x_valid = sparse.hstack([base[valid_idx], topo_valid], format="csr")
            x_test = sparse.hstack([base[test_idx], topo_test], format="csr")
            prob_valid, prob_test = fit_predict("counterfactual_logistic", x_train, shuffled, x_valid, x_test, SEED + si * 99)
            rows.append(metric_row(split_col, "counterfactual_shuffled_labels", shuffled, y[valid_idx], y[test_idx], prob_valid, prob_test, SEED + si * 2000))
    metrics = pd.DataFrame(rows)
    preds = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return metrics, preds


def robust_discovery_scores(metrics: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    shield_splits = [s for s in metrics["split_col"].unique() if s != "S0_random_split"]
    rows = []
    audit_lookup = audit.set_index("split_col").to_dict("index")
    for model, sub in metrics[metrics["split_col"].isin(shield_splits)].groupby("model"):
        if model in {"class_prior", "counterfactual_shuffled_labels"}:
            continue
        auprc = sub["auprc"].dropna()
        mcc = sub["mcc"].dropna()
        p50 = sub["precision_at_50"].dropna()
        split_count = sub["split_col"].nunique()
        missing_penalty = max(0, len(shield_splits) - split_count) / max(1, len(shield_splits))
        leakage_penalty = 0.0
        for split in sub["split_col"].unique():
            rec = audit_lookup.get(split, {})
            overlaps = [rec.get("train_test_overlap", 0), rec.get("train_valid_overlap", 0), rec.get("valid_test_overlap", 0)]
            if any(pd.notna(x) and x > 0 for x in overlaps):
                leakage_penalty += 0.10
        score = 0.50 * auprc.mean() + 0.25 * ((mcc.mean() + 1) / 2) + 0.25 * p50.mean()
        worst = auprc.min() if len(auprc) else np.nan
        score = score - 0.20 * missing_penalty - leakage_penalty - max(0.0, 0.50 - worst) * 0.20
        rows.append(
            {
                "model": model,
                "robust_discovery_score": float(score),
                "mean_shield_auprc": float(auprc.mean()),
                "worst_shield_auprc": float(worst),
                "mean_shield_mcc": float(mcc.mean()),
                "mean_precision_at_50": float(p50.mean()),
                "evaluated_shield_splits": int(split_count),
                "missing_split_penalty": float(missing_penalty),
                "leakage_penalty": float(leakage_penalty),
            }
        )
    return pd.DataFrame(rows).sort_values("robust_discovery_score", ascending=False)


def split_collapse_table(metrics: pd.DataFrame) -> pd.DataFrame:
    random = metrics[metrics["split_col"] == "S0_random_split"][["model", "auprc", "mcc", "ece"]].rename(
        columns={"auprc": "random_auprc", "mcc": "random_mcc", "ece": "random_ece"}
    )
    shield = (
        metrics[metrics["split_col"] != "S0_random_split"]
        .groupby("model", as_index=False)
        .agg(mean_shield_auprc=("auprc", "mean"), worst_shield_auprc=("auprc", "min"), mean_shield_mcc=("mcc", "mean"), mean_shield_ece=("ece", "mean"))
    )
    out = random.merge(shield, on="model", how="outer")
    out["random_to_mean_shield_auprc_drop"] = out["random_auprc"] - out["mean_shield_auprc"]
    out["random_to_worst_shield_auprc_drop"] = out["random_auprc"] - out["worst_shield_auprc"]
    return out.sort_values("random_to_worst_shield_auprc_drop", ascending=False)


def claim_gate_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rules = [
        ("new protein-family generalization", "S1_cold_homology_split", "topology_only"),
        ("new edge generalization", "S2_cold_edge_split", "topology_only"),
        ("new source/publication generalization", "S3_source_publication_split", "source_assay_only"),
        ("assay transfer", "S4_assay_out_split", "source_assay_only"),
        ("new motif mechanism generalization", "S5_motif_family_split", "motif_only"),
        ("kinase/readout transfer", "S6_kinase_proxy_split", "motif_only"),
        ("prospective temporal discovery", "S7_temporal_prospective_split", "topology_only"),
        ("topology-shielded network discovery", "S8_topology_shielded_split", "topology_only"),
        ("full-shield PTM-PPI discovery", "S9_full_shield_split", "topology_only"),
    ]
    candidate_models = [
        "counterfactual_mlp",
        "counterfactual_logistic",
        "counterfactual_no_ptmchem_logistic",
        "ptm_delta_only",
        "sequence_random_forest",
        "sequence_logistic",
    ]
    rows = []
    for claim, split, baseline in rules:
        sub = metrics[metrics["split_col"] == split]
        base = sub[sub["model"] == baseline]
        if base.empty:
            base_auprc = np.nan
        else:
            base_auprc = float(base.iloc[0]["auprc"])
        for model in candidate_models:
            rec = sub[sub["model"] == model]
            if rec.empty:
                continue
            auprc = float(rec.iloc[0]["auprc"])
            mcc = float(rec.iloc[0]["mcc"])
            ece = float(rec.iloc[0]["ece"])
            margin = auprc - base_auprc if pd.notna(base_auprc) else np.nan
            pass_gate = bool(pd.notna(margin) and margin >= 0.02 and mcc > 0.05 and ece < 0.25)
            rows.append(
                {
                    "claim": claim,
                    "split_col": split,
                    "model": model,
                    "baseline": baseline,
                    "model_auprc": auprc,
                    "baseline_auprc": base_auprc,
                    "auprc_margin": margin,
                    "mcc": mcc,
                    "ece": ece,
                    "passes_gate": pass_gate,
                }
            )
    return pd.DataFrame(rows)


def failure_taxonomy(metrics: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    full_split = "S9_full_shield_split"
    rows = []
    full = df[df[full_split] == "test"].copy()
    if full.empty:
        return pd.DataFrame()
    for axis in ["ptm_type", "assay_family", "motif_family", "kinase_proxy"]:
        counts = full.groupby(axis).agg(test_events=("event_id", "count"), enhance_rate=("label_binary", "mean")).reset_index()
        for _, rec in counts.iterrows():
            rows.append(
                {
                    "split_col": full_split,
                    "axis": axis,
                    "slice": rec[axis],
                    "test_events": int(rec["test_events"]),
                    "enhance_rate": float(rec["enhance_rate"]),
                    "risk": "small_slice" if rec["test_events"] < 20 else ("label_skew" if rec["enhance_rate"] < 0.15 or rec["enhance_rate"] > 0.85 else "ok"),
                }
            )
    return pd.DataFrame(rows).sort_values(["risk", "axis", "test_events"], ascending=[False, True, True])


def novelty_ablation_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    target_models = [
        "counterfactual_mlp",
        "counterfactual_logistic",
        "counterfactual_no_ptmchem_logistic",
        "ptm_delta_only",
        "sequence_random_forest",
        "topology_only",
        "motif_only",
        "source_assay_only",
    ]
    sub = metrics[(metrics["split_col"] != "S0_random_split") & metrics["model"].isin(target_models)].copy()
    summary = (
        sub.groupby("model", as_index=False)
        .agg(
            mean_auprc=("auprc", "mean"),
            worst_auprc=("auprc", "min"),
            mean_mcc=("mcc", "mean"),
            mean_ece=("ece", "mean"),
            gates_available=("split_col", "nunique"),
        )
        .sort_values("mean_auprc", ascending=False)
    )
    ref = summary.set_index("model")
    rows = []
    for model in summary["model"]:
        row = summary[summary["model"] == model].iloc[0].to_dict()
        if model.startswith("counterfactual"):
            row["delta_vs_topology_mean_auprc"] = row["mean_auprc"] - float(ref.loc["topology_only", "mean_auprc"]) if "topology_only" in ref.index else np.nan
            row["delta_vs_sequence_rf_mean_auprc"] = row["mean_auprc"] - float(ref.loc["sequence_random_forest", "mean_auprc"]) if "sequence_random_forest" in ref.index else np.nan
            row["delta_vs_no_ptmchem_mean_auprc"] = row["mean_auprc"] - float(ref.loc["counterfactual_no_ptmchem_logistic", "mean_auprc"]) if "counterfactual_no_ptmchem_logistic" in ref.index else np.nan
        else:
            row["delta_vs_topology_mean_auprc"] = np.nan
            row["delta_vs_sequence_rf_mean_auprc"] = np.nan
            row["delta_vs_no_ptmchem_mean_auprc"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_figures(df: pd.DataFrame, metrics: pd.DataFrame, robust: pd.DataFrame, collapse: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.countplot(data=df, y="assay_family", order=df["assay_family"].value_counts().index, ax=axes[0], color="#4C78A8")
    axes[0].set_title("Assay provenance")
    axes[0].set_xlabel("Events")
    axes[0].set_ylabel("")
    sns.countplot(data=df, y="motif_family", order=df["motif_family"].value_counts().index[:10], ax=axes[1], color="#59A14F")
    axes[1].set_title("PTM mechanism proxies")
    axes[1].set_xlabel("Events")
    axes[1].set_ylabel("")
    years = df["publication_year"].dropna().astype(int)
    sns.histplot(years, bins=24, ax=axes[2], color="#F28E2B")
    axes[2].set_title("Evidence publication years")
    axes[2].set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_v2_event_provenance.png", dpi=300)
    plt.close(fig)

    plot = metrics[metrics["model"].isin(["source_assay_only", "topology_only", "sequence_random_forest", "counterfactual_logistic", "counterfactual_mlp", "counterfactual_shuffled_labels"])].copy()
    fig, ax = plt.subplots(figsize=(13, 5.5))
    sns.barplot(data=plot, x="split_col", y="auprc", hue="model", ax=ax)
    ax.set_title("PTM-PPI Shield stress tests")
    ax.set_xlabel("")
    ax.set_ylabel("AUPRC")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_v2_shield_stress_tests.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=robust, y="model", x="robust_discovery_score", ax=ax, color="#9C755F")
    ax.set_title("Robust Discovery Score across shielded splits")
    ax.set_xlabel("Score")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_v2_robust_discovery_score.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot = collapse[collapse["model"].isin(["sequence_random_forest", "counterfactual_mlp", "counterfactual_logistic", "topology_only", "source_assay_only"])].copy()
    sns.barplot(data=plot, y="model", x="random_to_worst_shield_auprc_drop", ax=ax, color="#E15759")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Random-to-worst-shield AUPRC drop")
    ax.set_xlabel("AUPRC drop")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_v2_random_to_shield_drop.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot = metrics[metrics["split_col"].eq("S9_full_shield_split") & metrics["model"].isin(["sequence_random_forest", "counterfactual_mlp", "counterfactual_logistic", "counterfactual_no_ptmchem_logistic", "ptm_delta_only", "topology_only", "source_assay_only"])].copy()
    sns.scatterplot(data=plot, x="brier", y="ece", size="auprc", hue="model", ax=ax, sizes=(80, 260))
    ax.set_title("Full-shield calibration risk")
    ax.set_xlabel("Brier score")
    ax.set_ylabel("Expected calibration error")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_v2_calibration_risk.png", dpi=300)
    plt.close(fig)


def write_claims(df: pd.DataFrame, audit: pd.DataFrame, metrics: pd.DataFrame, robust: pd.DataFrame, gates: pd.DataFrame, collapse: pd.DataFrame) -> None:
    best = robust.head(1).iloc[0] if len(robust) else None
    full = metrics[metrics["split_col"] == "S9_full_shield_split"].sort_values("auprc", ascending=False)
    full_best = full[~full["model"].isin(["class_prior", "counterfactual_shuffled_labels"])].head(1)
    random = metrics[metrics["split_col"] == "S0_random_split"].sort_values("auprc", ascending=False).head(1)
    random_text = "NA" if random.empty else f"{random.iloc[0]['model']} AUPRC={random.iloc[0]['auprc']:.3f}"
    full_text = "NA" if full_best.empty else f"{full_best.iloc[0]['model']} AUPRC={full_best.iloc[0]['auprc']:.3f}, MCC={full_best.iloc[0]['mcc']:.3f}"
    best_text = "NA" if best is None else f"{best['model']} score={best['robust_discovery_score']:.3f}"
    gate_passes = gates[gates["passes_gate"] == True]
    counterfactual_gates = gate_passes[gate_passes["model"].str.startswith("counterfactual")]
    collapse_rf = collapse[collapse["model"] == "sequence_random_forest"]
    collapse_text = "NA" if collapse_rf.empty else f"{collapse_rf.iloc[0]['random_to_worst_shield_auprc_drop']:.3f}"
    claims = f"""PTM-PPI Shield v2 central claim:
This implementation upgrades the project from a PTMint classifier into an event-level PTM-PPI Shield benchmark: each evidence row carries assay, publication, motif/reader proxy, kinase proxy, homology-proxy, topology-community, and full-shield leakage-component annotations. The model result should be framed as a stress-tested baseline for counterfactual proteoform-edge prediction, not as a finished foundation model.

Dataset:
{len(df)} nonredundant PTM-PPI evidence events; {df['pmid'].nunique()} PMIDs; {df['assay_family'].nunique()} assay families; {df['motif_family'].nunique()} motif/mechanism proxy classes; {df['publication_year'].notna().sum()} events with PubMed year metadata.

Best shielded result:
Full-shield best non-control model: {full_text}.
Best Robust Discovery Score across non-random shield splits: {best_text}.
Random sanity split best: {random_text}. Random results are implementation checks, not discovery claims.
For sequence_random_forest, random-to-worst-shield AUPRC drop is {collapse_text}; use this as a leakage-sensitivity diagnostic.

Claim gates:
{len(gate_passes)} model/split claim gates passed under the current thresholds, including {len(counterfactual_gates)} counterfactual-model gates. See results_v2/tables/claim_gate_matrix_v2.tsv. A biological generalization claim is allowed only for rows where passes_gate=True.

What is now more novel:
1. Multi-axis leakage graph over event rows, not just pair/site split columns.
2. PTM-specific shield axes: motif-window, kinase proxy, assay family, publication/source, topology community, and homology-pair proxy.
3. Counterfactual feature construction with unmodified state, modified state, and explicit PTM-chemistry delta features.
4. Claim-gating: model claims are explicitly limited to shield axes where they beat topology/source/motif controls with usable MCC and calibration.

Still missing for Nature Computational Science:
1. True interface-similarity shielding with PINDER/PPIRef/iDist/Foldseek, not the current topology/homology proxy.
2. Reruns of PhosPPI, DeepPhosPPI, PTM-Mamba embeddings, Betts/Mechismo, ELM rules, and SIGNOR/OmniPath propagation under identical shields.
3. Independent prospective validation from post-cutoff literature or perturbation/AP-MS/proximity-labeling data.
4. Real ESM/PTM-Mamba/PTM-token model training on GPU. Current counterfactual model is a CPU-feasible prototype using deterministic sequence and PTM chemistry features.
5. Experimentally tested no-effect labels. Unknown or unlabeled PPIs must not be treated as no-effect negatives.

Safe manuscript sentence:
We introduce PTM-PPI Shield, a multi-axis leakage-audited evaluation framework for signed PTM-regulated protein-interaction effects, and show that counterfactual proteoform-state features can be evaluated against topology, source, motif, and publication shortcuts under substantially stricter discovery splits.
"""
    (RESULTS / "claims_for_ncs_upgrade.txt").write_text(claims, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    random.seed(SEED)
    np.random.seed(SEED)
    event_df = build_event_table()
    event_df, audit, full_edges = add_shield_splits(event_df)
    matrices = build_static_feature_matrices(event_df)
    metrics, predictions = train_and_evaluate(event_df, matrices)
    robust = robust_discovery_scores(metrics, audit)
    collapse = split_collapse_table(metrics)
    gates = claim_gate_table(metrics)
    failures = failure_taxonomy(metrics, event_df)
    ablations = novelty_ablation_summary(metrics)

    public_cols = [
        "event_id",
        "modified_uniprot",
        "modified_gene",
        "partner_uniprot",
        "partner_gene",
        "organism",
        "ptm_type",
        "residue",
        "position",
        "effect_label",
        "pmid",
        "publication_year",
        "detection_method",
        "assay_family",
        "motif_family",
        "motif_cluster",
        "kinase_proxy",
        "modified_homology_cluster",
        "partner_homology_cluster",
        "homology_pair_cluster",
        "topology_pair_community",
        "full_shield_component",
    ] + [c for c in event_df.columns if c.startswith("S") and c.endswith("_split")]
    event_df[public_cols].to_csv(TABLES / "event_table_v2.tsv", sep="\t", index=False)
    audit.to_csv(TABLES / "shield_split_audit_v2.tsv", sep="\t", index=False)
    full_edges.to_csv(TABLES / "full_shield_leakage_edges_v2.tsv", sep="\t", index=False)
    metrics.to_csv(TABLES / "shield_model_metrics_v2.tsv", sep="\t", index=False)
    robust.to_csv(TABLES / "robust_discovery_scores_v2.tsv", sep="\t", index=False)
    collapse.to_csv(TABLES / "split_collapse_diagnostics_v2.tsv", sep="\t", index=False)
    gates.to_csv(TABLES / "claim_gate_matrix_v2.tsv", sep="\t", index=False)
    failures.to_csv(TABLES / "failure_taxonomy_v2.tsv", sep="\t", index=False)
    ablations.to_csv(TABLES / "novelty_ablation_summary_v2.tsv", sep="\t", index=False)
    if len(predictions):
        predictions.sort_values("confidence", ascending=False).to_csv(TABLES / "full_shield_predictions_v2.tsv", sep="\t", index=False)
    else:
        pd.DataFrame().to_csv(TABLES / "full_shield_predictions_v2.tsv", sep="\t", index=False)
    write_figures(event_df, metrics, robust, collapse)
    write_claims(event_df, audit, metrics, robust, gates, collapse)
    log(f"Done. Results written to {RESULTS}")


if __name__ == "__main__":
    main()
