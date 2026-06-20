from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SEED = 20260620
ESM2_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AMINO_ACIDS)
AA_GROUPS = {
    "hydrophobic": set("AILMFWVY"),
    "polar": set("STNQCY"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "small": set("ACDGNPSTV"),
    "special": set("CGP"),
}

SPLIT_VALUES = {
    "train": "train",
    "training": "train",
    "tr": "train",
    "dev": "val",
    "valid": "val",
    "validation": "val",
    "val": "val",
    "holdout": "test",
    "heldout": "test",
    "test": "test",
    "testing": "test",
}

LABEL_ALIASES = ["effect", "effect_label", "label", "target", "class", "ppi_effect", "switch_effect"]
MOD_ACC_ALIASES = [
    "uniprot",
    "modified_uniprot",
    "mod_uniprot",
    "protein_uniprot",
    "substrate_uniprot",
    "modified_protein_uniprot",
    "modified_protein",
    "source_uniprot",
]
PARTNER_ACC_ALIASES = [
    "int_uniprot",
    "partner_uniprot",
    "interactor_uniprot",
    "int_protein",
    "interacting_uniprot",
    "target_uniprot",
    "partner_protein",
]
MOD_SEQ_ALIASES = [
    "modified_sequence",
    "mod_sequence",
    "protein_sequence",
    "substrate_sequence",
    "uniprot_sequence",
    "sequence",
    "modified_protein_sequence",
]
PARTNER_SEQ_ALIASES = [
    "partner_sequence",
    "interactor_sequence",
    "int_sequence",
    "int_protein_sequence",
    "partner_protein_sequence",
]
WINDOW_ALIASES = [
    "sequence_window_15",
    "ptm_window_15",
    "site_window_15",
    "window_15",
    "sequence window(-15,+15)",
    "sequence_window",
    "ptm_window",
    "site_window",
    "window",
    "sequence window(-5,+5)",
]
PTM_ALIASES = ["ptm", "modification", "modification_type", "ptm_type"]
AA_ALIASES = ["aa", "residue", "site_residue", "modified_residue"]
SITE_ALIASES = ["site", "position", "ptm_site", "site_position", "modified_position"]
ORGANISM_ALIASES = ["organism", "species", "taxon"]

ID_ALIASES = [
    "row_id",
    "source_row_id",
    "_source_row_id",
    "record_id",
    "example_id",
    "evidence_id",
    "benchmark_id",
    "id",
    "index",
    "original_index",
]


@dataclass
class FeatureDataset:
    rows: pd.DataFrame
    labels: np.ndarray
    splits: np.ndarray
    embeddings: np.ndarray
    mechanism: pd.DataFrame
    backend: str
    notes: list[str]
    metadata: dict


def normalized_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def find_column(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    lookup = {normalized_name(col): col for col in df.columns}
    for alias in aliases:
        col = lookup.get(normalized_name(alias))
        if col is not None:
            return col
    return None


def clean_sequence(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    seq = re.sub(r"[^A-Za-z]", "", str(value).strip()).upper()
    return "".join(ch if ch in AA_SET else "X" for ch in seq)


def stable_digest(text: str, size: int = 16) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=size).hexdigest()


def stable_bucket(text: str, buckets: int) -> tuple[int, float]:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little", signed=False)
    sign = 1.0 if (value >> 63) == 0 else -1.0
    return value % buckets, sign


def normalize_label(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    lowered = text.lower()
    if lowered == "enhance":
        return "Enhance"
    if lowered == "inhibit":
        return "Inhibit"
    if lowered == "induce":
        return "Induce"
    return text[:1].upper() + text[1:]


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table type: {path}")


def table_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    suffixes = {".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".parquet"}
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            lowered_parts = {part.lower() for part in path.parts}
            if {"feature_cache", "features", "embeddings"} & lowered_parts:
                continue
            files.append(path)
    return sorted(files)


def primary_split_files(project_dir: Path) -> dict[str, Path]:
    split_dir = project_dir / "data" / "processed" / "splits"
    files = table_files(split_dir)
    controls = ("degree_only_features", "shuffled_labels", "leakage_audit")
    primary: dict[str, Path] = {}
    for path in files:
        stem = path.stem
        if any(control in stem for control in controls):
            continue
        if path.suffix.lower() not in {".csv", ".tsv", ".parquet", ".json", ".jsonl", ".ndjson"}:
            continue
        primary[stem] = path
    return dict(sorted(primary.items()))


def add_source_ids(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["_source_row_id"] = np.arange(len(result), dtype=np.int64)
    if "source_row_id" not in result.columns:
        result["source_row_id"] = result["_source_row_id"]
    if "row_id" not in result.columns:
        result["row_id"] = result["_source_row_id"]
    return result


def has_model_columns(df: pd.DataFrame) -> bool:
    return (
        find_column(df, LABEL_ALIASES) is not None
        and find_column(df, MOD_ACC_ALIASES) is not None
        and find_column(df, PARTNER_ACC_ALIASES) is not None
    )


def source_candidates(project_dir: Path, notes: list[str]) -> list[tuple[Path, pd.DataFrame]]:
    candidates: list[tuple[Path, pd.DataFrame]] = []
    for path in table_files(project_dir / "data" / "processed") + table_files(project_dir / "data" / "raw"):
        try:
            df = read_table(path)
        except Exception as exc:
            notes.append(f"Skipped source candidate {path}: {exc}")
            continue
        if find_column(df, LABEL_ALIASES) is not None:
            candidates.append((path, add_source_ids(df)))
    candidates.sort(key=lambda item: (len(item[1]), len(item[1].columns)), reverse=True)
    return candidates


def split_from_filename(path: Path) -> str | None:
    tokens = re.split(r"[^A-Za-z0-9]+", path.stem.lower())
    for token in tokens:
        split = SPLIT_VALUES.get(token)
        if split:
            return split
    return None


def find_split_column(df: pd.DataFrame) -> str | None:
    for alias in ["split", "set", "subset", "partition", "fold"]:
        col = find_column(df, [alias])
        if col is None:
            continue
        values = {
            SPLIT_VALUES.get(str(value).strip().lower())
            for value in df[col].dropna().unique().tolist()
        }
        values.discard(None)
        if values:
            return col
    return None


def merge_split_with_source(
    split_df: pd.DataFrame,
    split_path: Path,
    sources: list[tuple[Path, pd.DataFrame]],
    notes: list[str],
) -> pd.DataFrame:
    if has_model_columns(split_df):
        return split_df.copy()

    for source_path, source_df in sources:
        common_ids = [col for col in ID_ALIASES if col in split_df.columns and col in source_df.columns]
        for id_col in common_ids:
            merged = split_df.merge(source_df, on=id_col, how="left", suffixes=("_split", ""))
            if has_model_columns(merged) and merged[find_column(merged, LABEL_ALIASES)].notna().any():
                notes.append(f"Merged {split_path.name} to {source_path.name} by {id_col}.")
                return merged

        if split_df.shape[1] == 1:
            only_col = split_df.columns[0]
            numeric_ids = pd.to_numeric(split_df[only_col], errors="coerce")
            if numeric_ids.notna().all():
                keyed = source_df.copy()
                keyed["_source_row_id_numeric"] = pd.to_numeric(keyed["_source_row_id"], errors="coerce")
                probe = pd.DataFrame({"_source_row_id_numeric": numeric_ids.astype(np.int64)})
                merged = probe.merge(keyed, on="_source_row_id_numeric", how="left")
                if has_model_columns(merged) and merged[find_column(merged, LABEL_ALIASES)].notna().any():
                    notes.append(f"Merged one-column split {split_path.name} to {source_path.name} by source row index.")
                    return merged

        key_cols = [
            col
            for col in ["Uniprot", "Int_uniprot", "PTM", "Site", "AA", "Effect", "PMID"]
            if col in split_df.columns and col in source_df.columns
        ]
        if len(key_cols) >= 3:
            merged = split_df.merge(source_df, on=key_cols, how="left", suffixes=("_split", ""))
            if has_model_columns(merged) and merged[find_column(merged, LABEL_ALIASES)].notna().any():
                notes.append(f"Merged {split_path.name} to {source_path.name} by keys: {', '.join(key_cols)}.")
                return merged

    raise RuntimeError(
        f"{split_path} looks like a split assignment file, but no source table could be merged to it."
    )


def available_split_strategies(project_dir: Path) -> list[str]:
    return list(primary_split_files(project_dir).keys())


def discover_auditor_splits(project_dir: Path, split_strategy: str = "random") -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    processed_dir = project_dir / "data" / "processed"
    primary = primary_split_files(project_dir)
    if split_strategy == "all":
        raise ValueError("discover_auditor_splits expects one split strategy, not 'all'.")
    if primary:
        if split_strategy not in primary:
            raise RuntimeError(
                f"Split strategy '{split_strategy}' was not found. Available strategies: {', '.join(primary)}"
            )
        files = [primary[split_strategy]]
    else:
        files = table_files(processed_dir)
    if not files:
        raise RuntimeError(
            "No auditor split files found under data/processed. Expected train/val/test tables "
            "or a table with a split column."
        )

    sources: list[tuple[Path, pd.DataFrame]] | None = None
    frames: list[pd.DataFrame] = []
    split_sources: list[str] = []

    for path in files:
        try:
            df = read_table(path)
        except Exception as exc:
            notes.append(f"Skipped split candidate {path}: {exc}")
            continue
        if df.empty:
            continue

        split_col = find_split_column(df)
        if split_col is not None:
            if has_model_columns(df):
                full_df = df.copy()
            else:
                if sources is None:
                    sources = source_candidates(project_dir, notes)
                full_df = merge_split_with_source(df, path, sources, notes)
            normalized = full_df[split_col].map(lambda value: SPLIT_VALUES.get(str(value).strip().lower()))
            full_df = full_df.loc[normalized.notna()].copy()
            full_df["_split"] = normalized.loc[normalized.notna()].to_numpy()
            full_df["_benchmark_split"] = split_strategy
            frames.append(full_df)
            split_sources.append(str(path.relative_to(project_dir)))
            continue

        split_name = split_from_filename(path)
        if split_name is not None:
            if has_model_columns(df):
                full_df = df.copy()
            else:
                if sources is None:
                    sources = source_candidates(project_dir, notes)
                full_df = merge_split_with_source(df, path, sources, notes)
            full_df = full_df.copy()
            full_df["_split"] = split_name
            full_df["_benchmark_split"] = split_strategy
            frames.append(full_df)
            split_sources.append(str(path.relative_to(project_dir)))

    if not frames:
        raise RuntimeError(
            "No auditor split files could be recognized. Use filenames containing train/val/test "
            "or include a split column with train/val/test values."
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    label_col = find_column(combined, LABEL_ALIASES)
    if label_col is None:
        raise RuntimeError("Recognized split files, but no label/effect column was available after merging.")
    combined[label_col] = combined[label_col].map(normalize_label)
    combined = combined[combined[label_col] != ""].copy()
    split_counts = combined["_split"].value_counts().to_dict()
    if split_counts.get("train", 0) == 0:
        raise RuntimeError(f"Auditor splits did not include any train rows. Found splits: {split_counts}")
    if len(split_counts) < 2:
        raise RuntimeError(f"Need at least train plus one evaluation split. Found splits: {split_counts}")

    notes.append(f"Benchmark split strategy: {split_strategy}.")
    notes.append(f"Consumed auditor split files: {', '.join(split_sources)}.")
    notes.append(f"Split sizes: {json.dumps(split_counts, sort_keys=True)}.")
    return combined.reset_index(drop=True), notes


def parse_fasta(text: str) -> dict[str, str]:
    sequences: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            parts = line[1:].split("|")
            if len(parts) >= 2:
                current = parts[1]
            else:
                current = line[1:].split()[0]
            sequences.setdefault(current, [])
        elif current:
            sequences[current].append(line)
    return {acc: clean_sequence("".join(seq_parts)) for acc, seq_parts in sequences.items()}


def load_local_sequence_sources(project_dir: Path, notes: list[str]) -> dict[str, str]:
    sequences: dict[str, str] = {}
    json_path = project_dir / "data" / "raw" / "uniprot_sequences.json"
    if json_path.exists():
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            for acc, seq in raw.items():
                cleaned = clean_sequence(seq)
                if cleaned:
                    sequences[str(acc)] = cleaned
            notes.append(f"Loaded {len(sequences)} sequences from {json_path.relative_to(project_dir)}.")
        except Exception as exc:
            notes.append(f"Could not load {json_path.relative_to(project_dir)}: {exc}")

    fasta_path = project_dir / "data" / "raw" / "uniprot_ptmint_accessions.fasta"
    if fasta_path.exists():
        try:
            parsed = parse_fasta(fasta_path.read_text(encoding="utf-8"))
            before = len(sequences)
            sequences.update({acc: seq for acc, seq in parsed.items() if seq})
            notes.append(
                f"Loaded {len(sequences) - before} additional sequences from {fasta_path.relative_to(project_dir)}."
            )
        except Exception as exc:
            notes.append(f"Could not load {fasta_path.relative_to(project_dir)}: {exc}")
    return sequences


def load_sequence_cache(cache_path: Path) -> dict[str, str]:
    if not cache_path.exists():
        return {}
    df = pd.read_csv(cache_path)
    if "accession" not in df.columns or "sequence" not in df.columns:
        return {}
    return {
        str(row.accession): clean_sequence(row.sequence)
        for row in df.itertuples(index=False)
        if isinstance(row.accession, str) and isinstance(row.sequence, str)
    }


def save_sequence_cache(cache_path: Path, sequences: dict[str, str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"accession": acc, "sequence": seq, "length": len(seq)} for acc, seq in sorted(sequences.items())]
    pd.DataFrame(rows).to_csv(cache_path, index=False)


def fetch_uniprot_sequences(
    accessions: Iterable[str],
    cache_path: Path,
    project_dir: Path,
    allow_downloads: bool,
    notes: list[str],
) -> dict[str, str]:
    cached = load_local_sequence_sources(project_dir, notes)
    cached.update(load_sequence_cache(cache_path))
    wanted = sorted({str(acc).strip() for acc in accessions if str(acc).strip() and str(acc).lower() != "nan"})
    missing = [acc for acc in wanted if acc not in cached]
    if not missing or not allow_downloads:
        if missing and not allow_downloads:
            notes.append(f"Sequence downloads disabled; {len(missing)} accessions will use identifier fallback features.")
        return cached

    try:
        import requests
    except Exception as exc:
        notes.append(f"requests unavailable; {len(missing)} accessions will use identifier fallback features: {exc}")
        return cached

    fetched = 0
    failed = 0
    batch_size = 100
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        try:
            response = requests.get(
                "https://rest.uniprot.org/uniprotkb/accessions",
                params={"accessions": ",".join(batch), "format": "fasta"},
                timeout=60,
            )
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:120]}")
            parsed = parse_fasta(response.text)
            for acc, seq in parsed.items():
                if seq:
                    cached[acc] = seq
            fetched += len(parsed)
            unresolved = [acc for acc in batch if acc not in parsed]
            failed += len(unresolved)
        except Exception as exc:
            notes.append(f"UniProt batch fetch failed for {len(batch)} accessions: {exc}")
            failed += len(batch)
        time.sleep(0.1)

    save_sequence_cache(cache_path, cached)
    notes.append(f"UniProt sequence cache updated: fetched={fetched}, unresolved_or_failed={failed}.")
    return cached


def extract_site_window(sequence: str, site: object, fallback_window: str, flank: int = 15) -> str:
    seq = clean_sequence(sequence)
    numeric_site = pd.to_numeric(pd.Series([site]), errors="coerce").iloc[0]
    if seq and pd.notna(numeric_site):
        center = int(numeric_site) - 1
        if 0 <= center < len(seq):
            start = max(0, center - flank)
            end = min(len(seq), center + flank + 1)
            return seq[start:end]
    return clean_sequence(fallback_window)


def deterministic_sequence_vector(sequence: str, hash_dim: int = 256) -> np.ndarray:
    seq = clean_sequence(sequence)
    if not seq:
        seq = "EMPTY"
    length = max(len(seq), 1)

    aa_counts = np.array([seq.count(aa) for aa in AMINO_ACIDS], dtype=np.float32) / float(length)
    group_counts = np.array(
        [sum(1 for ch in seq if ch in residues) / float(length) for residues in AA_GROUPS.values()],
        dtype=np.float32,
    )
    unknown_fraction = sum(1 for ch in seq if ch not in AA_SET) / float(length)
    hydrophobic = group_counts[0]
    polar = group_counts[1]
    positive = group_counts[2]
    negative = group_counts[3]
    aromatic = sum(seq.count(ch) for ch in "FWY") / float(length)
    helix_proxy = sum(seq.count(ch) for ch in "ALMQEKR") / float(length)
    disorder_proxy = sum(seq.count(ch) for ch in "PGSQRKED") / float(length)
    stats = np.array(
        [
            math.log1p(length),
            math.sqrt(length),
            unknown_fraction,
            hydrophobic,
            polar,
            positive,
            negative,
            positive - negative,
            aromatic,
            seq.count("G") / float(length),
            seq.count("P") / float(length),
            seq.count("C") / float(length),
            helix_proxy,
            disorder_proxy,
        ],
        dtype=np.float32,
    )

    hashed = np.zeros(hash_dim, dtype=np.float32)
    for k in (1, 2, 3):
        if len(seq) < k:
            continue
        for idx in range(0, len(seq) - k + 1):
            kmer = f"{k}:{seq[idx:idx+k]}"
            bucket, sign = stable_bucket(kmer, hash_dim)
            hashed[bucket] += sign
    norm = np.linalg.norm(hashed)
    if norm > 0:
        hashed /= norm
    return np.concatenate([aa_counts, group_counts, stats, hashed]).astype(np.float32)


class EmbeddingCache:
    def __init__(
        self,
        cache_root: Path,
        backend: str,
        allow_cpu_esm: bool,
        notes: list[str],
    ) -> None:
        self.cache_root = cache_root
        self.requested_backend = backend
        self.allow_cpu_esm = allow_cpu_esm
        self.notes = notes
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.device = None
        self.memory: dict[tuple[str, str], np.ndarray] = {}
        self.backend = self._resolve_backend()
        self.dim = 480 if self.backend == "esm2_t12_35m" else len(deterministic_sequence_vector("ACD"))
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _resolve_backend(self) -> str:
        if self.requested_backend == "deterministic":
            self.notes.append("Using deterministic sequence features by request.")
            return "deterministic"
        if self.requested_backend not in {"auto", "esm2"}:
            raise ValueError(f"Unknown backend: {self.requested_backend}")

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:
            self.notes.append(f"ESM2 unavailable; falling back to deterministic features: {exc}")
            return "deterministic"

        has_cuda = torch.cuda.is_available()
        if not has_cuda and not self.allow_cpu_esm:
            self.notes.append("CUDA unavailable; falling back to deterministic sequence features.")
            return "deterministic"

        try:
            self.torch = torch
            self.device = torch.device("cuda" if has_cuda else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)
            self.model = AutoModel.from_pretrained(ESM2_MODEL_NAME).to(self.device)
            self.model.eval()
            self.notes.append(f"Using {ESM2_MODEL_NAME} on {self.device}.")
            return "esm2_t12_35m"
        except Exception as exc:
            self.notes.append(f"ESM2 model load/download failed; falling back to deterministic features: {exc}")
            self.model = None
            self.tokenizer = None
            self.torch = None
            self.device = None
            return "deterministic"

    def vector(self, role: str, sequence_or_id: str) -> np.ndarray:
        role_dir = self.cache_root / self.backend / role
        role_dir.mkdir(parents=True, exist_ok=True)
        key = stable_digest(f"{self.backend}|{role}|{sequence_or_id}")
        memory_key = (role, key)
        if memory_key in self.memory:
            return self.memory[memory_key]
        path = role_dir / f"{key}.npy"
        if path.exists():
            vector = np.load(path).astype(np.float32)
            self.memory[memory_key] = vector
            return vector

        if self.backend == "esm2_t12_35m":
            vector = self._esm_vector(sequence_or_id)
        else:
            vector = deterministic_sequence_vector(sequence_or_id)
        np.save(path, vector.astype(np.float32))
        vector = vector.astype(np.float32)
        self.memory[memory_key] = vector
        return vector

    def _esm_vector(self, sequence: str) -> np.ndarray:
        if self.model is None or self.tokenizer is None or self.torch is None:
            return deterministic_sequence_vector(sequence)
        seq = clean_sequence(sequence)
        if not seq:
            seq = "X"
        max_residues = 1000
        chunks = [seq[start : start + max_residues] for start in range(0, len(seq), max_residues)]
        chunk_vectors: list[np.ndarray] = []
        for chunk in chunks:
            encoded = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=max_residues + 2,
                return_special_tokens_mask=True,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            special = encoded.pop("special_tokens_mask")
            with self.torch.no_grad():
                output = self.model(**encoded)
            mask = encoded["attention_mask"].bool() & (~special.bool())
            token_embeddings = output.last_hidden_state[0][mask[0]]
            chunk_vectors.append(token_embeddings.mean(dim=0).detach().cpu().numpy().astype(np.float32))
        return np.mean(np.vstack(chunk_vectors), axis=0).astype(np.float32)


def mechanism_frame(rows: pd.DataFrame) -> pd.DataFrame:
    ptm_col = find_column(rows, PTM_ALIASES)
    aa_col = find_column(rows, AA_ALIASES)
    site_col = find_column(rows, SITE_ALIASES)
    organism_col = find_column(rows, ORGANISM_ALIASES)
    mod_acc_col = find_column(rows, MOD_ACC_ALIASES)
    partner_acc_col = find_column(rows, PARTNER_ACC_ALIASES)

    result = pd.DataFrame(index=rows.index)
    result["ptm"] = rows[ptm_col].fillna("unknown").astype(str) if ptm_col else "unknown"
    result["residue"] = rows[aa_col].fillna("unknown").astype(str).str.upper() if aa_col else "unknown"
    result["organism"] = rows[organism_col].fillna("unknown").astype(str) if organism_col else "unknown"

    site = pd.to_numeric(rows[site_col], errors="coerce") if site_col else pd.Series(np.nan, index=rows.index)
    result["site"] = site.fillna(-1).astype(float)
    result["site_log1p"] = np.log1p(site.clip(lower=0).fillna(0).astype(float))

    mod_lengths = rows["_mod_sequence"].map(len).astype(float)
    partner_lengths = rows["_partner_sequence"].map(len).astype(float)
    window_lengths = rows["_ptm_window"].map(len).astype(float)
    result["mod_length_log1p"] = np.log1p(mod_lengths)
    result["partner_length_log1p"] = np.log1p(partner_lengths)
    result["window_length"] = window_lengths
    result["site_fraction"] = np.where(mod_lengths > 0, site.fillna(0).astype(float) / np.maximum(mod_lengths, 1.0), 0.0)
    result["same_uniprot"] = (
        rows[mod_acc_col].fillna("").astype(str).to_numpy()
        == rows[partner_acc_col].fillna("").astype(str).to_numpy()
        if mod_acc_col and partner_acc_col
        else False
    )
    result["has_full_mod_sequence"] = rows["_mod_sequence"].map(bool).astype(bool)
    result["has_full_partner_sequence"] = rows["_partner_sequence"].map(bool).astype(bool)
    result["has_site_window"] = rows["_ptm_window"].map(bool).astype(bool)
    return result


def attach_sequences(
    rows: pd.DataFrame,
    project_dir: Path,
    cache_root: Path,
    allow_sequence_downloads: bool,
    notes: list[str],
) -> pd.DataFrame:
    result = rows.copy()
    mod_acc_col = find_column(result, MOD_ACC_ALIASES)
    partner_acc_col = find_column(result, PARTNER_ACC_ALIASES)
    mod_seq_col = find_column(result, MOD_SEQ_ALIASES)
    partner_seq_col = find_column(result, PARTNER_SEQ_ALIASES)
    site_col = find_column(result, SITE_ALIASES)
    window_col = find_column(result, WINDOW_ALIASES)

    if mod_acc_col is None or partner_acc_col is None:
        raise RuntimeError("Split rows must include modified and partner protein accession columns.")

    accessions = pd.concat([result[mod_acc_col], result[partner_acc_col]], ignore_index=True)
    sequence_cache = fetch_uniprot_sequences(
        accessions,
        cache_root / "uniprot_sequences.csv",
        project_dir,
        allow_downloads=allow_sequence_downloads,
        notes=notes,
    )

    def sequence_for(row: pd.Series, seq_col: str | None, acc_col: str) -> str:
        if seq_col is not None:
            seq = clean_sequence(row.get(seq_col))
            if seq:
                return seq
        acc = str(row.get(acc_col, "")).strip()
        return sequence_cache.get(acc, "")

    result["_mod_sequence"] = [sequence_for(row, mod_seq_col, mod_acc_col) for _, row in result.iterrows()]
    result["_partner_sequence"] = [sequence_for(row, partner_seq_col, partner_acc_col) for _, row in result.iterrows()]
    fallback_windows = result[window_col].fillna("").astype(str) if window_col else pd.Series("", index=result.index)
    sites = result[site_col] if site_col else pd.Series(np.nan, index=result.index)
    result["_ptm_window"] = [
        extract_site_window(seq, site, fallback)
        for seq, site, fallback in zip(result["_mod_sequence"], sites, fallback_windows, strict=False)
    ]

    missing_mod = int((result["_mod_sequence"].str.len() == 0).sum())
    missing_partner = int((result["_partner_sequence"].str.len() == 0).sum())
    notes.append(f"Rows missing full modified-protein sequence: {missing_mod}/{len(result)}.")
    notes.append(f"Rows missing full partner-protein sequence: {missing_partner}/{len(result)}.")
    return result


def build_feature_dataset(
    project_dir: Path,
    split_strategy: str = "random",
    backend: str = "auto",
    allow_cpu_esm: bool = False,
    allow_sequence_downloads: bool = True,
    force: bool = False,
    seed: int = SEED,
) -> FeatureDataset:
    random.seed(seed)
    np.random.seed(seed)
    notes: list[str] = [f"Feature seed: {seed}."]
    rows, split_notes = discover_auditor_splits(project_dir, split_strategy=split_strategy)
    notes.extend(split_notes)

    cache_root = project_dir / "data" / "processed" / "feature_cache"
    rows = attach_sequences(rows, project_dir, cache_root, allow_sequence_downloads, notes)
    embedder = EmbeddingCache(cache_root, backend, allow_cpu_esm, notes)

    label_col = find_column(rows, LABEL_ALIASES)
    mod_acc_col = find_column(rows, MOD_ACC_ALIASES)
    partner_acc_col = find_column(rows, PARTNER_ACC_ALIASES)
    if label_col is None or mod_acc_col is None or partner_acc_col is None:
        raise RuntimeError("Could not resolve required label/accession columns.")

    safe_split = re.sub(r"[^A-Za-z0-9_.-]+", "_", split_strategy)
    feature_npz = cache_root / f"row_features_{safe_split}_{embedder.backend}.npz"
    metadata_csv = cache_root / f"row_metadata_{safe_split}_{embedder.backend}.csv"
    if feature_npz.exists() and metadata_csv.exists() and not force:
        metadata = pd.read_csv(metadata_csv)
        if len(metadata) == len(rows) and metadata.get("_split", pd.Series(dtype=str)).tolist() == rows["_split"].tolist():
            loaded = np.load(feature_npz)
            notes.append(f"Loaded cached row feature matrix: {feature_npz}.")
            mechanism = mechanism_frame(rows)
            return FeatureDataset(
                rows=rows,
                labels=rows[label_col].to_numpy(),
                splits=rows["_split"].to_numpy(),
                embeddings=loaded["embeddings"].astype(np.float32),
                mechanism=mechanism,
                backend=embedder.backend,
                notes=notes,
                metadata={"feature_npz": str(feature_npz), "metadata_csv": str(metadata_csv)},
            )

    vectors: list[np.ndarray] = []
    for row in rows.itertuples(index=False):
        row_dict = row._asdict()
        mod_acc = str(row_dict.get(mod_acc_col, "")).strip()
        partner_acc = str(row_dict.get(partner_acc_col, "")).strip()
        mod_sequence = row_dict.get("_mod_sequence") or f"ID:{mod_acc}"
        partner_sequence = row_dict.get("_partner_sequence") or f"ID:{partner_acc}"
        ptm_window = row_dict.get("_ptm_window") or f"ID:{mod_acc}:{row_dict.get(find_column(rows, SITE_ALIASES) or '', '')}"

        mod_vec = embedder.vector("modified_protein", str(mod_sequence))
        partner_vec = embedder.vector("partner_protein", str(partner_sequence))
        window_vec = embedder.vector("ptm_site_window", str(ptm_window))
        pair_delta = np.abs(mod_vec - partner_vec)
        pair_product = mod_vec * partner_vec
        vectors.append(np.concatenate([mod_vec, partner_vec, window_vec, pair_delta, pair_product]).astype(np.float32))

    embeddings = np.vstack(vectors).astype(np.float32)
    mechanism = mechanism_frame(rows)
    feature_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feature_npz, embeddings=embeddings)
    metadata_cols = ["_benchmark_split", "_split", label_col, mod_acc_col, partner_acc_col]
    if "_split_row_id" in rows.columns:
        metadata_cols.insert(1, "_split_row_id")
    rows[metadata_cols].to_csv(metadata_csv, index=False)
    notes.append(f"Wrote row feature matrix: {feature_npz}.")
    notes.append(f"Embedding backend={embedder.backend}, base_dim={embedder.dim}, row_dim={embeddings.shape[1]}.")

    return FeatureDataset(
        rows=rows,
        labels=rows[label_col].to_numpy(),
        splits=rows["_split"].to_numpy(),
        embeddings=embeddings,
        mechanism=mechanism,
        backend=embedder.backend,
        notes=notes,
        metadata={"feature_npz": str(feature_npz), "metadata_csv": str(metadata_csv)},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cached SwitchPPI sequence features from auditor splits.")
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split-strategy", default="random")
    parser.add_argument("--backend", choices=["auto", "esm2", "deterministic"], default="auto")
    parser.add_argument("--allow-cpu-esm", action="store_true")
    parser.add_argument("--no-sequence-downloads", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    dataset = build_feature_dataset(
        project_dir=args.project_dir,
        split_strategy=args.split_strategy,
        backend=args.backend,
        allow_cpu_esm=args.allow_cpu_esm,
        allow_sequence_downloads=not args.no_sequence_downloads,
        force=args.force,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "rows": int(len(dataset.rows)),
                "backend": dataset.backend,
                "embedding_shape": list(dataset.embeddings.shape),
                "splits": pd.Series(dataset.splits).value_counts().to_dict(),
                "labels": pd.Series(dataset.labels).value_counts().to_dict(),
                "notes": dataset.notes,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
