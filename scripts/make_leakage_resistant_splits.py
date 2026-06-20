#!/usr/bin/env python3
"""Create leakage-resistant benchmark splits and audit artifacts.

This script deliberately lives outside ingestion and model code. It reads an
already-processed benchmark table, writes split datasets, and emits proofs that
the entities forbidden by each split do not cross train/validation/test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "splits"
DEFAULT_REPORT = ROOT / "results" / "tables" / "split_leakage_report.md"
DEFAULT_AUDIT_JSON = ROOT / "results" / "tables" / "split_leakage_audit.json"

PARTITIONS = ("train", "validation", "test")

COLUMN_ALIASES = {
    "protein_a": (
        "protein_a",
        "protein1",
        "protein_1",
        "protein_a_id",
        "protein1_id",
        "modified_uniprot",
        "uniprot",
        "uniprot_a",
        "uniprot1",
        "interactor_a",
        "bait",
        "source_protein",
        "source",
    ),
    "protein_b": (
        "protein_b",
        "protein2",
        "protein_2",
        "protein_b_id",
        "protein2_id",
        "partner_uniprot",
        "int_uniprot",
        "interactor_uniprot",
        "interacting_uniprot",
        "uniprot_b",
        "uniprot2",
        "interactor_b",
        "prey",
        "target_protein",
        "target",
        "binding_protein",
        "partner",
    ),
    "modified_protein": (
        "modified_protein",
        "modified_protein_id",
        "modified_uniprot",
        "protein_with_modification",
        "ptm_protein",
        "phosphoprotein",
        "uniprot",
        "substrate",
        "substrate_id",
        "protein_id",
    ),
    "site": (
        "site",
        "site_id",
        "mod_site",
        "modification_site",
        "modified_site",
        "ptm_site",
        "phosphosite",
        "residue_position",
        "position",
        "modified_residue",
        "residue",
    ),
    "pmid": (
        "pmid",
        "pubmed_id",
        "pubmed",
        "publication_id",
        "reference_pmid",
        "source_pmid",
        "evidence_pmid",
    ),
    "label": (
        "label",
        "y",
        "class",
        "target",
        "is_positive",
        "interaction",
        "response",
        "effect",
        "effect_label",
        "switch_label",
        "outcome",
    ),
}

POSITIVE_LABELS = {"1", "true", "t", "yes", "y", "positive", "pos", "interacting"}
NEGATIVE_LABELS = {"0", "false", "f", "no", "n", "negative", "neg", "non_interacting"}
TOKEN_SPLIT_RE = re.compile(r"\s*(?:[;,|]|\band\b)\s*", re.IGNORECASE)


@dataclass
class SplitResult:
    name: str
    status: str
    reason: str | None = None
    constraint: str | None = None
    split_rows: list[dict[str, str]] = field(default_factory=list)
    audit: dict = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)


@dataclass
class SplitUnit:
    row_indices: list[int]
    labels: Counter

    @property
    def size(self) -> int:
        return len(self.row_indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate leakage-resistant benchmark splits and audit files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help=(
            "Benchmark table. If omitted, common files under data/processed "
            "are auto-detected first, with data/raw used only as a fallback."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.10)
    parser.add_argument("--protein-a-column")
    parser.add_argument("--protein-b-column")
    parser.add_argument("--modified-protein-column")
    parser.add_argument("--site-column")
    parser.add_argument("--pmid-column")
    parser.add_argument("--label-column")
    parser.add_argument(
        "--id-column",
        help="Optional stable row identifier column. A generated ID is used otherwise.",
    )
    parser.add_argument(
        "--write-audits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write shuffled-label and degree-only audit-preparation files.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode()).hexdigest()
    return int(digest[:16], 16)


def normalize_cell(value: object) -> str:
    return str(value).strip() if value is not None else ""


def tokens(value: object) -> list[str]:
    raw = normalize_cell(value)
    if not raw:
        return []
    pieces = [piece.strip() for piece in TOKEN_SPLIT_RE.split(raw)]
    return [piece for piece in pieces if piece and piece.lower() not in {"na", "nan", "none", "null"}]


def normalize_entity(value: object) -> str:
    return normalize_cell(value).upper()


def read_table(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        delimiter = "\t" if suffix in {".tsv", ".txt"} else ","
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            if suffix == ".txt":
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
                    delimiter = dialect.delimiter
                except csv.Error:
                    delimiter = "\t"
            reader = csv.DictReader(handle, delimiter=delimiter)
            return [dict(row) for row in reader], list(reader.fieldnames or [])
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
        return rows, fieldnames
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            rows = payload.get("rows") or payload.get("data") or payload.get("records")
        else:
            rows = payload
        if not isinstance(rows, list):
            raise ValueError(f"{path} does not contain a list of records")
        fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
        return [dict(row) for row in rows], fieldnames
    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Reading parquet requires pandas and a parquet engine.") from exc
        frame = pd.read_parquet(path)
        return frame.fillna("").astype(str).to_dict("records"), list(frame.columns)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def auto_detect_input() -> Path | None:
    suffixes = {".csv", ".tsv", ".txt", ".jsonl", ".ndjson", ".json", ".parquet"}
    preferred_stems = (
        "benchmark",
        "benchmarks",
        "interactions",
        "switchppi",
        "ptmint_experimental_evidence",
        "ptmint_normalized",
        "dataset",
        "training",
        "examples",
    )

    for directory in (ROOT / "data" / "processed", ROOT / "data" / "raw"):
        if not directory.exists():
            continue
        candidates: list[Path] = []
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            lowered_stem = path.stem.lower()
            if any(
                marker in lowered_stem
                for marker in (
                    "summary",
                    "metadata",
                    "invalid",
                    "audit",
                    "leakage",
                    "shuffled",
                    "degree_only",
                )
            ):
                continue
            if DEFAULT_OUT_DIR in path.parents:
                continue
            candidates.append(path)
        if candidates:
            candidates.sort(key=lambda p: (p.stem.lower() not in preferred_stems, len(p.parts), str(p)))
            return candidates[0]
    return None


def resolve_column(
    fieldnames: list[str],
    explicit: str | None,
    alias_key: str,
) -> tuple[str | None, str | None]:
    if explicit:
        if explicit in fieldnames:
            return explicit, None
        return None, f"explicit column {explicit!r} was not found"

    by_lower = {field.lower(): field for field in fieldnames}
    for alias in COLUMN_ALIASES[alias_key]:
        if alias.lower() in by_lower:
            return by_lower[alias.lower()], None
    return None, f"no {alias_key.replace('_', ' ')} column detected"


def resolve_columns(args: argparse.Namespace, fieldnames: list[str]) -> dict[str, str | None]:
    resolutions = {
        "protein_a": resolve_column(fieldnames, args.protein_a_column, "protein_a"),
        "protein_b": resolve_column(fieldnames, args.protein_b_column, "protein_b"),
        "modified_protein": resolve_column(
            fieldnames, args.modified_protein_column, "modified_protein"
        ),
        "site": resolve_column(fieldnames, args.site_column, "site"),
        "pmid": resolve_column(fieldnames, args.pmid_column, "pmid"),
        "label": resolve_column(fieldnames, args.label_column, "label"),
    }
    return {key: value for key, (value, _reason) in resolutions.items()}


def column_resolution_notes(args: argparse.Namespace, fieldnames: list[str]) -> dict[str, dict[str, str | None]]:
    explicit_by_key = {
        "protein_a": args.protein_a_column,
        "protein_b": args.protein_b_column,
        "modified_protein": args.modified_protein_column,
        "site": args.site_column,
        "pmid": args.pmid_column,
        "label": args.label_column,
    }
    notes = {}
    for key, explicit in explicit_by_key.items():
        column, reason = resolve_column(fieldnames, explicit, key)
        notes[key] = {"column": column, "note": reason}
    return notes


def row_id(row: dict[str, str], index: int, id_column: str | None) -> str:
    if id_column and normalize_cell(row.get(id_column)):
        return normalize_cell(row.get(id_column))
    return str(index)


def pair_key(row: dict[str, str], protein_a: str, protein_b: str) -> str:
    left = normalize_entity(row.get(protein_a))
    right = normalize_entity(row.get(protein_b))
    return "||".join(sorted((left, right)))


def pair_entities(protein_a: str, protein_b: str) -> Callable[[dict[str, str]], set[str]]:
    def _entities(row: dict[str, str]) -> set[str]:
        key = pair_key(row, protein_a, protein_b)
        return {key} if key != "||" else set()

    return _entities


def modified_protein_entities(column: str) -> Callable[[dict[str, str]], set[str]]:
    def _entities(row: dict[str, str]) -> set[str]:
        return {normalize_entity(value) for value in tokens(row.get(column))}

    return _entities


def site_entities(site_column: str, modified_protein_column: str | None) -> Callable[[dict[str, str]], set[str]]:
    def _entities(row: dict[str, str]) -> set[str]:
        site_values = [normalize_entity(value) for value in tokens(row.get(site_column))]
        if not site_values:
            return set()
        if modified_protein_column:
            proteins = [normalize_entity(value) for value in tokens(row.get(modified_protein_column))]
            if proteins:
                return {f"{protein}:{site}" for protein in proteins for site in site_values}
        return set(site_values)

    return _entities


def pmid_entities(column: str) -> Callable[[dict[str, str]], set[str]]:
    def _entities(row: dict[str, str]) -> set[str]:
        return {normalize_entity(value) for value in tokens(row.get(column))}

    return _entities


def label_value(row: dict[str, str], label_column: str | None) -> str:
    if not label_column:
        return "__unlabeled__"
    value = normalize_cell(row.get(label_column))
    return value if value else "__missing_label__"


def build_row_units(rows: list[dict[str, str]], label_column: str | None) -> list[SplitUnit]:
    return [
        SplitUnit(row_indices=[index], labels=Counter({label_value(row, label_column): 1}))
        for index, row in enumerate(rows)
    ]


def build_component_units(
    rows: list[dict[str, str]],
    entity_fn: Callable[[dict[str, str]], set[str]],
    label_column: str | None,
) -> tuple[list[SplitUnit], dict]:
    row_to_entities: list[set[str]] = [entity_fn(row) for row in rows]
    missing_rows = [index for index, entities in enumerate(row_to_entities) if not entities]
    entity_to_rows: dict[str, list[int]] = defaultdict(list)
    for index, entities in enumerate(row_to_entities):
        for entity in entities:
            entity_to_rows[entity].append(index)

    visited_rows: set[int] = set()
    units: list[SplitUnit] = []
    for start in range(len(rows)):
        if start in visited_rows:
            continue
        queue = deque([start])
        visited_rows.add(start)
        component_rows: list[int] = []
        component_entities: set[str] = set()
        while queue:
            row_index = queue.popleft()
            component_rows.append(row_index)
            for entity in row_to_entities[row_index]:
                if entity in component_entities:
                    continue
                component_entities.add(entity)
                for neighbor in entity_to_rows[entity]:
                    if neighbor not in visited_rows:
                        visited_rows.add(neighbor)
                        queue.append(neighbor)
        labels = Counter(label_value(rows[index], label_column) for index in component_rows)
        units.append(SplitUnit(row_indices=component_rows, labels=labels))

    metadata = {
        "entity_count": len(entity_to_rows),
        "missing_entity_rows": len(missing_rows),
        "component_count": len(units),
        "largest_component_rows": max((unit.size for unit in units), default=0),
    }
    return units, metadata


def target_counts(row_count: int, validation_size: float, test_size: float) -> dict[str, int]:
    if not 0 <= validation_size < 1 or not 0 <= test_size < 1:
        raise ValueError("validation-size and test-size must be in [0, 1)")
    if validation_size + test_size >= 1:
        raise ValueError("validation-size + test-size must be less than 1")
    test = int(round(row_count * test_size))
    validation = int(round(row_count * validation_size))
    if row_count >= 2 and test_size > 0:
        test = max(1, test)
    if row_count >= 3 and validation_size > 0:
        validation = max(1, validation)
    if test + validation >= row_count:
        validation = max(0, min(validation, row_count - 2))
        test = max(1 if row_count >= 2 else 0, min(test, row_count - validation - 1))
    train = row_count - validation - test
    return {"train": train, "validation": validation, "test": test}


def score_assignment(
    counts: dict[str, int],
    label_counts: dict[str, Counter],
    unit: SplitUnit,
    partition: str,
    targets: dict[str, int],
    total_labels: Counter,
) -> tuple[float, int]:
    projected_counts = dict(counts)
    projected_counts[partition] += unit.size

    target = max(targets[partition], 1)
    overshoot = max(0, projected_counts[partition] - targets[partition])
    size_score = projected_counts[partition] / target
    if overshoot:
        size_score += 10 * (overshoot / target)

    label_score = 0.0
    if len(total_labels) > 1:
        total_rows = sum(total_labels.values())
        for label, total_count in total_labels.items():
            global_rate = total_count / total_rows
            projected_label_count = label_counts[partition][label] + unit.labels[label]
            if projected_counts[partition]:
                part_rate = projected_label_count / projected_counts[partition]
                label_score += (part_rate - global_rate) ** 2

    return size_score + (0.25 * label_score), overshoot


def assign_units(
    units: list[SplitUnit],
    row_count: int,
    label_column: str | None,
    validation_size: float,
    test_size: float,
    seed: int,
) -> tuple[dict[int, str], dict]:
    if row_count == 0:
        raise ValueError("cannot split zero rows")
    targets = target_counts(row_count, validation_size, test_size)
    if targets["test"] > 0 and len(units) < 2:
        raise ValueError("fewer than two independent units; cannot create train/test split")

    rng = random.Random(seed)
    shuffled_units = list(units)
    rng.shuffle(shuffled_units)
    shuffled_units.sort(key=lambda unit: (unit.size, sum(v > 0 for v in unit.labels.values())), reverse=True)

    counts = {partition: 0 for partition in PARTITIONS}
    label_counts = {partition: Counter() for partition in PARTITIONS}
    assignments: dict[int, str] = {}
    total_labels = Counter()
    for unit in units:
        total_labels.update(unit.labels)

    mandatory = [partition for partition in PARTITIONS if targets[partition] > 0]
    remaining_units = len(shuffled_units)
    for unit in shuffled_units:
        remaining_units -= 1
        empty_mandatory = [partition for partition in mandatory if counts[partition] == 0]
        if empty_mandatory and remaining_units < len(empty_mandatory):
            candidate_partitions = empty_mandatory
        else:
            candidate_partitions = [partition for partition in PARTITIONS if targets[partition] > 0]

        candidate_partitions = list(candidate_partitions)
        rng.shuffle(candidate_partitions)
        partition = min(
            candidate_partitions,
            key=lambda part: score_assignment(counts, label_counts, unit, part, targets, total_labels),
        )
        for row_index in unit.row_indices:
            assignments[row_index] = partition
        counts[partition] += unit.size
        label_counts[partition].update(unit.labels)

    metadata = {
        "target_counts": targets,
        "actual_counts": counts,
        "label_counts": {partition: dict(counter) for partition, counter in label_counts.items()},
        "unit_count": len(units),
        "stratified_by_label": bool(label_column),
    }
    return assignments, metadata


def attach_split(
    rows: list[dict[str, str]],
    assignments: dict[int, str],
    id_column: str | None,
) -> list[dict[str, str]]:
    split_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        output = dict(row)
        output["_split_row_id"] = row_id(row, index, id_column)
        output["_split"] = assignments[index]
        split_rows.append(output)
    return split_rows


def partition_counts(split_rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row["_split"] for row in split_rows)
    return {partition: counts.get(partition, 0) for partition in PARTITIONS}


def label_distribution(split_rows: list[dict[str, str]], label_column: str | None) -> dict[str, dict[str, int]]:
    if not label_column:
        return {}
    distribution: dict[str, Counter] = {partition: Counter() for partition in PARTITIONS}
    for row in split_rows:
        distribution[row["_split"]][label_value(row, label_column)] += 1
    return {partition: dict(counter) for partition, counter in distribution.items()}


def leakage_audit(
    split_rows: list[dict[str, str]],
    entity_fn: Callable[[dict[str, str]], set[str]] | None,
    label_column: str | None,
) -> dict:
    audit = {
        "partition_counts": partition_counts(split_rows),
        "label_distribution": label_distribution(split_rows, label_column),
        "forbidden_entity_overlap_count": None,
        "train_test_overlap_count": None,
        "passed": True,
        "examples": [],
    }
    if entity_fn is None:
        return audit

    entity_partitions: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        for entity in entity_fn(row):
            entity_partitions[entity].add(row["_split"])
    overlaps = {
        entity: sorted(partitions)
        for entity, partitions in entity_partitions.items()
        if len(partitions) > 1
    }
    train_test_overlaps = {
        entity: partitions
        for entity, partitions in overlaps.items()
        if "train" in partitions and "test" in partitions
    }
    audit.update(
        {
            "forbidden_entity_count": len(entity_partitions),
            "forbidden_entity_overlap_count": len(overlaps),
            "train_test_overlap_count": len(train_test_overlaps),
            "passed": len(overlaps) == 0,
            "examples": [
                {"entity": entity, "partitions": partitions}
                for entity, partitions in list(overlaps.items())[:10]
            ],
        }
    )
    return audit


def is_positive_label(value: str) -> bool | None:
    normalized = normalize_cell(value).lower()
    if normalized in POSITIVE_LABELS:
        return True
    if normalized in NEGATIVE_LABELS:
        return False
    return None


def prepare_shuffled_labels(
    split_rows: list[dict[str, str]],
    label_column: str,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    by_partition: dict[str, list[int]] = defaultdict(list)
    shuffled_rows = [dict(row) for row in split_rows]
    for index, row in enumerate(shuffled_rows):
        by_partition[row["_split"]].append(index)
    for partition, indices in by_partition.items():
        labels = [normalize_cell(shuffled_rows[index].get(label_column)) for index in indices]
        rng.shuffle(labels)
        for index, label in zip(indices, labels):
            shuffled_rows[index][label_column] = label
            shuffled_rows[index]["_audit_kind"] = "shuffled_label_within_split"
    return shuffled_rows


def train_degree_maps(
    split_rows: list[dict[str, str]],
    protein_a: str,
    protein_b: str,
    label_column: str | None,
) -> tuple[Counter, set[str], str]:
    train_rows = [row for row in split_rows if row["_split"] == "train"]
    positive_rows: list[dict[str, str]] = []
    if label_column:
        for row in train_rows:
            positive = is_positive_label(normalize_cell(row.get(label_column)))
            if positive is True:
                positive_rows.append(row)
    source_rows = positive_rows if positive_rows else train_rows
    source = "train_positive_rows" if positive_rows else "all_train_rows"

    degrees: Counter = Counter()
    seen_pairs: set[str] = set()
    for row in source_rows:
        left = normalize_entity(row.get(protein_a))
        right = normalize_entity(row.get(protein_b))
        if not left or not right:
            continue
        degrees[left] += 1
        degrees[right] += 1
        seen_pairs.add("||".join(sorted((left, right))))
    return degrees, seen_pairs, source


def prepare_degree_only_features(
    split_rows: list[dict[str, str]],
    protein_a: str,
    protein_b: str,
    label_column: str | None,
) -> tuple[list[dict[str, object]], str]:
    degrees, seen_pairs, source = train_degree_maps(split_rows, protein_a, protein_b, label_column)
    rows: list[dict[str, object]] = []
    for row in split_rows:
        left = normalize_entity(row.get(protein_a))
        right = normalize_entity(row.get(protein_b))
        degree_left = int(degrees.get(left, 0))
        degree_right = int(degrees.get(right, 0))
        pair = "||".join(sorted((left, right)))
        output = {
            "_split_row_id": row["_split_row_id"],
            "_split": row["_split"],
            "protein_a": left,
            "protein_b": right,
            "train_degree_a": degree_left,
            "train_degree_b": degree_right,
            "train_degree_sum": degree_left + degree_right,
            "train_degree_abs_diff": abs(degree_left - degree_right),
            "train_degree_min": min(degree_left, degree_right),
            "train_degree_max": max(degree_left, degree_right),
            "train_degree_product": degree_left * degree_right,
            "either_protein_seen_in_train": int(degree_left > 0 or degree_right > 0),
            "both_proteins_seen_in_train": int(degree_left > 0 and degree_right > 0),
            "pair_seen_in_train": int(pair in seen_pairs),
        }
        if label_column:
            output[label_column] = normalize_cell(row.get(label_column))
        rows.append(output)
    return rows, source


def output_fieldnames(fieldnames: list[str]) -> list[str]:
    names = list(fieldnames)
    for extra in ("_split_row_id", "_split"):
        if extra not in names:
            names.append(extra)
    return names


def write_split_artifacts(
    result: SplitResult,
    fieldnames: list[str],
    columns: dict[str, str | None],
    args: argparse.Namespace,
) -> None:
    if result.status != "created":
        return
    split_path = args.out_dir / f"{result.name}.csv"
    write_csv(split_path, result.split_rows, output_fieldnames(fieldnames))
    result.files["split"] = str(split_path)

    audit_path = args.out_dir / f"{result.name}_leakage_audit.json"
    write_json(audit_path, result.audit)
    result.files["leakage_audit"] = str(audit_path)

    if not args.write_audits:
        return

    label_column = columns["label"]
    if label_column:
        shuffled_rows = prepare_shuffled_labels(
            result.split_rows,
            label_column,
            stable_seed(args.seed, result.name, "shuffled-label"),
        )
        shuffled_path = args.out_dir / f"{result.name}_shuffled_labels.csv"
        shuffled_fieldnames = output_fieldnames(fieldnames)
        if "_audit_kind" not in shuffled_fieldnames:
            shuffled_fieldnames.append("_audit_kind")
        write_csv(shuffled_path, shuffled_rows, shuffled_fieldnames)
        result.files["shuffled_labels"] = str(shuffled_path)

    if columns["protein_a"] and columns["protein_b"]:
        degree_rows, source = prepare_degree_only_features(
            result.split_rows,
            columns["protein_a"],
            columns["protein_b"],
            label_column,
        )
        degree_path = args.out_dir / f"{result.name}_degree_only_features.csv"
        degree_fieldnames = [
            "_split_row_id",
            "_split",
            "protein_a",
            "protein_b",
            "train_degree_a",
            "train_degree_b",
            "train_degree_sum",
            "train_degree_abs_diff",
            "train_degree_min",
            "train_degree_max",
            "train_degree_product",
            "either_protein_seen_in_train",
            "both_proteins_seen_in_train",
            "pair_seen_in_train",
        ]
        if label_column:
            degree_fieldnames.append(label_column)
        write_csv(degree_path, degree_rows, degree_fieldnames)
        result.files["degree_only_features"] = str(degree_path)
        result.audit["degree_only_feature_source"] = source
        write_json(Path(result.files["leakage_audit"]), result.audit)


def make_split(
    name: str,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    id_column: str | None,
    label_column: str | None,
    validation_size: float,
    test_size: float,
    seed: int,
    entity_fn: Callable[[dict[str, str]], set[str]] | None = None,
    constraint: str | None = None,
) -> SplitResult:
    result = SplitResult(name=name, status="skipped", constraint=constraint)
    try:
        if entity_fn is None:
            units = build_row_units(rows, label_column)
            unit_metadata = {"unit_count": len(units)}
        else:
            units, unit_metadata = build_component_units(rows, entity_fn, label_column)
        assignments, assignment_metadata = assign_units(
            units,
            len(rows),
            label_column,
            validation_size,
            test_size,
            seed,
        )
        result.split_rows = attach_split(rows, assignments, id_column)
        result.audit = leakage_audit(result.split_rows, entity_fn, label_column)
        result.audit.update(
            {
                "split_name": name,
                "constraint": constraint,
                "generated_at": utc_now(),
                "unit_metadata": unit_metadata,
                "assignment_metadata": assignment_metadata,
            }
        )
        result.status = "created" if result.audit["passed"] else "failed"
        if not result.audit["passed"]:
            result.reason = "leakage audit failed"
    except Exception as exc:  # Keep one impossible split from blocking the rest.
        result.status = "skipped"
        result.reason = str(exc)
    return result


def write_report(
    path: Path,
    input_path: Path | None,
    rows: list[dict[str, str]] | None,
    fieldnames: list[str] | None,
    column_notes: dict[str, dict[str, str | None]] | None,
    results: list[SplitResult],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Split Leakage Report",
        "",
        f"Generated: {utc_now()}",
        f"Seed: {args.seed}",
        "",
    ]
    if input_path is None:
        lines.extend(
            [
                "## Input",
                "",
                "No benchmark table was found under `data/processed` or `data/raw`, and no `--input` was provided.",
                "No split files were materialized in this run.",
                "",
                "Expected input formats: CSV, TSV, JSONL, JSON records, or Parquet with optional pandas support.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Input",
                "",
                f"Input table: `{input_path}`",
                f"Rows: {len(rows or [])}",
                f"Columns: {', '.join(fieldnames or [])}",
                "",
                "## Column Resolution",
                "",
                "| Entity | Column | Note |",
                "| --- | --- | --- |",
            ]
        )
        for key, note in (column_notes or {}).items():
            lines.append(
                f"| {key} | `{note.get('column') or ''}` | {note.get('note') or 'ok'} |"
            )
        lines.extend(["", "## Split Audits", "", "| Split | Status | Forbidden entity | Train | Validation | Test | Entity overlaps | Train/test overlaps | Audit prep |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"])
        for result in results:
            if result.status == "created":
                counts = result.audit.get("partition_counts", {})
                overlap = result.audit.get("forbidden_entity_overlap_count")
                train_test = result.audit.get("train_test_overlap_count")
                prep = []
                if "shuffled_labels" in result.files:
                    prep.append("shuffled-label")
                if "degree_only_features" in result.files:
                    prep.append("degree-only")
                lines.append(
                    "| {name} | {status} | {constraint} | {train} | {validation} | {test} | {overlap} | {train_test} | {prep} |".format(
                        name=result.name,
                        status="passed",
                        constraint=result.constraint or "none",
                        train=counts.get("train", 0),
                        validation=counts.get("validation", 0),
                        test=counts.get("test", 0),
                        overlap=overlap if overlap is not None else "n/a",
                        train_test=train_test if train_test is not None else "n/a",
                        prep=", ".join(prep) if prep else "none",
                    )
                )
            else:
                lines.append(
                    f"| {result.name} | {result.status}: {result.reason or ''} | {result.constraint or 'none'} | 0 | 0 | 0 | n/a | n/a | none |"
                )
        lines.extend(["", "## Files", ""])
        for result in results:
            if not result.files:
                continue
            lines.append(f"### {result.name}")
            for kind, file_path in result.files.items():
                lines.append(f"- {kind}: `{file_path}`")
            lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def audit_json_payload(
    input_path: Path | None,
    rows: list[dict[str, str]] | None,
    fieldnames: list[str] | None,
    column_notes: dict[str, dict[str, str | None]] | None,
    results: list[SplitResult],
    args: argparse.Namespace,
) -> dict:
    return {
        "generated_at": utc_now(),
        "seed": args.seed,
        "input": str(input_path) if input_path else None,
        "row_count": len(rows or []),
        "columns": fieldnames or [],
        "column_resolution": column_notes or {},
        "splits": [
            {
                "name": result.name,
                "status": result.status,
                "reason": result.reason,
                "constraint": result.constraint,
                "audit": result.audit,
                "files": result.files,
            }
            for result in results
        ],
    }


def main() -> int:
    args = parse_args()
    input_path = args.input or auto_detect_input()
    if input_path is None:
        write_report(args.report, None, None, None, None, [], args)
        write_json(args.audit_json, audit_json_payload(None, None, None, None, [], args))
        print(f"No input table found. Wrote report: {args.report}")
        return 0

    rows, fieldnames = read_table(input_path)
    if not rows:
        raise SystemExit(f"{input_path} contains no rows")

    if args.id_column and args.id_column not in fieldnames:
        raise SystemExit(f"--id-column {args.id_column!r} was not found")

    column_notes = column_resolution_notes(args, fieldnames)
    columns = resolve_columns(args, fieldnames)

    results: list[SplitResult] = []
    label_column = columns["label"]

    results.append(
        make_split(
            "random",
            rows,
            fieldnames,
            args.id_column,
            label_column,
            args.validation_size,
            args.test_size,
            stable_seed(args.seed, "random"),
            entity_fn=None,
            constraint=None,
        )
    )

    if columns["protein_a"] and columns["protein_b"]:
        results.append(
            make_split(
                "pair_disjoint",
                rows,
                fieldnames,
                args.id_column,
                label_column,
                args.validation_size,
                args.test_size,
                stable_seed(args.seed, "pair_disjoint"),
                entity_fn=pair_entities(columns["protein_a"], columns["protein_b"]),
                constraint="unordered protein pair",
            )
        )
    else:
        results.append(
            SplitResult(
                name="pair_disjoint",
                status="skipped",
                reason="requires detected or explicit protein-a and protein-b columns",
                constraint="unordered protein pair",
            )
        )

    if columns["modified_protein"]:
        results.append(
            make_split(
                "modified_protein_disjoint",
                rows,
                fieldnames,
                args.id_column,
                label_column,
                args.validation_size,
                args.test_size,
                stable_seed(args.seed, "modified_protein_disjoint"),
                entity_fn=modified_protein_entities(columns["modified_protein"]),
                constraint="modified protein",
            )
        )
    else:
        results.append(
            SplitResult(
                name="modified_protein_disjoint",
                status="skipped",
                reason="requires detected or explicit modified-protein column",
                constraint="modified protein",
            )
        )

    if columns["site"]:
        results.append(
            make_split(
                "site_disjoint",
                rows,
                fieldnames,
                args.id_column,
                label_column,
                args.validation_size,
                args.test_size,
                stable_seed(args.seed, "site_disjoint"),
                entity_fn=site_entities(columns["site"], columns["modified_protein"]),
                constraint="modified protein + site when available, otherwise site",
            )
        )
    else:
        results.append(
            SplitResult(
                name="site_disjoint",
                status="skipped",
                reason="requires detected or explicit site column",
                constraint="modified protein + site when available, otherwise site",
            )
        )

    if columns["pmid"]:
        results.append(
            make_split(
                "pmid_disjoint",
                rows,
                fieldnames,
                args.id_column,
                label_column,
                args.validation_size,
                args.test_size,
                stable_seed(args.seed, "pmid_disjoint"),
                entity_fn=pmid_entities(columns["pmid"]),
                constraint="PMID",
            )
        )
    else:
        results.append(
            SplitResult(
                name="pmid_disjoint",
                status="skipped",
                reason="requires detected or explicit PMID column",
                constraint="PMID",
            )
        )

    for result in results:
        write_split_artifacts(result, fieldnames, columns, args)

    write_report(args.report, input_path, rows, fieldnames, column_notes, results, args)
    write_json(args.audit_json, audit_json_payload(input_path, rows, fieldnames, column_notes, results, args))

    failed = [result for result in results if result.status == "failed"]
    if failed:
        for result in failed:
            print(f"Leakage audit failed for {result.name}: {result.reason}")
        return 1

    print(f"Wrote split leakage report: {args.report}")
    print(f"Wrote split audit JSON: {args.audit_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
