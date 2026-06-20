#!/usr/bin/env python3
"""Download, normalize, and validate PTMint PTM-PPI evidence rows.

The script keeps ingestion scoped to data assets:

* data/raw/ptmint_experimental_evidence.csv
* data/raw/uniprot_ptmint_accessions.fasta
* data/processed/ptmint_normalized.csv
* data/processed/ptmint_invalid_rows.csv
* data/processed/ptmint_ingest_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PTMINT_URL = "https://ptmint.sjtu.edu.cn/data/PTM%20experimental%20evidence.csv"
UNIPROT_ACCESSIONS_URL = "https://rest.uniprot.org/uniprotkb/accessions"
SOURCE_NAME = "PTMint"

OUTPUT_COLUMNS = [
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
    "detection_method",
    "disease",
    "source",
]

INVALID_COLUMNS = OUTPUT_COLUMNS + [
    "invalid_reason",
    "uniprot_residue",
    "uniprot_length",
]

REQUIRED_RAW_COLUMNS = {
    "Organism",
    "Gene",
    "Uniprot",
    "PTM",
    "Site",
    "AA",
    "Int_uniprot",
    "Int_gene",
    "Effect",
    "Method",
    "Disease",
    "PMID",
}

MISSING_MARKERS = {"", "-", "na", "n/a", "none", "null"}
ACCESSION_SPLIT_RE = re.compile(r"[;,\s]+")
EFFECT_LABELS = {
    "enhance": "Enhance",
    "inhibit": "Inhibit",
    "induce": "Induce",
}
PTM_TYPES = {
    "ac": "Ac",
    "glyco": "Glyco",
    "me": "Me",
    "phos": "Phos",
    "sumo": "Sumo",
    "ub": "Ub",
}


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = default_project_root()
    parser = argparse.ArgumentParser(
        description="Ingest PTMint evidence and validate PTM residues against UniProt sequences."
    )
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--ptmint-url",
        default=PTMINT_URL,
        help="PTMint evidence CSV URL.",
    )
    parser.add_argument(
        "--raw-ptmint",
        type=Path,
        default=root / "data" / "raw" / "ptmint_experimental_evidence.csv",
        help="Path for the raw PTMint evidence CSV.",
    )
    parser.add_argument(
        "--uniprot-fasta",
        type=Path,
        default=root / "data" / "raw" / "uniprot_ptmint_accessions.fasta",
        help="Path for downloaded UniProt FASTA records.",
    )
    parser.add_argument(
        "--normalized-output",
        type=Path,
        default=root / "data" / "processed" / "ptmint_normalized.csv",
    )
    parser.add_argument(
        "--invalid-output",
        type=Path,
        default=root / "data" / "processed" / "ptmint_invalid_rows.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=root / "data" / "processed" / "ptmint_ingest_summary.json",
    )
    parser.add_argument(
        "--refresh-ptmint",
        action="store_true",
        help="Download PTMint even when the raw CSV already exists.",
    )
    parser.add_argument(
        "--refresh-uniprot",
        action="store_true",
        help="Redownload all UniProt FASTA records.",
    )
    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Use only local raw inputs. Missing inputs will still fail.",
    )
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def request_url(url: str, timeout: int, retries: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "switchppi-sprint-ptmint-ingest/1.0",
            "Accept": "*/*",
        },
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"failed to download {url}: {last_error}") from last_error


def download_file(url: str, output_path: Path, timeout: int, retries: int) -> None:
    ensure_parent(output_path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_bytes(request_url(url, timeout=timeout, retries=retries))
    tmp_path.replace(output_path)


def clean_text(value: object, *, missing_to_empty: bool = True) -> str:
    text = "" if value is None else str(value).strip()
    if missing_to_empty and text.lower() in MISSING_MARKERS:
        return ""
    return text


def clean_accession(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    return ACCESSION_SPLIT_RE.split(text)[0].strip()


def clean_effect_label(value: object) -> str:
    text = clean_text(value)
    return EFFECT_LABELS.get(text.lower(), text)


def clean_ptm_type(value: object) -> str:
    text = clean_text(value)
    return PTM_TYPES.get(text.lower(), text)


def clean_position(value: object) -> str:
    text = clean_text(value, missing_to_empty=False)
    try:
        return str(int(text))
    except ValueError:
        return text


def read_ptmint_rows(raw_path: Path) -> list[dict[str, str]]:
    with raw_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_RAW_COLUMNS - fieldnames)
        if missing_columns:
            raise ValueError(
                f"{raw_path} is missing required PTMint columns: {', '.join(missing_columns)}"
            )
        return [dict(row) for row in reader]


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "modified_uniprot": clean_accession(row.get("Uniprot")),
        "modified_gene": clean_text(row.get("Gene")),
        "partner_uniprot": clean_accession(row.get("Int_uniprot")),
        "partner_gene": clean_text(row.get("Int_gene")),
        "organism": clean_text(row.get("Organism")),
        "ptm_type": clean_ptm_type(row.get("PTM")),
        "residue": clean_text(row.get("AA"), missing_to_empty=False).upper(),
        "position": clean_position(row.get("Site")),
        "effect_label": clean_effect_label(row.get("Effect")),
        "pmid": clean_text(row.get("PMID")),
        "detection_method": clean_text(row.get("Method")),
        "disease": clean_text(row.get("Disease")),
        "source": SOURCE_NAME,
    }


def iter_accessions(rows: Iterable[dict[str, str]]) -> set[str]:
    accessions: set[str] = set()
    for row in rows:
        for column in ("modified_uniprot", "partner_uniprot"):
            accession = clean_accession(row.get(column))
            if accession:
                accessions.add(accession)
    return accessions


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def parse_fasta(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    sequences: dict[str, list[str]] = {}
    accession: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                accession = fasta_accession(line)
                if accession:
                    sequences.setdefault(accession, [])
                continue
            if accession:
                sequences[accession].append(line)
    return {accession: "".join(parts).upper() for accession, parts in sequences.items()}


def fasta_accession(header: str) -> str:
    first_token = header[1:].split(None, 1)[0]
    pieces = first_token.split("|")
    if len(pieces) >= 2:
        return pieces[1]
    return first_token


def fetch_uniprot_fasta(
    accessions: set[str],
    fasta_path: Path,
    *,
    refresh: bool,
    batch_size: int,
    timeout: int,
    retries: int,
) -> dict[str, str]:
    ensure_parent(fasta_path)
    existing_sequences = {} if refresh else parse_fasta(fasta_path)
    missing = sorted(accessions - set(existing_sequences))

    if not missing:
        return existing_sequences

    mode = "w" if refresh or not fasta_path.exists() else "a"
    with fasta_path.open(mode, encoding="utf-8", newline="\n") as handle:
        if mode == "a" and fasta_path.stat().st_size > 0:
            handle.write("\n")
        for batch in chunks(missing, batch_size):
            query = urllib.parse.urlencode(
                {
                    "accessions": ",".join(batch),
                    "format": "fasta",
                }
            )
            payload = request_url(
                f"{UNIPROT_ACCESSIONS_URL}?{query}",
                timeout=timeout,
                retries=retries,
            ).decode("utf-8")
            if payload.strip():
                handle.write(payload.rstrip() + "\n")

    return parse_fasta(fasta_path)


def validate_row(row: dict[str, str], sequences: dict[str, str]) -> tuple[bool, str, str, str]:
    reasons: list[str] = []
    accession = row["modified_uniprot"]
    residue = row["residue"]
    position_text = row["position"]

    if not accession:
        reasons.append("missing_modified_uniprot")
    if not residue:
        reasons.append("missing_residue")
    elif not re.fullmatch(r"[A-Z]", residue):
        reasons.append("invalid_residue")

    try:
        position = int(position_text)
    except ValueError:
        position = 0
        reasons.append("invalid_position")

    sequence = sequences.get(accession, "")
    if accession and not sequence:
        reasons.append("missing_uniprot_sequence")

    uniprot_residue = ""
    uniprot_length = str(len(sequence)) if sequence else ""
    if sequence and position:
        if position < 1 or position > len(sequence):
            reasons.append("position_out_of_range")
        else:
            uniprot_residue = sequence[position - 1]
            if residue and uniprot_residue != residue:
                reasons.append("residue_mismatch")

    return not reasons, ";".join(reasons), uniprot_residue, uniprot_length


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, str]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: dict[str, object]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run() -> int:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    if not args.skip_downloads and (args.refresh_ptmint or not args.raw_ptmint.exists()):
        download_file(
            args.ptmint_url,
            args.raw_ptmint,
            timeout=args.timeout,
            retries=args.retries,
        )

    if not args.raw_ptmint.exists():
        raise FileNotFoundError(f"PTMint CSV not found: {args.raw_ptmint}")

    raw_rows = read_ptmint_rows(args.raw_ptmint)
    normalized_candidates = [normalize_row(row) for row in raw_rows]
    accessions = iter_accessions(normalized_candidates)

    if args.skip_downloads:
        sequences = parse_fasta(args.uniprot_fasta)
    else:
        sequences = fetch_uniprot_fasta(
            accessions,
            args.uniprot_fasta,
            refresh=args.refresh_uniprot,
            batch_size=args.batch_size,
            timeout=args.timeout,
            retries=args.retries,
        )

    valid_rows: list[dict[str, str]] = []
    invalid_rows: list[dict[str, str]] = []
    invalid_reason_counts: Counter[str] = Counter()

    for row in normalized_candidates:
        is_valid, reason_text, uniprot_residue, uniprot_length = validate_row(row, sequences)
        if is_valid:
            valid_rows.append(row)
        else:
            invalid_row = dict(row)
            invalid_row["invalid_reason"] = reason_text
            invalid_row["uniprot_residue"] = uniprot_residue
            invalid_row["uniprot_length"] = uniprot_length
            invalid_rows.append(invalid_row)
            invalid_reason_counts.update(reason_text.split(";"))

    valid_label_counts = Counter(row["effect_label"] for row in valid_rows)
    raw_label_counts = Counter(row["effect_label"] for row in normalized_candidates)

    write_csv(args.normalized_output, OUTPUT_COLUMNS, valid_rows)
    write_csv(args.invalid_output, INVALID_COLUMNS, invalid_rows)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE_NAME,
        "ptmint_url": args.ptmint_url,
        "uniprot_accessions_url": UNIPROT_ACCESSIONS_URL,
        "raw_ptmint_csv": str(args.raw_ptmint),
        "uniprot_fasta": str(args.uniprot_fasta),
        "normalized_output": str(args.normalized_output),
        "invalid_output": str(args.invalid_output),
        "raw_rows": len(raw_rows),
        "normalized_rows": len(valid_rows),
        "invalid_rows": len(invalid_rows),
        "unique_uniprot_accessions_requested": len(accessions),
        "unique_uniprot_sequences_loaded": len(sequences),
        "valid_label_distribution": dict(sorted(valid_label_counts.items())),
        "raw_label_distribution": dict(sorted(raw_label_counts.items())),
        "invalid_reason_distribution": dict(sorted(invalid_reason_counts.items())),
    }
    write_summary(args.summary_output, summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
