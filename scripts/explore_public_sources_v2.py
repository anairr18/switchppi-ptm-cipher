#!/usr/bin/env python3
"""Probe public PTM/PPI source candidates for PTM-PPI Shield v2.

This is an exploration-only script. It does not modify the existing PTMint
foreground pipeline. It writes clearly named v2 source-feasibility artifacts:

* data/processed/v2_public_source_feasibility.json
* results/tables/v2_public_source_feasibility.md
"""

from __future__ import annotations

import csv
import json
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PTMINT_NORMALIZED = ROOT / "data" / "processed" / "ptmint_normalized.csv"
JSON_OUTPUT = ROOT / "data" / "processed" / "v2_public_source_feasibility.json"
MD_OUTPUT = ROOT / "results" / "tables" / "v2_public_source_feasibility.md"

USER_AGENT = "switchppi-sprint-public-source-probe-v2/1.0"
TEXT_MAX_BYTES = 12_000_000
TIMEOUT_SECONDS = 45
UNIPROT_RE = re.compile(r"\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])(?:-\d+)?\b")


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    name: str
    endpoint: str
    probe_kind: str
    format: str
    fit: str
    recommendation: str
    caveat: str


SOURCES = [
    SourceSpec(
        source_id="phosphosignor_all",
        name="PhosphoSIGNOR all phosphorylation/dephosphorylation data",
        endpoint=(
            "https://signor.uniroma2.it/PhosphoSIGNOR/apis/v1/index.php"
            "?role=all&format=tsv&header=yes"
        ),
        probe_kind="text",
        format="TSV",
        fit="best primary v2 candidate for signed phosphosite causal evidence",
        recommendation=(
            "Add first as a v2 source: parse *_phSer/*_phThr/*_phTyr nodes into "
            "modified_uniprot, residue, position, effect_label, pmid, and source."
        ),
        caveat=(
            "This is signed phosphorylation causal biology, not always a direct "
            "PTM-regulated PPI effect like PTMint."
        ),
    ),
    SourceSpec(
        source_id="phosphosignor_kinases",
        name="PhosphoSIGNOR kinase subset",
        endpoint=(
            "https://signor.uniroma2.it/PhosphoSIGNOR/apis/v1/index.php"
            "?role=kinaseALL&format=tsv&header=yes"
        ),
        probe_kind="text",
        format="TSV",
        fit="kinase-to-site context for v2 features and evidence provenance",
        recommendation="Use after all-data import if kinase-specific features are useful.",
        caveat="Same schema family as PhosphoSIGNOR all; mostly auxiliary to the all-data endpoint.",
    ),
    SourceSpec(
        source_id="omnipath_interactions",
        name="OmniPath signed causal interactions",
        endpoint=(
            "https://omnipathdb.org/interactions?format=tsv&datasets=omnipath"
            "&fields=sources,references,curation_effort"
        ),
        probe_kind="text",
        format="TSV",
        fit="signed directed PPI/regulatory prior for existing PTMint pairs",
        recommendation=(
            "Add as v2 auxiliary features: signed edge prior, source count, reference count, "
            "and train-only degree controls."
        ),
        caveat="No PTM site columns; do not use as PTM-site labels without another source.",
    ),
    SourceSpec(
        source_id="omnipath_enzsub",
        name="OmniPath enzyme-substrate PTM sites",
        endpoint="https://omnipathdb.org/enzsub?format=tsv&fields=sources,references",
        probe_kind="text",
        format="TSV",
        fit="enzyme-substrate PTM site prior",
        recommendation=(
            "Add as v2 site context: kinase/phosphatase evidence for modified_uniprot, "
            "residue, position, and PTM type."
        ),
        caveat="Does not encode PTM-regulated partner PPI effect labels.",
    ),
    SourceSpec(
        source_id="elm_classes",
        name="ELM motif classes",
        endpoint="http://elm.eu.org/elms/elms_index.tsv",
        probe_kind="text",
        format="TSV",
        fit="motif/window annotations for PTM site features",
        recommendation="Add as v2 feature metadata: MOD/LIG/DOC class regex matches around the PTMint site window.",
        caveat="Class definitions are not evidence rows and are not signed effects.",
    ),
    SourceSpec(
        source_id="elm_instances_default",
        name="ELM experimentally curated motif instances",
        endpoint="http://elm.eu.org/instances.tsv",
        probe_kind="text",
        format="TSV",
        fit="curated motif instance overlap checks",
        recommendation=(
            "Use only as a secondary v2 annotation unless the full all-instance export is "
            "confirmed stable; the no-query endpoint returns a small default page."
        ),
        caveat="The attempted all-query export was slow in manual probing; default endpoint is only a sample/page.",
    ),
    SourceSpec(
        source_id="elm_interaction_domains",
        name="ELM interaction-domain mappings",
        endpoint="http://elm.eu.org/interactiondomains.tsv?q=*",
        probe_kind="text",
        format="TSV",
        fit="motif-to-domain partner context",
        recommendation="Add with ELM classes if motif-domain features are desired.",
        caveat="Does not include PTMint-style signed effect labels.",
    ),
    SourceSpec(
        source_id="biogrid_ptms",
        name="BioGRID PTM/PTMREL latest release",
        endpoint="https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-PTMS-LATEST.ptm.zip",
        probe_kind="head",
        format="ZIP with PTMTAB/PTMREL text files",
        fit="public PTM site and PTM relationship evidence",
        recommendation=(
            "Feasible but heavier: add after SIGNOR/OmniPath if v2 needs broad PTMREL "
            "relationship annotations."
        ),
        caveat=(
            "Requires zip parsing and identifier normalization; PTMTAB uses BioGRID/Entrez/RefSeq/sequence "
            "fields rather than direct PTMint-normalized UniProt pairs."
        ),
    ),
    SourceSpec(
        source_id="biogrid_all_tab3",
        name="BioGRID all interactions Tab 3 latest release",
        endpoint="https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-ALL-LATEST.tab3.zip",
        probe_kind="head",
        format="ZIP with BioGRID Tab 3 text file",
        fit="general PPI background, not PTM-site labels",
        recommendation="Do not prioritize for today's source expansion unless broad PPI background is required.",
        caveat="Large download and mostly not signed PTM-site-specific effect evidence.",
    ),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_ptmint_context(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "path": str(path),
            "present": False,
            "row_count": 0,
            "unique_modified_uniprot": 0,
            "unique_partner_uniprot": 0,
            "unique_any_uniprot": 0,
            "columns": [],
        }

    modified: set[str] = set()
    partners: set[str] = set()
    rows = 0
    columns: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        for row in reader:
            rows += 1
            if row.get("modified_uniprot"):
                modified.add(row["modified_uniprot"])
            if row.get("partner_uniprot"):
                partners.add(row["partner_uniprot"])

    return {
        "path": str(path),
        "present": True,
        "row_count": rows,
        "unique_modified_uniprot": len(modified),
        "unique_partner_uniprot": len(partners),
        "unique_any_uniprot": len(modified | partners),
        "columns": columns,
        "any_uniprot_accessions": sorted(modified | partners),
    }


def request_headers(url: str, method: str, timeout: int) -> tuple[int | None, dict[str, str], str | None]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), None
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers.items()), str(exc)
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        return None, {}, str(exc)


def fetch_text(url: str, timeout: int, max_bytes: int) -> tuple[int | None, dict[str, str], bytes, bool, str | None, float]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    started = time.perf_counter()
    chunks: list[bytes] = []
    total = 0
    truncated = False
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            while True:
                chunk = response.read(min(65536, max_bytes - total + 1))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    truncated = True
                    break
            elapsed = time.perf_counter() - started
            return response.status, dict(response.headers.items()), b"".join(chunks)[:max_bytes], truncated, None, elapsed
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        elapsed = time.perf_counter() - started
        return None, {}, b"".join(chunks)[:max_bytes], truncated, str(exc), elapsed


def decode_payload(payload: bytes, headers: dict[str, str]) -> str:
    content_type = headers.get("Content-Type", "")
    match = re.search(r"charset=([A-Za-z0-9_.-]+)", content_type)
    encodings = [match.group(1)] if match else []
    encodings.extend(["utf-8-sig", "latin-1"])
    for encoding in encodings:
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def parse_tsv(text: str) -> dict[str, object]:
    lines = [line.rstrip("\r") for line in text.splitlines() if line.strip()]
    metadata = [line for line in lines if line.startswith("#")][:8]
    header_index = None
    for index, line in enumerate(lines):
        if line.startswith("#"):
            continue
        if "\t" in line and not line.lstrip().startswith("<"):
            header_index = index
            break

    if header_index is None:
        return {
            "metadata": metadata,
            "columns": [],
            "observed_rows": 0,
            "sample_rows": [],
            "looks_like_html": any("<html" in line.lower() for line in lines[:5]),
        }

    header = next(csv.reader([lines[header_index]], delimiter="\t"))
    sample_rows: list[dict[str, str]] = []
    observed_rows = 0
    for line in lines[header_index + 1 :]:
        if line.startswith("#") or "\t" not in line:
            continue
        values = next(csv.reader([line], delimiter="\t"))
        if len(values) < len(header):
            values.extend([""] * (len(header) - len(values)))
        observed_rows += 1
        if len(sample_rows) < 3:
            sample_rows.append(dict(zip(header, values)))

    return {
        "metadata": metadata,
        "columns": header,
        "observed_rows": observed_rows,
        "sample_rows": sample_rows,
        "looks_like_html": False,
    }


def extract_uniprot_accessions(text: str) -> set[str]:
    accessions = {match.group(0).split("-")[0] for match in UNIPROT_RE.finditer(text)}
    return accessions


def probe_source(spec: SourceSpec, ptmint_accessions: set[str]) -> dict[str, object]:
    base = {
        "source_id": spec.source_id,
        "name": spec.name,
        "endpoint": spec.endpoint,
        "probe_kind": spec.probe_kind,
        "format": spec.format,
        "fit": spec.fit,
        "recommendation": spec.recommendation,
        "caveat": spec.caveat,
    }

    if spec.probe_kind == "head":
        status, headers, error = request_headers(spec.endpoint, "HEAD", TIMEOUT_SECONDS)
        base.update(
            {
                "status": "ok" if status and 200 <= status < 400 and not error else "check_needed",
                "http_status": status,
                "error": error,
                "headers": {
                    key: value
                    for key, value in headers.items()
                    if key.lower()
                    in {
                        "content-type",
                        "content-length",
                        "content-disposition",
                        "last-modified",
                        "cache-control",
                    }
                },
            }
        )
        return base

    status, headers, payload, truncated, error, elapsed = fetch_text(
        spec.endpoint,
        timeout=TIMEOUT_SECONDS,
        max_bytes=TEXT_MAX_BYTES,
    )
    text = decode_payload(payload, headers)
    parsed = parse_tsv(text)
    observed_accessions = extract_uniprot_accessions(text)
    overlap = sorted(observed_accessions & ptmint_accessions)
    columns = parsed.get("columns") or []
    base.update(
        {
            "status": (
                "ok"
                if status and 200 <= status < 400 and columns and not parsed.get("looks_like_html")
                else "check_needed"
            ),
            "http_status": status,
            "error": error,
            "elapsed_seconds": round(elapsed, 3),
            "bytes_read": len(payload),
            "truncated_at_max_bytes": truncated,
            "headers": {
                key: value
                for key, value in headers.items()
                if key.lower() in {"content-type", "content-length", "last-modified", "cache-control"}
            },
            "columns": columns,
            "observed_rows": parsed.get("observed_rows"),
            "metadata": parsed.get("metadata"),
            "sample_rows": parsed.get("sample_rows"),
            "observed_uniprot_accessions": len(observed_accessions),
            "ptmint_accession_overlap_observed": len(overlap),
            "ptmint_accession_overlap_examples": overlap[:20],
        }
    )
    return base


def md_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def endpoint_link(url: str) -> str:
    return f"[endpoint]({url})"


def row_count_text(result: dict[str, object]) -> str:
    if result.get("probe_kind") == "head":
        length = result.get("headers", {}).get("Content-Length") if isinstance(result.get("headers"), dict) else None
        return f"HEAD ok; {length or 'size not advertised'}"
    count = result.get("observed_rows")
    if result.get("truncated_at_max_bytes"):
        return f">={count}"
    return str(count if count is not None else "")


def write_json(path: Path, payload: dict[str, object]) -> None:
    ensure_parent(path)
    json_payload = dict(payload)
    context = dict(json_payload.get("ptmint_context", {}))
    context.pop("any_uniprot_accessions", None)
    json_payload["ptmint_context"] = context
    path.write_text(json.dumps(json_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    ensure_parent(path)
    context = payload["ptmint_context"]
    results: list[dict[str, object]] = payload["sources"]  # type: ignore[assignment]

    lines = [
        "# PTM-PPI Shield v2 Public Source Feasibility",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "## Existing PTMint Context",
        "",
        f"- Normalized table: `{context['path']}`",
        f"- Rows: {context['row_count']}",
        f"- Unique UniProt accessions across modified and partner proteins: {context['unique_any_uniprot']}",
        f"- Columns: `{', '.join(context['columns'])}`",
        "",
        "## Source Checks",
        "",
        "| Source | Endpoint | Probe | Rows / Size | Columns Seen | Fit | Recommendation |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for result in results:
        columns = result.get("columns") or []
        if isinstance(columns, list) and columns:
            column_text = ", ".join(str(column) for column in columns[:8])
            if len(columns) > 8:
                column_text += f", ... ({len(columns)} cols)"
        else:
            column_text = ""
        lines.append(
            "| {source} | {endpoint} | {status} | {rows} | {columns} | {fit} | {recommendation} |".format(
                source=md_escape(result["name"]),
                endpoint=endpoint_link(str(result["endpoint"])),
                status=md_escape(result["status"]),
                rows=md_escape(row_count_text(result)),
                columns=md_escape(column_text),
                fit=md_escape(result["fit"]),
                recommendation=md_escape(result["recommendation"]),
            )
        )

    lines.extend(
        [
            "",
            "## Feasible Additions Today",
            "",
            "1. PhosphoSIGNOR all-data TSV is the strongest new public source. It provides SIGNOR-scored, PMID-backed signed phosphorylation/dephosphorylation causal relationships and encodes modified residue/position in node IDs such as `Q9UKV8_phSer387`.",
            "2. OmniPath interactions can be added as signed causal network priors for PTMint pairs. Keep it auxiliary because it is not site-specific.",
            "3. OmniPath `enzsub` can add kinase/substrate/PTM-site context for `modified_uniprot + residue + position` rows. Keep it auxiliary because it lacks partner-effect labels.",
            "4. ELM classes and interaction-domain mappings are easy feature additions for motif/window annotation. ELM all-instance export needs a more careful download path because the `q=*` probe timed out manually.",
            "5. BioGRID PTMS is public and feasible, but it is a second-pass task because the ZIP is larger and PTMTAB/PTMREL need identifier normalization before they fit the PTMint schema.",
            "",
            "## Not Good Primary Label Additions",
            "",
            "- BioGRID all Tab 3 is a useful broad PPI background, but it is large and mostly not signed PTM-site effect evidence.",
            "- OmniPath signed interactions should not be converted directly into PTM-site labels without site evidence.",
            "- ELM motif classes are feature annotations, not positive/negative effect labels.",
            "",
            "## Suggested v2 Integration Order",
            "",
            "1. Build `ingest_phosphosignor_v2.py` to normalize PhosphoSIGNOR into a new `data/processed/v2_phosphosignor_normalized.csv` table.",
            "2. Build `annotate_omnipath_v2.py` to generate pair-level signed-prior and site-level enzyme-substrate feature tables keyed to PTMint rows.",
            "3. Build `annotate_elm_v2.py` for motif class regex hits on PTM site windows.",
            "4. Add BioGRID PTMS only after the first three sources are stable.",
            "",
            "## Caveats",
            "",
        ]
    )
    for result in results:
        lines.append(f"- {result['name']}: {result['caveat']}")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    ptmint_context = read_ptmint_context(PTMINT_NORMALIZED)
    ptmint_accessions = set(ptmint_context.get("any_uniprot_accessions", []))
    results = [probe_source(source, ptmint_accessions) for source in SOURCES]
    payload = {
        "generated_at_utc": utc_now(),
        "script": str(Path(__file__).resolve()),
        "text_max_bytes": TEXT_MAX_BYTES,
        "timeout_seconds": TIMEOUT_SECONDS,
        "ptmint_context": ptmint_context,
        "sources": results,
    }
    write_json(JSON_OUTPUT, payload)
    write_markdown(MD_OUTPUT, payload)
    print(f"Wrote {JSON_OUTPUT}")
    print(f"Wrote {MD_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
