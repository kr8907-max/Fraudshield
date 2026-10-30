"""
FraudShield - dataset loading and cleaning
==========================================

Turns a CSV file with two columns (url,label) into the three lists the
machine-learning stage needs:

    X     = feature vectors, one list of 14 numbers per URL
    y     = labels, 0 (safe) or 1 (malicious)
    urls  = the matching URL strings, for inspection and debugging

This module deliberately does NOT train anything and does NOT split data
into train/test sets. It only reads, validates, de-duplicates, and converts.

Library rules for this project:
    - Standard library csv/pathlib only.
    - NO pandas.
    - Feature order always comes from backend.feature_extractor.FEATURE_NAMES.

How to run the demo report:
    python backend/dataset_loader.py
    python backend/dataset_loader.py data/some_other_file.csv
    python backend/dataset_loader.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# When this file is run directly (python backend/dataset_loader.py), Python
# only puts backend/ on the import path. Add the project root so the shared
# "backend.feature_extractor" import below works in every launch style.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.feature_extractor import FEATURE_NAMES, extract_features

# Project root is one directory above backend/. This avoids hardcoding
# C:\fraudshield and works no matter which folder the terminal starts in.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "test_urls.csv"

# Labels accepted out of the box. Keys are compared lowercase after
# stripping whitespace, so " Safe " and "SAFE" both work.
# Unknown labels are NEVER guessed - they are reported as invalid rows.
DEFAULT_LABEL_MAP: dict[str, int] = {
    "0": 0,
    "1": 1,
    "safe": 0,
    "benign": 0,
    "legitimate": 0,
    "clean": 0,
    "malicious": 1,
    "phishing": 1,
    "malware": 1,
    "fraud": 1,
    "suspicious": 1,
}


class DatasetError(Exception):
    """Raised when a dataset file cannot be used at all."""


# ---------------------------------------------------------------------------
# Report structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowIssue:
    """One rejected row and the reason it was rejected."""

    line: int
    url: str
    reason: str


@dataclass(frozen=True)
class DuplicateConflict:
    """The same URL appears with different labels; it must not be trained on."""

    url: str
    labels: tuple[int, ...]
    lines: tuple[int, ...]


@dataclass
class DatasetReport:
    """Factual statistics collected while reading a dataset."""

    path: Path
    total_rows: int = 0          # non-blank data rows (header excluded)
    blank_rows: int = 0
    valid_rows: int = 0          # passed URL + label validation, before de-duplication
    invalid_rows: int = 0
    duplicate_rows_removed: int = 0
    conflict_rows_removed: int = 0
    usable_rows: int = 0         # rows that actually became X/y entries
    safe_count: int = 0
    malicious_count: int = 0
    conflicts: list[DuplicateConflict] = field(default_factory=list)
    issues: list[RowIssue] = field(default_factory=list)
    # Original (pre-binary-mapping) label text counts, for provenance reports.
    # Example keys for a Kaggle "type" column: benign/phishing/defacement/malware.
    raw_label_counts: dict[str, int] = field(default_factory=dict)
    # Source-label counts only for rows that survived cleaning and are in X/y.
    usable_source_counts: dict[str, int] = field(default_factory=dict)

    @property
    def feature_count(self) -> int:
        return len(FEATURE_NAMES)

    @property
    def has_problems(self) -> bool:
        """True when at least one row was rejected or one label conflict exists."""
        return bool(self.invalid_rows or self.conflicts)

    def _percent(self, count: int) -> str:
        if self.usable_rows == 0:
            return "0.0%"
        return f"{count / self.usable_rows * 100:.1f}%"

    def format(self) -> str:
        """Build the human-readable report printed by the command line."""
        lines = [
            "Dataset Report",
            "-" * 62,
            f"{'File:':<24}{self.path}",
            f"{'Total data rows:':<24}{self.total_rows}",
            f"{'Blank rows skipped:':<24}{self.blank_rows}",
            f"{'Valid rows:':<24}{self.valid_rows}",
            f"{'Invalid rows:':<24}{self.invalid_rows}",
            f"{'Duplicates removed:':<24}{self.duplicate_rows_removed}",
            f"{'Conflicting URLs:':<24}{len(self.conflicts)}",
            f"{'Conflict rows removed:':<24}{self.conflict_rows_removed}",
            f"{'Usable rows:':<24}{self.usable_rows}",
            "",
            f"{'Safe (0):':<24}{self.safe_count} ({self._percent(self.safe_count)})",
            f"{'Malicious (1):':<24}{self.malicious_count} ({self._percent(self.malicious_count)})",
            f"{'Features per URL:':<24}{self.feature_count}",
        ]

        if self.issues:
            lines.extend(["", "Rejected rows:"])
            for issue in self.issues:
                shown_url = issue.url if issue.url else "(empty)"
                lines.append(f"  line {issue.line}: {shown_url} -> {issue.reason}")

        if self.conflicts:
            lines.extend(["", "Conflicting duplicates (excluded from the dataset):"])
            for conflict in self.conflicts:
                label_text = ", ".join(
                    f"{label} (line {line})" for label, line in zip(conflict.labels, conflict.lines)
                )
                lines.append(f"  {conflict.url} -> {label_text}")

        if self.usable_rows == 0:
            lines.extend(["", "WARNING: no usable rows - nothing can be trained on this file."])

        return "\n".join(lines)


@dataclass
class DatasetBundle:
    """Everything later stages need, plus the report explaining how it was built."""

    X: list[list[int]]
    y: list[int]
    urls: list[str]
    report: DatasetReport


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _build_label_map(extra: dict[str, int] | None) -> dict[str, int]:
    """Merge caller-supplied label aliases over the defaults, validating values."""
    mapping = dict(DEFAULT_LABEL_MAP)
    if extra:
        for raw_key, raw_value in extra.items():
            key = str(raw_key).strip().lower()
            if raw_value not in (0, 1):
                raise DatasetError(
                    f"Custom label mapping for {raw_key!r} must be 0 or 1, got {raw_value!r}"
                )
            mapping[key] = int(raw_value)
    return mapping


def _parse_label(text: str, label_map: dict[str, int]) -> int | None:
    """Return 0/1 for a known label, or None for an unknown one (never guess)."""
    return label_map.get(text.strip().lower())


def _field(row: list[str], index: int) -> str:
    """Read a column safely even if the row is shorter than the header."""
    return row[index] if index < len(row) else ""


def load_dataset(
    csv_path: str | Path,
    *,
    label_map: dict[str, int] | None = None,
    url_column: str = "url",
    label_column: str = "label",
    progress_every: int = 0,
) -> DatasetBundle:
    """
    Read, validate, clean, and convert a url,label CSV.

    Cleaning rules:
      - UTF-8 is read safely (a BOM at the start is ignored).
      - Rows that are entirely blank are skipped.
      - Surrounding whitespace is stripped from the URL and label only.
        The URL itself is NOT rewritten, lowercased, or shortened, because
        its path, query, subdomains, and scheme are real ML features.
      - A URL is usable when the shared feature extractor can parse it into
        a non-empty URL and hostname.
      - Duplicate URLs are collapsed to their first occurrence.
      - A URL that appears with BOTH label 0 and label 1 is reported and
        excluded completely; picking one side would corrupt training.

    Extra options:
      - url_column / label_column let other CSV layouts (e.g. Kaggle's
        "url,type") reuse this loader without changing the file format.
      - label_map adds explicit aliases; unknown labels are still rejected.
      - progress_every prints a row counter every N non-blank rows (0 = off).

    Raises:
        DatasetError: missing file, unreadable text, empty file, missing
        columns, or an invalid custom label mapping.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise DatasetError(f"Dataset file not found: {path}")

    mapping = _build_label_map(label_map)
    report = DatasetReport(path=path)

    # (line, URL, binary label, original label text, features) for valid rows.
    valid_rows: list[tuple[int, str, int, str, dict[str, int]]] = []

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)

            try:
                raw_header = next(reader)
            except StopIteration:
                raise DatasetError(f"Dataset file is empty: {path}") from None

            header = [name.strip().lower() for name in raw_header]
            wanted_url = url_column.strip().lower()
            wanted_label = label_column.strip().lower()

            if wanted_url not in header or wanted_label not in header:
                raise DatasetError(
                    f"Dataset must contain '{url_column}' and '{label_column}' columns; "
                    f"found: {raw_header}"
                )

            url_index = header.index(wanted_url)
            label_index = header.index(wanted_label)

            for row in reader:
                line = reader.line_num

                # Skip blank lines (csv returns [] or only whitespace cells).
                if not row or all(not cell.strip() for cell in row):
                    report.blank_rows += 1
                    continue

                report.total_rows += 1
                url = _field(row, url_index).strip()
                label_text = _field(row, label_index).strip()
                raw_label = label_text.lower()

                # Keep provenance counts of the original label vocabulary.
                report.raw_label_counts[raw_label] = report.raw_label_counts.get(raw_label, 0) + 1

                if progress_every and report.total_rows % progress_every == 0:
                    print(f"  Processed {report.total_rows:,} rows...", flush=True)

                # 1) URL must be usable by the shared extractor.
                features = extract_features(url)
                if features["url_length"] == 0 or features["hostname_length"] == 0:
                    report.invalid_rows += 1
                    report.issues.append(
                        RowIssue(line, url, "invalid URL: not a usable http(s) web address")
                    )
                    continue

                # 2) Label must be recognised.
                label = _parse_label(label_text, mapping)
                if label is None:
                    report.invalid_rows += 1
                    report.issues.append(
                        RowIssue(
                            line,
                            url,
                            f"unsupported label {label_text!r} (expected 0/1 or a known alias)",
                        )
                    )
                    continue

                valid_rows.append((line, url, label, raw_label, features))

    except UnicodeDecodeError as error:
        raise DatasetError(f"Dataset is not valid UTF-8: {path} ({error})") from error

    report.valid_rows = len(valid_rows)

    # --- Group by exact URL to find duplicates and conflicting labels ------
    # Exact match only: no aggressive normalisation, so two genuinely
    # different URLs are never merged by accident.
    grouped: dict[str, list[tuple[int, int]]] = {}
    for line, url, label, _raw_label, _features in valid_rows:
        grouped.setdefault(url, []).append((line, label))

    conflicting_urls: set[str] = set()
    for url, entries in grouped.items():
        distinct_labels = tuple(sorted({label for _line, label in entries}))
        if len(distinct_labels) > 1:
            conflicting_urls.add(url)
            report.conflicts.append(
                DuplicateConflict(
                    url=url,
                    labels=distinct_labels,
                    lines=tuple(line for line, _label in entries),
                )
            )

    # --- Build final X / y / urls (first occurrence wins) -------------------
    final_urls: list[str] = []
    final_y: list[int] = []
    final_X: list[list[int]] = []
    seen_urls: set[str] = set()

    for _line, url, label, raw_label, features in valid_rows:
        if url in conflicting_urls or url in seen_urls:
            if url in seen_urls and url not in conflicting_urls:
                report.duplicate_rows_removed += 1
            continue

        seen_urls.add(url)
        final_urls.append(url)
        final_y.append(label)
        report.usable_source_counts[raw_label] = report.usable_source_counts.get(raw_label, 0) + 1
        # The one official feature order. No second hand-written list.
        final_X.append([features[name] for name in FEATURE_NAMES])

    report.conflict_rows_removed = sum(len(grouped[url]) for url in conflicting_urls)
    report.usable_rows = len(final_urls)
    report.safe_count = final_y.count(0)
    report.malicious_count = final_y.count(1)

    return DatasetBundle(X=final_X, y=final_y, urls=final_urls, report=report)


# ---------------------------------------------------------------------------
# Command line entry point
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FraudShield dataset loader")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(DEFAULT_DATASET),
        help="CSV file with url,label columns (default: data/test_urls.csv)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the built-in deterministic tests instead of loading a file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with code 1 if any row is invalid or any label conflicts",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return _run_self_test()

    try:
        bundle = load_dataset(args.csv_path)
    except DatasetError as error:
        print(f"Dataset error: {error}")
        return 1

    print(bundle.report.format())
    print()

    if bundle.X:
        sample_url = bundle.urls[0]
        sample_vector = bundle.X[0]
        print("Pipeline dry run (features only, no model):")
        print(f"  X: {len(bundle.X)} rows x {len(bundle.X[0])} features")
        print(f"  y: {len(bundle.y)} labels")
        print(f"  urls: {len(bundle.urls)} URLs")
        print(f"  sample URL: {sample_url}")
        print(f"  sample vector: {sample_vector}")
        print()
        print("Stage 4 will pass these lists to train_test_split + RandomForestClassifier.")
    else:
        print("No usable rows; fix the dataset before training.")

    if args.strict and bundle.report.has_problems:
        print()
        print("Strict mode: rejected rows or label conflicts were found.")
        return 1

    return 0


# ---------------------------------------------------------------------------
# Deterministic self-tests
# ---------------------------------------------------------------------------

_SELF_TEST_CSV = """url,label
https://www.google.com,0
http://secure-login.example/verify-account,1
https://www.google.com,0
https://www.github.com,0
https://www.github.com,1
not a url,0
https://www.wikipedia.org,maybe
"https://example.example/path?next=https://evil.example/login&x=1",1
  https://www.python.org  ,  0

https://www.cloudflare.com,0
"""


def _run_self_test() -> int:
    """Run checks against temporary files. Returns 0 when all pass."""
    checks: list[tuple[str, bool]] = []

    def check(label: str, condition: bool) -> None:
        checks.append((label, bool(condition)))

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "self_test.csv"
        temp_path.write_text(_SELF_TEST_CSV, encoding="utf-8")

        bundle = load_dataset(temp_path)
        report = bundle.report

        # --- Report counts, calculated from the known file above ------------
        check("total non-blank data rows == 10", report.total_rows == 10)
        check("blank rows skipped == 1", report.blank_rows == 1)
        check("invalid rows == 2", report.invalid_rows == 2)
        check("valid rows before cleaning == 8", report.valid_rows == 8)
        check("exact duplicate removed == 1", report.duplicate_rows_removed == 1)
        check("one conflicting URL detected", len(report.conflicts) == 1)
        check("both conflicting rows removed", report.conflict_rows_removed == 2)
        check("usable rows == 5", report.usable_rows == 5)
        check("safe count == 3", report.safe_count == 3)
        check("malicious count == 2", report.malicious_count == 2)
        check(
            "raw label counts include invalid-label rows",
            report.raw_label_counts == {"0": 6, "1": 3, "maybe": 1},
        )
        check(
            "usable source counts match final rows",
            report.usable_source_counts == {"0": 3, "1": 2},
        )

        # --- Invalid rows are reported, not silently ignored ---------------
        reasons = " | ".join(issue.reason for issue in report.issues)
        check("invalid URL is reported", "invalid URL" in reasons)
        check("unknown label is reported", "unsupported label" in reasons)

        # --- Conflict details are visible ----------------------------------
        conflict = report.conflicts[0] if report.conflicts else None
        check(
            "conflict lists URL and both labels",
            conflict is not None
            and conflict.url == "https://www.github.com"
            and conflict.labels == (0, 1)
            and len(conflict.lines) == 2,
        )

        # --- Normal safe URL survives with label 0 -------------------------
        check(
            "normal safe URL kept with label 0",
            "https://www.google.com" in bundle.urls
            and bundle.y[bundle.urls.index("https://www.google.com")] == 0,
        )

        # --- Suspicious .example URL survives with label 1 ----------------
        suspicious_url = "http://secure-login.example/verify-account"
        check(
            "suspicious .example URL kept with label 1",
            suspicious_url in bundle.urls
            and bundle.y[bundle.urls.index(suspicious_url)] == 1,
        )

        # --- CSV quoting for URLs containing commas ------------------------
        comma_url = "https://example.example/path?next=https://evil.example/login&x=1"
        check("comma inside quoted URL parsed intact", comma_url in bundle.urls)

        # --- Whitespace is stripped, URL content is preserved -------------
        check(
            "surrounding whitespace stripped from URL",
            "https://www.python.org" in bundle.urls,
        )

        # --- Output shape and feature contract ----------------------------
        check("X, y, urls lengths match", len(bundle.X) == len(bundle.y) == len(bundle.urls))
        check("every vector has len(FEATURE_NAMES) values", all(len(row) == len(FEATURE_NAMES) for row in bundle.X))
        check(
            "every feature value is numeric",
            all(isinstance(value, (int, float)) and not isinstance(value, bool) for row in bundle.X for value in row),
        )
        check(
            "vector order matches FEATURE_NAMES exactly",
            all(
                row == [extract_features(url)[name] for name in FEATURE_NAMES]
                for url, row in zip(bundle.urls, bundle.X)
            ),
        )

        # --- Determinism ---------------------------------------------------
        again = load_dataset(temp_path)
        check("loading twice gives identical X/y", again.X == bundle.X and again.y == bundle.y)

        # --- UTF-8 BOM is handled ------------------------------------------
        bom_path = Path(temp_dir) / "bom.csv"
        bom_path.write_text("url,label\nhttps://www.example.org,0\n", encoding="utf-8-sig")
        bom_bundle = load_dataset(bom_path)
        check("UTF-8 BOM file loads one row", bom_bundle.report.usable_rows == 1)

        # --- Textual labels map explicitly, unknown ones do not ------------
        text_path = Path(temp_dir) / "text_labels.csv"
        text_path.write_text(
            "url,label\nhttps://www.example.org,benign\nhttps://secure.example.org,phishing\n",
            encoding="utf-8",
        )
        text_bundle = load_dataset(text_path)
        check(
            "textual labels mapped to 0/1",
            text_bundle.y == [0, 1] and text_bundle.report.invalid_rows == 0,
        )

        custom_path = Path(temp_dir) / "custom_labels.csv"
        custom_path.write_text("url,label\nhttps://www.example.org,good\n", encoding="utf-8")
        custom_bundle = load_dataset(custom_path, label_map={"good": 0})
        check("custom label mapping respected", custom_bundle.y == [0])

        unknown_path = Path(temp_dir) / "unknown_labels.csv"
        unknown_path.write_text("url,label\nhttps://www.example.org,good\n", encoding="utf-8")
        unknown_bundle = load_dataset(unknown_path)
        check(
            "unknown label rejected, not guessed",
            unknown_bundle.report.invalid_rows == 1 and unknown_bundle.report.usable_rows == 0,
        )

        # --- Structural errors are clear -----------------------------------
        try:
            load_dataset(Path(temp_dir) / "missing.csv")
            check("missing file raises DatasetError", False)
        except DatasetError:
            check("missing file raises DatasetError", True)

        wrong_header = Path(temp_dir) / "wrong_header.csv"
        wrong_header.write_text("address,class\nhttps://www.example.org,0\n", encoding="utf-8")
        try:
            load_dataset(wrong_header)
            check("missing columns raise DatasetError", False)
        except DatasetError:
            check("missing columns raise DatasetError", True)

        empty_path = Path(temp_dir) / "empty.csv"
        empty_path.write_text("", encoding="utf-8")
        try:
            load_dataset(empty_path)
            check("empty file raises DatasetError", False)
        except DatasetError:
            check("empty file raises DatasetError", True)

    failed = 0
    for label, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {label}")
        if not passed:
            failed += 1

    print()
    print(f"{len(checks) - failed}/{len(checks)} checks passed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_main())
