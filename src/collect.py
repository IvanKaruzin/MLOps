"""Build the raw chat dataset from a pinned Hugging Face CSV snapshot."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.config import load_params, source_files
from src.schema import Example, dump

REVISION_RE = re.compile(r"[0-9a-f]{40}")
DATASET_ID_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class Candidate:
    """Internal audit fields that are intentionally not written to JSONL."""

    source_id: str
    release_year: int
    language: str
    description: str
    genres: tuple[str, ...]


def normalize_text(value: object) -> str:
    """Normalize source text without adding title or genre information."""
    return " ".join(unicodedata.normalize("NFC", str(value or "")).split())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_source_snapshot(params: dict) -> list[Path]:
    """Use the local snapshot, or download exactly the configured HF revision."""
    cfg = params["collect"]
    source = cfg["source"]
    paths = source_files(params, require_exists=False)

    if source["kind"] != "huggingface_snapshot":
        raise ValueError(f"unsupported collect.source.kind: {source['kind']!r}")
    dataset_id = str(source["dataset_id"])
    if DATASET_ID_RE.fullmatch(dataset_id) is None:
        raise ValueError("collect.source.dataset_id must be an owner/name pair")
    revision = str(source["revision"])
    if REVISION_RE.fullmatch(revision) is None:
        raise ValueError("collect.source.revision must be a full 40-character commit hash")
    filename = Path(str(source["file"]))
    if filename.is_absolute() or ".." in filename.parts:
        raise ValueError("collect.source.file must be a safe relative path")
    if len(paths) != 1 or paths[0].name != filename.name:
        raise ValueError("the selected version must point to the configured HF file")

    target = paths[0]
    if not target.resolve().is_relative_to((Path.cwd() / "sources").resolve()):
        raise ValueError("the source snapshot target must stay inside sources/")
    expected_checksum = str(source["sha256"]).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_checksum):
        raise ValueError("collect.source.sha256 must be a 64-character SHA-256 digest")

    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        # Download into an isolated temporary cache. Only the requested file is
        # copied into sources/, so Hub metadata and unrelated repository files
        # never become dataset inputs or DVC artifacts.
        with tempfile.TemporaryDirectory(prefix="movies-hf-") as cache_dir:
            downloaded = Path(
                hf_hub_download(
                    repo_id=dataset_id,
                    repo_type="dataset",
                    revision=revision,
                    filename=filename.as_posix(),
                    cache_dir=cache_dir,
                )
            )
            temporary_target = target.with_suffix(target.suffix + ".part")
            shutil.copyfile(downloaded, temporary_target)
            os.replace(temporary_target, target)

    actual_checksum = file_sha256(target)
    if actual_checksum != expected_checksum:
        raise ValueError(
            f"source checksum mismatch for {target}: "
            f"expected {expected_checksum}, got {actual_checksum}"
        )
    return source_files(params)


def parse_release_date(value: object) -> date | None:
    text = normalize_text(value)
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.isoformat() == text else None


def stable_id(title: str, release_date: str, language: str, cfg: dict) -> str:
    id_cfg = cfg["transforms"]["id"]
    if id_cfg["algorithm"] != "sha256":
        raise ValueError("collect.transforms.id.algorithm must be sha256")
    payload = id_cfg["separator"].join((title, release_date, language))
    return id_cfg["prefix"] + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pick_prompt(source_id: str, variants: list[str]) -> str:
    """Choose a prompt deterministically; Python hash() is deliberately avoided."""
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()
    return variants[int(digest, 16) % len(variants)]


def _read_source_rows(paths: list[Path], required_columns: set[str]):
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            columns = set(reader.fieldnames or ())
            missing = required_columns - columns
            if missing:
                raise ValueError(f"{path}: missing required columns: {sorted(missing)}")
            yield from reader


def collect_candidates(params: dict, paths: list[Path]) -> tuple[list[Candidate], dict]:
    cfg = params["collect"]
    transforms = cfg["transforms"]
    required = set(transforms["required_columns"])
    aliases = {normalize_text(k): normalize_text(v) for k, v in transforms["genre_aliases"].items()}
    excluded = {normalize_text(v) for v in transforms["excluded_source_labels"]}
    separator = str(transforms["genre_separator"])
    excluded_source_ids = set(transforms.get("excluded_source_ids", ()))
    issues: Counter[str] = Counter()
    canonical_frequency: Counter[str] = Counter()
    parsed_rows: list[tuple[str, str, str, int, str, tuple[str, ...]]] = []
    rows_scanned = 0

    for row in _read_source_rows(paths, required):
        rows_scanned += 1
        title = normalize_text(row.get(transforms["title_column"]))
        description = normalize_text(row.get(transforms["description_column"]))
        raw_genres = normalize_text(row.get(transforms["genres_column"]))
        release_date_text = normalize_text(row.get(transforms["release_date_column"]))
        language = normalize_text(row.get(transforms["language_column"])).lower()
        parsed_date = parse_release_date(release_date_text)
        source_labels = tuple(
            sorted({normalize_text(label) for label in raw_genres.split(separator) if normalize_text(label)})
        )

        missing_description = not description
        missing_genres = not source_labels
        invalid_date = parsed_date is None
        wrong_type = bool(set(source_labels) & excluded)
        unknown_labels = tuple(
            label for label in source_labels if label not in aliases and label not in excluded
        )
        missing_identity = not title or not language

        if missing_description:
            issues["without_description"] += 1
        if missing_genres:
            issues["without_genres"] += 1
        if invalid_date:
            issues["invalid_date"] += 1
        if wrong_type:
            issues["wrong_type"] += 1
        if unknown_labels:
            issues["with_unknown_label"] += 1
            issues["unknown_labels_total"] += len(unknown_labels)
        if missing_identity:
            issues["missing_identity"] += 1

        if missing_description or missing_genres or invalid_date or wrong_type or missing_identity:
            continue
        canonical = tuple(sorted({aliases[label] for label in source_labels if label in aliases}))
        if not canonical:
            issues["unknown_label_only"] += 1
            continue
        canonical_frequency.update(canonical)
        parsed_rows.append(
            (title, release_date_text, language, parsed_date.year, description, canonical)
        )

    min_genre_count = int(transforms["min_genre_count"])
    candidates: list[Candidate] = []
    for title, release_date_text, language, release_year, description, genres in parsed_rows:
        frequent_genres = tuple(
            genre for genre in genres if canonical_frequency[genre] >= min_genre_count
        )
        if not frequent_genres:
            issues["rare_genre_only"] += 1
            continue
        source_id = stable_id(title, release_date_text, language, cfg)
        if source_id in excluded_source_ids:
            issues["audit_excluded"] += 1
            continue
        candidates.append(
            Candidate(
                source_id=source_id,
                release_year=release_year,
                language=language,
                description=description,
                genres=frequent_genres,
            )
        )

    # Resolve a hypothetical source-id collision independently of CSV order.
    # Identical source rows therefore cannot create duplicate JSONL ids.
    by_id: dict[str, Candidate] = {}
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.source_id,
            hashlib.sha256(
                (item.description + "\x1f" + "\x1f".join(item.genres)).encode("utf-8")
            ).hexdigest(),
        ),
    ):
        if candidate.source_id in by_id:
            issues["duplicate_source_id"] += 1
            continue
        by_id[candidate.source_id] = candidate

    for issue_name in (
        "without_description",
        "without_genres",
        "invalid_date",
        "wrong_type",
        "with_unknown_label",
        "unknown_labels_total",
        "unknown_label_only",
        "missing_identity",
        "rare_genre_only",
        "duplicate_source_id",
        "audit_excluded",
    ):
        issues.setdefault(issue_name, 0)

    audit = {
        "rows_scanned": rows_scanned,
        "quality_issues": dict(sorted(issues.items())),
        "canonical_frequency_before_version_sampling": dict(sorted(canonical_frequency.items())),
    }
    return list(by_id.values()), audit


def build_example(candidate: Candidate, prompts: list[str], topic_template: str) -> Example:
    return Example.model_validate(
        {
            "id": candidate.source_id,
            "topic": topic_template.format(release_year=candidate.release_year),
            "messages": [
                {"role": "system", "content": pick_prompt(candidate.source_id, prompts)},
                # This is exactly the normalized Overview. Title and the raw
                # Genre field are never concatenated into the user message.
                {"role": "user", "content": candidate.description},
                {"role": "assistant", "content": ", ".join(candidate.genres)},
            ],
        }
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    params = load_params()
    cfg = params["collect"]
    version = str(cfg["version"])
    target = int(cfg["target_clean_rows"][version])
    prompts = [normalize_text(prompt) for prompt in cfg["system_prompts"]]
    if len(prompts) < 3 or any(not prompt for prompt in prompts):
        raise ValueError("collect.system_prompts must contain at least three non-empty prompts")

    paths = ensure_source_snapshot(params)
    candidates, audit = collect_candidates(params, paths)
    ranked = sorted(
        candidates,
        key=lambda item: (hashlib.sha256(item.source_id.encode("utf-8")).hexdigest(), item.source_id),
    )
    if len(ranked) < target:
        raise RuntimeError(
            f"collect {version}: only {len(ranked)} eligible unique rows; target is {target}"
        )
    selected = ranked[:target]
    examples = [
        build_example(candidate, prompts, cfg["transforms"]["topic"]["template"])
        for candidate in selected
    ]

    output = Path(params["paths"]["raw"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".part")
    with temporary_output.open("w", encoding="utf-8", newline="\n") as sink:
        for example in examples:
            sink.write(dump(example) + "\n")
    os.replace(temporary_output, output)

    genre_distribution: Counter[str] = Counter()
    language_distribution: Counter[str] = Counter()
    year_distribution: Counter[str] = Counter()
    prompt_distribution: Counter[str] = Counter()
    for candidate, example in zip(selected, examples, strict=True):
        genre_distribution.update(candidate.genres)
        language_distribution[candidate.language] += 1
        year_distribution[str(candidate.release_year)] += 1
        prompt_distribution[example.messages[0].content] += 1

    source_cfg = cfg["source"]
    metrics = {
        "version": version,
        "source": {
            "kind": source_cfg["kind"],
            "dataset_id": source_cfg["dataset_id"],
            "revision": source_cfg["revision"],
            "file": source_cfg["file"],
            "sha256": source_cfg["sha256"],
            "files": len(paths),
        },
        "rows_scanned": audit["rows_scanned"],
        "rows_eligible_unique": len(ranked),
        "rows_written": len(examples),
        "rows_dropped_before_version_sampling": audit["rows_scanned"] - len(ranked),
        "rows_not_selected_for_version": len(ranked) - len(examples),
        "drop_reasons": audit["quality_issues"],
        "year_count": len(year_distribution),
        "language_count": len(language_distribution),
        "genre_count": len(genre_distribution),
        "genre_distribution": dict(sorted(genre_distribution.items())),
        "language_distribution": dict(sorted(language_distribution.items())),
        "year_distribution": dict(sorted(year_distribution.items())),
        "multi_label_examples": sum(len(candidate.genres) > 1 for candidate in selected),
        "system_prompt_variants": len(prompt_distribution),
        "system_prompt_distribution": dict(sorted(prompt_distribution.items())),
    }
    write_json(Path(params["paths"]["metrics_collect"]), metrics)
    print(
        f"collect: {version}, scanned {metrics['rows_scanned']}, "
        f"eligible {metrics['rows_eligible_unique']}, wrote {metrics['rows_written']}, "
        f"years {metrics['year_count']}, languages {metrics['language_count']}, "
        f"genres {metrics['genre_count']}, prompts {metrics['system_prompt_variants']} "
        f"-> {output}"
    )


if __name__ == "__main__":
    main()
