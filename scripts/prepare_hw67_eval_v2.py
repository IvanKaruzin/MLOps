"""Prepare a random stratified free-form movie-genre evaluation set.

Sampling never reads model outputs. The pinned source supplies genre labels;
exact source agreement is verified separately from subjective label quality.
The HW4 tensors and model weights are read, never rewritten.
"""

import argparse
from collections import Counter
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_hw67_eval import (
    REVIEW_EXCLUSIONS, SOURCE_REVISION, SOURCE_URL, canonical_hash, digest,
    make_archive, read_rows, source_parts, verify_tensor_links, write_json, write_rows,
)

VERSION = "movie-genres-free-v2"
SAMPLE_SEED = 20261025
ORDER_SEED = 20261026
PER_STRATUM = 15
ERAS = ("before2000", "since2000")
CARDINALITIES = ("single_genre", "two_genres", "three_plus_genres")
BEHAVIOR_SOURCE_ID = "movie_d3999a141d72d9f56d2018dd609315ea94015f43e02bb2bb84ccc1e96277135a"
SOURCE_REVIEW_NOTES = {
    "movie_9119e304fe4986f1d3a575a64aa3a3f633a6daf67e2db084c94b864873bfd1c7":
        "The synopsis discusses Scream without explicitly saying this is a documentary; the source metadata determines that label.",
    "movie_eacedd42274d7a7ec8c1a5b6a38185896acc7c32dcb5150d83115d6abcb1243c":
        "The synopsis includes time travel but source labels list animation and romance only; plausible extra genres can fail exact-set accuracy.",
    "movie_cf74de0ce930a7fcad661a1ef0074b474389d46234e6c7d88bfc7ca11b2f83b9":
        "The single source label drama does not fully specify the tone of this synopsis; treat exact-set agreement as source-taxonomy agreement.",
}


def stratum(record):
    year = int(record["topic"].removeprefix("release_year:"))
    if record["topic"] != f"release_year:{year}":
        raise ValueError("Expected the original release-year split grouping")
    count = len(source_parts(record)[1])
    return (ERAS[0] if year < 2000 else ERAS[1], CARDINALITIES[min(count, 3) - 1])


def select_examples(test):
    eligible = [row for row in test if row["id"] not in REVIEW_EXCLUSIONS]
    support = Counter(stratum(row) for row in eligible)
    rng = random.Random(SAMPLE_SEED)
    selected = []
    for era in ERAS:
        for cardinality in CARDINALITIES:
            pool = sorted((row for row in eligible if stratum(row) == (era, cardinality)),
                          key=lambda row: row["id"])
            if len(pool) < PER_STRATUM:
                raise ValueError(f"Insufficient support for {(era, cardinality)}")
            rng.shuffle(pool)
            selected.extend(pool[:PER_STRATUM])
    random.Random(ORDER_SEED).shuffle(selected)
    return selected, support, eligible


def normalized(value):
    return " ".join(value.casefold().split())


def make_behavioral(golden, choices):
    source = next(row for row in golden if row["source_id"] == BEHAVIOR_SOURCE_ID)
    behavior = [{**source, "id": "behavior-format", "type": "format"}]
    behavior.append({
        **source, "id": "behavior-paraphrase", "type": "paraphrase", "paired_with": "behavior-format",
        "input": "Jed Cooper escapes death when a lynch mob tries to hang him. Determined to take revenge on the nine men who strung him up, he resumes his old job as a lawman and begins dispensing his own justice in the Wild West.",
        "source": "Author-created paraphrase of the paired source synopsis; same plot facts and genre annotation, written before model generation",
    })
    for identifier, prompt in (
        ("geography", "Назови столицу Франции."),
        ("arithmetic", "Вычисли произведение 17 и 23."),
    ):
        behavior.append({
            "id": f"behavior-refusal-{identifier}", "group_id": f"unsupported-{identifier}",
            "task": "multilabel", "input": prompt, "expected": "REFUSE", "choices": choices,
            "slices": ["unsupported_domain", f"unsupported_{identifier}"], "type": "refusal",
            "source": "Author-created out-of-domain request; movie-genre-only boundary fixed before generation",
        })
    behavior.append({
        "id": "behavior-format-known", "group_id": "author-animation-technique",
        "task": "multilabel", "input": "A short film made entirely of hand-drawn frames shows simple geometric shapes moving. There are no filmed actors, interviews, historical events or narrative plot.",
        "expected": "анимация", "choices": choices, "slices": ["obvious_animation"], "type": "format",
        "source": "Author-created definition-level animation case; the described visual production technique supplies its label",
    })
    return behavior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    import yaml
    from evaluation.contract import validate_golden
    from src.collect import collect_candidates

    params = yaml.safe_load((ROOT / args.params).read_text())
    source_file = ROOT / "sources/movies-dataset/9000plus.csv"
    source_sha256 = digest(source_file)
    if source_sha256 != params["collect"]["source"]["sha256"]:
        raise ValueError("Raw public source differs from the pinned snapshot")
    splits = {name: read_rows(ROOT / f"data/{name}.jsonl") for name in ("train", "val", "test")}
    if {name: len(rows) for name, rows in splits.items()} != {"train": 4800, "val": 599, "test": 599}:
        raise ValueError("Expected the full original HW3 split, without resplitting")
    split_proof = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        ids = {row["id"] for row in splits[left]} & {row["id"] for row in splits[right]}
        groups = {row["topic"] for row in splits[left]} & {row["topic"] for row in splits[right]}
        if ids or groups:
            raise ValueError(f"Source id/group overlap: {left}/{right}")
        split_proof[f"{left}/{right}"] = dict(source_id_intersections=0, source_group_intersections=0)

    selected, support, eligible = select_examples(splits["test"])
    taxonomy = sorted(set(params["collect"]["transforms"]["genre_aliases"].values()))
    if len(taxonomy) != 18:
        raise ValueError("Expected the original 18-genre taxonomy")
    choices = taxonomy + ["REFUSE"]
    candidates, _ = collect_candidates(params, [source_file])
    source_by_id = {candidate.source_id: candidate for candidate in candidates}
    golden, annotations = [], []
    for index, record in enumerate(selected):
        description, genres = source_parts(record)
        source = source_by_id[record["id"]]
        year = int(record["topic"].split(":")[1])
        if (source.description, source.genres, source.release_year) != (description, genres, year):
            raise ValueError(f"Processed row differs from pinned CSV metadata: {record['id']}")
        if not set(genres) <= set(taxonomy):
            raise ValueError("Source genre outside the fixed taxonomy")
        era, cardinality = stratum(record)
        item = {
            "id": f"movie-free-{index + 1:03d}", "group_id": record["topic"], "task": "multilabel",
            "input": description, "expected": ", ".join(genres), "choices": choices,
            "slices": [era, cardinality, "long_description" if len(description) > 400 else "short_description"],
            "source": f"{SOURCE_URL}; held-out test; source_id={record['id']}",
            "source_id": record["id"], "release_year": year, "stratum": f"{era}__{cardinality}",
        }
        golden.append(item)
        annotations.append({
            "id": item["id"], "source_id": record["id"], "source_record_sha256": canonical_hash(record),
            "source_description": description, "source_genres": list(genres), "expected": item["expected"],
            "release_year": year, "stratum": item["stratum"],
            "source_consistency": "Exact description, genre-set and year equality with the pinned CSV via collect_candidates",
            "review": "AI assistant inspected the synopsis and source labels before model generation; no independent human relabeling",
            "review_note": SOURCE_REVIEW_NOTES.get(record["id"], "Source annotation retained; source genres may be incomplete or subjective."),
        })

    behavioral = make_behavioral(golden, choices)
    training = [{
        "id": row["id"], "group_id": row["topic"], "input": source_parts(row)[0],
        "source_record_sha256": canonical_hash(row),
    } for row in splits["train"]]
    leakage = validate_golden(golden, training, behavioral)
    source_plot_leakage = {}
    for split in ("train", "val"):
        plots = {normalized(source_parts(record)[0]) for record in splits[split]}
        intersections = [item["id"] for item in golden + behavioral if normalized(item["input"]) in plots]
        if intersections:
            raise ValueError(f"Evaluation plot intersects {split}: {intersections}")
        source_plot_leakage[split] = dict(source_records_checked=len(splits[split]),
                                         evaluation_items_checked=len(golden + behavioral),
                                         normalized_original_plot_intersections=0)
    tensor_links = verify_tensor_links(params, splits)
    directory = ROOT / "data/eval"
    directory.mkdir(parents=True, exist_ok=True)
    write_rows(directory / "golden.jsonl", golden)
    write_rows(directory / "training-index.jsonl", training)
    write_rows(directory / "behavioral.jsonl", behavioral)
    manifest = {
        "version": VERSION, "task": "multilabel", "source_dataset": "Pablinho/movies-dataset",
        "source_revision": SOURCE_REVISION, "source_license": "CC0-1.0", "source_url": SOURCE_URL,
        "raw_source_sha256": source_sha256,
        "source_sha256": {name: digest(ROOT / f"data/{name}.jsonl") for name in splits},
        "source_counts": {name: len(rows) for name, rows in splits.items()},
        "taxonomy": taxonomy, "refusal_label": "REFUSE", "selected_source_ids": [row["id"] for row in selected],
        "sampling": {
            "method": "Uniform sampling without replacement within era x genre-cardinality cells; independently shuffled final order",
            "seed": SAMPLE_SEED, "final_order_seed": ORDER_SEED, "per_stratum": PER_STRATUM,
            "within_cell_input_order": "source_id ascending", "stratum_iteration": [f"{era}__{card}" for era in ERAS for card in CARDINALITIES],
            "stratum_support": {f"{era}__{card}": support[(era, card)] for era in ERAS for card in CARDINALITIES},
            "selected_strata": dict(Counter(row["stratum"] for row in golden)),
            "eligible_test_count": len(eligible),
            "year_support": dict(sorted(Counter(int(row["topic"].split(":")[1]) for row in eligible).items())),
            "selected_year_counts": dict(sorted(Counter(row["release_year"] for row in golden).items())),
            "before2000_single_genre_is_census": support[("before2000", "single_genre")] == PER_STRATUM,
            "model_outputs_used": False,
        },
        "slice_counts": dict(Counter(name for row in golden for name in row["slices"])),
        "label_support": dict(sorted(Counter(genre for row in selected for genre in source_parts(row)[1]).items())),
        "leakage": leakage, "source_plot_leakage": source_plot_leakage, "source_split_proof": split_proof,
        "tokenized_source_links": tensor_links, "pre_generation_source_review_exclusions": REVIEW_EXCLUSIONS,
        "annotation_review": "All 90 selected descriptions and source genre annotations inspected by an AI assistant before outputs; exact pinned-CSV consistency verified for all 90. No claim of independent human relabeling.",
        "behavioral_paraphrase_review": {
            "source_id": BEHAVIOR_SOURCE_ID, "paired_ids": ["behavior-format", "behavior-paraphrase"],
            "preserved_facts": ["Jed Cooper survives an attempted lynching", "returns to his former lawman role", "pursues the nine responsible men for revenge", "Wild West setting"],
            "changed": "The actual plot wording and sentence structure, not just the evaluation instruction",
        },
        "limitations": [
            "Exact set accuracy measures agreement with a possibly incomplete and subjective source taxonomy. Plausible extra genres can be counted wrong.",
            "The balanced 6-cell sample is not prevalence-weighted test-set accuracy; the oldest single-genre cell is a census of 15 items.",
            "This formerly held-out test subset is now a development golden set, not an independent final audit.",
            "Exact/group comparisons do not rule out all semantic paraphrases, franchise overlap or model pretraining contamination.",
            "Refusal and strict JSON were not trained as auxiliary targets. An untrained behavior may fail; labels and thresholds must not be changed to hide that failure.",
        ],
        "preparation_code_sha256": {name: digest(ROOT / name) for name in (
            "scripts/prepare_hw67_eval_v2.py", "scripts/prepare_hw67_eval.py", "src/collect.py", "src/tokenize_data.py", "src/prompt.py")},
        "annotations": annotations,
        "file_sha256": {path.name: digest(path) for path in sorted(directory.glob("*.jsonl"))},
    }
    write_json(directory / "manifest.json", manifest)
    archive = make_archive(directory)
    print(json.dumps(dict(version=VERSION, golden=len(golden), behavioral=len(behavioral),
                          training_index=len(training), sampling=manifest["sampling"],
                          slices=manifest["slice_counts"], manifest_sha256=digest(directory / "manifest.json"),
                          archive_sha256=digest(archive)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
