"""Build the fixed movie-genre evaluation set from the held-out HW3 split.

Labels are inherited from the pinned public source, not supplied by a judge.
The default run also re-encodes every train/val record and checks the actual
HW4 tensor files. It never changes those files or selects a model candidate.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VERSION = "movie-genres-mcq-v1"
SOURCE_REVISION = "97da8cd79f0c0ee06c2f45dd30a878967d6d87a4"
SOURCE_URL = f"https://huggingface.co/datasets/Pablinho/movies-dataset/tree/{SOURCE_REVISION}"
SLICE_NAMES = ("single_genre", "two_genres", "three_plus_genres")
REVIEW_EXCLUSIONS = {
    "movie_88c6628229ecf1533a1546fc79ad18165467c2b848bb70e13c38f90990ee3a44":
        "Description explicitly calls the film a romantic comedy, but source labels omit comedy; contradictory MCQ annotation.",
    "movie_00a04214fa9e6059ab2953acb5ab5af37715ba0e64e7a709037a53e237aabf2c":
        "Poorly edited source synopsis with broken referents; unsuitable for a checked golden question.",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate ids in {path}")
    return rows


def canonical_hash(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def source_parts(record):
    messages = record["messages"]
    users = [message["content"] for message in messages if message["role"] == "user"]
    if len(users) != 1 or messages[-1]["role"] != "assistant":
        raise ValueError(f"Unexpected source conversation: {record['id']}")
    genres = tuple(sorted(part.strip() for part in messages[-1]["content"].split(",")))
    if len(genres) != len(set(genres)) or any(not genre for genre in genres):
        raise ValueError(f"Invalid genre annotation: {record['id']}")
    return users[0], genres


def stable_key(record):
    return hashlib.sha256((VERSION + "\0" + record["id"]).encode()).hexdigest()


def genre_slice(genres):
    return SLICE_NAMES[min(len(genres), 3) - 1]


def choices_for(record, vocabulary, answer_position):
    _, genres = source_parts(record)
    absent = sorted(set(vocabulary) - set(genres), key=lambda genre: canonical_hash([record["id"], genre]))
    # Every distractor has the same cardinality. A label-count shortcut cannot
    # identify the correct choice; each false set replaces one source genre.
    distractors = []
    for index, replacement in enumerate(absent[:3]):
        remove = genres[index % len(genres)]
        distractors.append(tuple(sorted((set(genres) - {remove}) | {replacement})))
    if len(set(distractors + [genres])) != 4:
        raise ValueError("Choices must contain exactly one source-matching answer")
    options = list(distractors)
    options.insert(answer_position, genres)
    return {label: ", ".join(option) for label, option in zip("ABCD", options)}


def question(description, options, paraphrase=False):
    instruction = (
        "Сопоставь описание фильма с одним из полных наборов жанров. Выбери вариант A, B, C или D."
        if paraphrase else
        "Какой полный набор жанров соответствует описанию фильма? Выбери один вариант A, B, C или D."
    )
    return instruction + "\nОписание: " + description + "\nВарианты:\n" + "\n".join(
        f"{label}: {genres}" for label, genres in options.items()
    )


def verify_tensor_links(params, splits):
    import torch
    from transformers import AutoTokenizer
    from src.tokenize_data import encode_example

    tokenizer = AutoTokenizer.from_pretrained(params["model"]["name"], local_files_only=True)
    tokenizer.padding_side = params["tokenize"]["padding_side"]
    result = {}
    for split in ("train", "val"):
        path = ROOT / "data/tokenized" / f"{split}.pt"
        package = torch.load(path, map_location="cpu", weights_only=True)
        if package["model"] != params["model"]["name"]:
            raise ValueError("Tensor model differs from source tokenizer")
        encoded = []
        for row in splits[split]:
            example = encode_example(tokenizer, row, params, package["max_seq_len"])
            meta = example.pop("_meta")
            if meta["supervised"]:
                encoded.append(example)
        if encoded != package["examples"]:
            raise ValueError(f"Full re-encoding differs from {path}")
        if len(encoded) != len(splits[split]):
            raise ValueError("The full training index requires every source row to survive tokenization")
        result[split] = {
            "tensor_path": str(path.relative_to(ROOT)),
            "tensor_sha256": digest(path),
            "source_records": len(splits[split]),
            "tensor_records": len(encoded),
            "ordered_ids_sha256": canonical_hash([row["id"] for row in encoded]),
            "all_input_ids_attention_masks_labels_equal": True,
            "check": "full re-encoding with src.tokenize_data.encode_example; no sampling",
        }
    return result


def make_archive(directory):
    target = ROOT / "docs/datasets/hw67-evaluation.tar.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Fixed archive metadata allows byte-for-byte reproduction.
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in sorted(directory.glob("*.json*")):
            data = path.read_bytes()
            info = tarfile.TarInfo(str(path.relative_to(ROOT)))
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    target.write_bytes(gzip.compress(buffer.getvalue(), mtime=0))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    import yaml
    from evaluation.contract import validate_golden

    params = yaml.safe_load((ROOT / args.params).read_text())
    source_file = ROOT / "sources/movies-dataset/9000plus.csv"
    source_sha256 = digest(source_file)
    if source_sha256 != params["collect"]["source"]["sha256"]:
        raise ValueError("Public source snapshot checksum differs from the pinned HW3 source")
    splits = {name: read_rows(ROOT / f"data/{name}.jsonl") for name in ("train", "val", "test")}
    if len(splits["train"]) != 4800:
        raise ValueError("Expected the complete 4800-row HW4 training source")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if {r["id"] for r in splits[left]} & {r["id"] for r in splits[right]}:
            raise ValueError(f"Source ids overlap: {left}/{right}")
        if {r["topic"] for r in splits[left]} & {r["topic"] for r in splits[right]}:
            raise ValueError(f"Source split groups overlap: {left}/{right}")
    vocabulary = sorted({genre for rows in splits.values() for row in rows for genre in source_parts(row)[1]})
    selected = []
    for name in SLICE_NAMES:
        pool = [row for row in splits["test"] if genre_slice(source_parts(row)[1]) == name
                and row["id"] not in REVIEW_EXCLUSIONS]
        if len(pool) < 20:
            raise ValueError(f"Insufficient held-out examples in {name}")
        selected.extend(sorted(pool, key=stable_key)[:20])
    selected.sort(key=stable_key)
    # Compare original plots, not the MCQ wrapper: prefixes/options otherwise
    # hide an exact source-text leak from a direct train-input comparison.
    normalize = lambda text: " ".join(text.casefold().split())
    selected_plots = {normalize(source_parts(record)[0]) for record in selected}
    source_plot_leakage = {}
    for split in ("train", "val"):
        source_plots = {normalize(source_parts(record)[0]) for record in splits[split]}
        intersections = selected_plots & source_plots
        if intersections:
            raise ValueError(f"Original plot overlap between golden and {split}")
        source_plot_leakage[split] = {
            "source_records_checked": len(splits[split]),
            "golden_source_plots_checked": len(selected_plots),
            "normalized_original_plot_intersections": len(intersections),
        }
    golden = []
    annotations = []
    for index, record in enumerate(selected):
        description, genres = source_parts(record)
        options = choices_for(record, vocabulary, index % 4)
        answer = "ABCD"[index % 4]
        if options[answer] != ", ".join(genres):
            raise ValueError("Transformed annotation differs from source")
        golden.append({
            "id": f"movie-mcq-{index + 1:03d}", "group_id": record["topic"],
            "input": question(description, options), "expected": answer,
            "choices": list(options),
            "slices": [genre_slice(genres), "long_description" if len(description) > 400 else "short_description"],
            "source": f"{SOURCE_URL}; test split; source_id={record['id']}",
        })
        annotations.append({
            "id": golden[-1]["id"], "source_id": record["id"],
            "source_record_sha256": canonical_hash(record), "source_description": description,
            "source_genres": list(genres), "options": options, "expected": answer,
            "verification": "source label equality, unique choices, equal cardinality, disjoint source ids/groups",
        })

    training = [{
        "id": row["id"], "group_id": row["topic"], "input": source_parts(row)[0],
        "source_record_sha256": canonical_hash(row),
    } for row in splits["train"]]
    # Isolate JSON/permutation behavior on a plot with an explicit horror cue,
    # instead of a source example with more subjective social genre labels.
    behavior_index = next(index for index, row in enumerate(annotations)
                          if "haunted house" in row["source_description"])
    first = golden[behavior_index]
    behavioral = [{**first, "id": "behavior-format", "type": "format"}]
    # Keep the same plot, but rephrase the instruction and rotate the options:
    # the semantic answer stays fixed while its letter must change.
    annotation = annotations[behavior_index]
    rotated = dict(zip("ABCD", list(annotation["options"].values())[1:] + list(annotation["options"].values())[:1]))
    rotated_answer = next(label for label, value in rotated.items() if value == ", ".join(annotation["source_genres"]))
    behavioral.append({
        **first, "id": "behavior-paraphrase", "type": "paraphrase",
        "paired_with": "behavior-format",
        "input": question(annotation["source_description"], rotated, paraphrase=True),
        "expected": rotated_answer,
    })
    behavioral.append({
        "id": "behavior-refusal", "group_id": "unsupported-geography",
        "input": "Назови столицу Франции. Варианты: A: Париж; B: Рим; C: Берлин; D: Мадрид. Это вопрос по географии, не описание фильма.",
        "expected": "REFUSE", "choices": ["A", "B", "C", "D", "REFUSE"],
        "slices": ["unsupported_domain"], "type": "refusal",
        "source": "Author-created out-of-domain request; movie-only task boundary fixed before generation",
    })
    behavioral.append({
        "id": "behavior-format-known", "group_id": "author-animated-film",
        "input": "Описание фильма: рисованный короткометражный фильм, в котором оживлённые персонажи нарисованы покадрово. Выбери жанровую метку. Варианты: A: документальный; B: анимация; C: вестерн; D: военный.",
        "expected": "B", "choices": ["A", "B", "C", "D"],
        "slices": ["obvious_animation"], "type": "format",
        "source": "Author-created definition-level animation case; annotation checked by explicit description",
    })
    leakage = validate_golden(golden, training, behavioral)
    tensor_links = verify_tensor_links(params, splits)
    directory = ROOT / "data/eval"
    directory.mkdir(parents=True, exist_ok=True)
    write_rows(directory / "golden.jsonl", golden)
    write_rows(directory / "training-index.jsonl", training)
    write_rows(directory / "behavioral.jsonl", behavioral)
    manifest = {
        "version": VERSION, "source_dataset": "Pablinho/movies-dataset",
        "source_revision": SOURCE_REVISION, "source_license": "CC0-1.0",
        "source_url": SOURCE_URL,
        "raw_source_sha256": source_sha256,
        "source_sha256": {name: digest(ROOT / f"data/{name}.jsonl") for name in splits},
        "source_counts": {name: len(rows) for name, rows in splits.items()},
        "selection": "20 smallest SHA256(version + NUL + source_id) per label-cardinality slice; then global stable ordering",
        "selected_label_counts": dict(Counter(row["expected"] for row in golden)),
        "slice_counts": dict(Counter(name for row in golden for name in row["slices"])),
        "leakage": leakage, "tokenized_source_links": tensor_links,
        "source_plot_leakage": source_plot_leakage,
        "annotation_review": "An AI assistant reviewed all selected descriptions and options before generation; labels checked against pinned source; no claim of independent human relabeling",
        "pre_generation_source_review_exclusions": REVIEW_EXCLUSIONS,
        "limitations": [
            "Source genre labels can be incomplete or subjective; MCQ tests agreement with this source taxonomy, not an independent cinematic truth.",
            "The held-out test subset becomes a development golden set and must not be called an independent final audit.",
            "Exact/group checks do not rule out all paraphrases, franchise overlap, or pretraining contamination.",
            "Slice-balanced selection and four-way choices change the distribution and task format relative to free-form multilabel training.",
        ],
        "annotations": annotations,
        "file_sha256": {path.name: digest(path) for path in sorted(directory.glob("*.jsonl"))},
    }
    write_json(directory / "manifest.json", manifest)
    archive = make_archive(directory)
    print(json.dumps({"golden": len(golden), "behavioral": len(behavioral), "training_index": len(training),
                      "slices": manifest["slice_counts"], "leakage": leakage,
                      "archive": str(archive.relative_to(ROOT)), "archive_sha256": digest(archive)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
