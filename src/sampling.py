"""Deterministic stratified samples linked to complete HW4 source records."""

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


METHOD = "stratified_shuffle"
STRATA = tuple(f"{era}__{cardinality}" for era in ("before2000", "since2000")
               for cardinality in ("single_genre", "two_genres", "three_plus_genres"))


def source_metadata(row):
    """Read the release year and number of canonical labels from a source row."""
    if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
        raise ValueError("Source records require nonempty string ids")
    topic = row.get("topic")
    match = re.fullmatch(r"release_year:([0-9]{4})", topic) if isinstance(topic, str) else None
    if match is None or not 1800 <= int(match.group(1)) <= 2200:
        raise ValueError(f"{row['id']}: invalid release_year topic")
    messages = row.get("messages")
    if (not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict)
            or messages[-1].get("role") != "assistant"
            or not isinstance(messages[-1].get("content"), str)):
        raise ValueError(f"{row['id']}: missing assistant genre labels")
    labels = [label.strip().casefold() for label in messages[-1]["content"].split(",")]
    if any(not label for label in labels) or len(labels) != len(set(labels)):
        raise ValueError(f"{row['id']}: empty or duplicate genre labels")
    year, count = int(match.group(1)), len(labels)
    era = "before2000" if year < 2000 else "since2000"
    cardinality = {1: "single_genre", 2: "two_genres"}.get(count, "three_plus_genres")
    return {"id": row["id"], "release_year": year, "genre_cardinality": count,
            "stratum": f"{era}__{cardinality}"}


def sample_split(blob, source_path, limit, seed, split_name, method=METHOD):
    """Return a shuffled balanced sample and its exact ordered provenance.

    Sorting by id before shuffling makes alignment independent of serialized
    source/tensor row order. Each stratum has its own random stream, so the
    smaller control sample is contained in a larger sample with the same seed.
    """
    if method != METHOD:
        raise ValueError(f"Unsupported sampling method: {method!r}")
    if type(seed) is not int or seed < 0:
        raise ValueError("Sampling seed must be a nonnegative integer")
    if type(limit) is not int or limit < len(STRATA):
        raise ValueError("Sampling limit must cover all six strata")
    if split_name not in ("train", "val"):
        raise ValueError("Sampling split must be train or val")
    if not isinstance(blob, dict) or not isinstance(blob.get("examples"), list) or not blob["examples"]:
        raise ValueError("Tensor input requires a nonempty examples list")
    source = Path(source_path)
    if not source.is_file():
        raise ValueError(f"Source JSONL is not a file: {source}")
    raw = source.read_bytes()
    try:
        records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid source JSONL: {source}") from exc
    metadata = {}
    for row in records:
        item = source_metadata(row)
        if item["id"] in metadata:
            raise ValueError(f"Duplicate source id: {item['id']}")
        metadata[item["id"]] = item
    tensors = {}
    for example in blob["examples"]:
        if (not isinstance(example, dict) or not isinstance(example.get("id"), str)
                or not example["id"].strip() or example["id"] in tensors):
            raise ValueError("Tensor examples require unique nonempty string ids")
        tensors[example["id"]] = example
    if set(metadata) != set(tensors):
        raise ValueError("Source and tensor ids must match exactly for the complete split")
    if limit > len(tensors):
        raise ValueError("Sampling limit exceeds the complete split")
    strata = {name: [] for name in STRATA}
    for identity, item in metadata.items():
        strata[item["stratum"]].append(identity)
    base, remainder = divmod(limit, len(STRATA))
    extra_order = list(STRATA)
    random.Random(f"{seed}:{split_name}:quotas").shuffle(extra_order)
    extra = set(extra_order[:remainder])
    quotas = {name: base + (name in extra) for name in STRATA}
    selected_ids = []
    for name in STRATA:
        identities = sorted(strata[name])
        if len(identities) < quotas[name]:
            raise ValueError(f"Insufficient {split_name} stratum {name}: {len(identities)} < {quotas[name]}")
        random.Random(f"{seed}:{split_name}:{name}").shuffle(identities)
        selected_ids.extend(identities[:quotas[name]])
    random.Random(f"{seed}:{split_name}:order").shuffle(selected_ids)
    selected = [tensors[identity] for identity in selected_ids]
    selected_metadata = [metadata[identity] for identity in selected_ids]
    manifest = {
        "method": method, "seed": seed, "split": split_name,
        "source_path": str(source), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_rows": len(records), "tensor_rows": len(tensors), "selected_rows": len(selected),
        "ids": selected_ids, "selected_records": selected_metadata,
        "strata_counts": quotas,
        "source_strata_counts": {name: len(strata[name]) for name in STRATA},
        "year_counts": dict(sorted(Counter(str(item["release_year"]) for item in selected_metadata).items())),
        "genre_cardinality_counts": dict(sorted(Counter(str(item["genre_cardinality"])
                                                       for item in selected_metadata).items())),
    }
    return selected, manifest


def verify_tensor_sources(params):
    """Re-encode every HW4 source record before any candidate can train."""
    import yaml
    from transformers import AutoTokenizer
    from src.data import load_split
    from src.tokenize_data import encode_example

    paths = {key: Path(params["data"][key]) for key in
             ("train", "val", "source_train", "source_val", "tokenization_params")}
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    before = {key: digest(path) for key, path in paths.items()}
    tokenization = yaml.safe_load(paths["tokenization_params"].read_text(encoding="utf-8"))
    if tokenization["model"]["name"] != params["model"]["name"]:
        raise ValueError("Tokenization configuration and experiment model differ")
    for split in ("train", "val"):
        if (Path(tokenization["data"][f"{split}_jsonl"]).resolve() != paths[f"source_{split}"].resolve()
                or Path(tokenization["data"][split]).resolve() != paths[split].resolve()):
            raise ValueError("Experiment paths differ from the original tokenization configuration")
    tokenizer = AutoTokenizer.from_pretrained(params["model"]["name"],
                                              revision=params["model"]["revision"], local_files_only=True)
    tokenizer.padding_side = tokenization["tokenize"]["padding_side"]
    proof = {"passed": True, "tokenization_params_sha256": before["tokenization_params"],
             "tokenizer": {"name": params["model"]["name"], "revision": params["model"]["revision"]},
             "check": "full src.tokenize_data.encode_example re-encoding; no sampled rows", "splits": {}}
    complete_ids = {}
    for split in ("train", "val"):
        source = [json.loads(line) for line in paths[f"source_{split}"].read_text(encoding="utf-8").splitlines() if line.strip()]
        blob = load_split(paths[split])
        if (blob.get("model") != tokenization["model"]["name"]
                or blob.get("max_seq_len") != tokenization["tokenize"]["max_seq_len"]
                or blob.get("padding_side") != tokenizer.padding_side
                or blob["pad_token_id"] != tokenizer.pad_token_id):
            raise ValueError("Tensor metadata differs from the original tokenization contract")
        reconstructed = {}
        for record in source:
            identity = record.get("id")
            if not isinstance(identity, str) or not identity or identity in reconstructed:
                raise ValueError("Source ids must be unique nonempty strings")
            encoded = encode_example(tokenizer, record, tokenization, blob["max_seq_len"])
            meta = encoded.pop("_meta")
            if not meta["supervised"]:
                raise ValueError("A full source record has no surviving supervised token")
            reconstructed[identity] = encoded
        tensors = {row["id"]: row for row in blob["examples"]}
        if len(tensors) != len(blob["examples"]) or reconstructed != tensors:
            raise ValueError(f"Full source plots/labels re-encoding differs from {paths[split]}")
        complete_ids[split] = set(tensors)
        proof["splits"][split] = {
            "source_path": str(paths[f"source_{split}"]), "tensor_path": str(paths[split]),
            "source_sha256": before[f"source_{split}"], "tensor_sha256": before[split],
            "source_records": len(source), "tensor_records": len(tensors),
            "all_input_ids_attention_masks_labels_equal": True,
        }
    if complete_ids["train"] & complete_ids["val"]:
        raise ValueError("Train and val source ids overlap")
    if before != {key: digest(path) for key, path in paths.items()}:
        raise ValueError("Source or tensor bytes changed during the full re-encoding check")
    return proof
