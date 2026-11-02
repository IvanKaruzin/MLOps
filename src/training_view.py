"""Deterministic SFT views derived from verified immutable HW4 source data."""

import copy
import hashlib
import json
import re
from pathlib import Path

import yaml


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def normalized_input(value):
    return " ".join(re.findall(r"\w+", value.casefold()))


def load_auxiliary(params):
    """Read only training/development data; never open a holdout manifest path."""
    path = params["data"].get("auxiliary")
    if not path:
        return {"train": [], "val": []}, None
    if params["train"].get("response_format") != "json_genres":
        raise ValueError("Auxiliary data requires the JSON training view")
    raw = Path(path).read_bytes()
    bundle = json.loads(raw)
    if (not isinstance(bundle, dict) or type(bundle.get("version")) not in (str, int)
            or not isinstance(bundle.get("train"), list) or not isinstance(bundle.get("dev"), list)
            or "sealed_final" in bundle):
        raise ValueError("Auxiliary bundle requires version/train/dev and no sealed holdout records")
    originals = {}
    for split in ("train", "val"):
        originals[split] = [json.loads(line) for line in Path(params["data"][f"source_{split}"])
                           .read_text(encoding="utf-8").splitlines() if line.strip()]
    original_ids = {row["id"] for rows in originals.values() for row in rows}
    original_texts = {normalized_input(message["content"]) for rows in originals.values() for row in rows
                      for message in row["messages"] if message["role"] == "user"}
    original_groups = {split: {row["topic"] for row in rows} for split, rows in originals.items()}
    vocabulary = {label.strip().casefold() for rows in originals.values() for row in rows
                  for label in row["messages"][-1]["content"].split(",")}
    seen_ids, seen_texts, groups = set(original_ids), set(original_texts), {}
    prepared = {}
    for split, key in (("train", "train"), ("val", "dev")):
        prepared[split], groups[split] = [], set()
        for row in bundle[key]:
            if not isinstance(row, dict) or any(not isinstance(row.get(field), str) or not row[field].strip()
                                                for field in ("id", "input", "expected", "group_id", "kind", "source")):
                raise ValueError("Auxiliary rows require nonempty id/input/expected/group_id/kind/source")
            if row["kind"] not in ("genre", "refusal") or row["source"] not in ("pinned_source", "author_synthetic"):
                raise ValueError("Unknown auxiliary kind or source")
            answer = row["expected"]
            if row["kind"] == "refusal":
                if answer != "REFUSE":
                    raise ValueError("Refusal targets must be uppercase REFUSE")
            else:
                parts = answer.split(", ")
                if answer != ", ".join(sorted(set(parts))) or any(part not in vocabulary for part in parts):
                    raise ValueError("Auxiliary genre targets must be canonical known genre combinations")
            text = normalized_input(row["input"])
            if row["id"] in seen_ids or not text or text in seen_texts:
                raise ValueError("Auxiliary ids and normalized inputs must be unique and separate from original data")
            opposite = "val" if split == "train" else "train"
            if row["group_id"] in original_groups[opposite]:
                raise ValueError("Auxiliary group leaks into the opposite original split")
            seen_ids.add(row["id"])
            seen_texts.add(text)
            groups[split].add(row["group_id"])
            prepared[split].append(copy.deepcopy(row))
    if groups["train"] & groups["val"]:
        raise ValueError("Auxiliary training and development families/groups overlap")
    return prepared, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                      "version": bundle["version"], "train_rows": len(prepared["train"]),
                      "dev_rows": len(prepared["val"])}


def auxiliary_metadata(row):
    parts = [] if row["expected"] == "REFUSE" else row["expected"].split(", ")
    return {"id": row["id"], "kind": row["kind"], "source": row["source"],
            "group_id": row["group_id"], "stratum": "aux__" + row["kind"],
            "release_year": row.get("release_year"), "genre_cardinality": len(parts)}


def auxiliary_record(row, system):
    return {"id": row["id"], "messages": [{"role": "system", "content": system},
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": json.dumps({"answer": row["expected"]}, ensure_ascii=False)}]}


def canonical_answer(record):
    messages = record.get("messages", [])
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Training source must end with assistant genre labels")
    genres = [genre.strip().casefold() for genre in messages[-1]["content"].split(",")]
    if not genres or any(not genre for genre in genres) or len(genres) != len(set(genres)):
        raise ValueError("Source contains empty or duplicate genre labels")
    return json.dumps({"answer": ", ".join(sorted(genres))}, ensure_ascii=False)


def json_record(record, system):
    users = [message for message in record["messages"] if message["role"] == "user"]
    if len(users) != 1 or not isinstance(users[0].get("content"), str) or not users[0]["content"].strip():
        raise ValueError("JSON training view requires exactly one original film plot")
    result = copy.deepcopy(record)
    result["messages"] = [{"role": "system", "content": system}, copy.deepcopy(users[0]),
                          {"role": "assistant", "content": canonical_answer(record)}]
    return result


def build_training_view(params, train_examples, val_examples, tokenizer=None):
    """Return actual encoded train/val arrays and an independently reproducible proof."""
    mode = params["train"].get("response_format", "source")
    if mode not in ("source", "json_genres"):
        raise ValueError("Unsupported training response_format")
    source_config = yaml.safe_load(Path(params["data"]["tokenization_params"]).read_text(encoding="utf-8"))
    max_length = params["train"].get("view_max_seq_len") if mode == "json_genres" else source_config["tokenize"]["max_seq_len"]
    if type(max_length) is not int or max_length <= 0:
        raise ValueError("Training view requires a positive sequence cap")
    system = params["generation"]["system"] if mode == "json_genres" else None
    if mode == "json_genres" and (not isinstance(system, str) or not system.strip()):
        raise ValueError("JSON training view requires the serving system instruction")
    if mode == "json_genres":
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(params["model"]["name"], revision=params["model"]["revision"],
                                                      local_files_only=True)
        tokenizer.padding_side = source_config["tokenize"]["padding_side"]
        source_config = copy.deepcopy(source_config)
        source_config["generate"]["enable_thinking"] = params["model"]["enable_thinking"]
        source_config["tokenize"]["max_seq_len"] = max_length
    encoding_hash = hashlib.sha256()
    for name in ("src/training_view.py", "src/tokenize_data.py", "src/prompt.py"):
        encoding_hash.update(name.encode())
        encoding_hash.update(Path(name).read_bytes())
    proof = {"response_format": mode, "view_max_seq_len": max_length, "system": system,
             "system_sha256": fingerprint(system), "encoding_code_sha256": encoding_hash.hexdigest(),
             "tokenization_params_sha256": hashlib.sha256(Path(params["data"]["tokenization_params"]).read_bytes()).hexdigest(),
             "tokenizer": {"name": params["model"]["name"], "revision": params["model"]["revision"]},
             "splits": {}}
    auxiliary, auxiliary_proof = load_auxiliary(params)
    if auxiliary_proof:
        proof["auxiliary"] = auxiliary_proof
    prepared = {}
    for split, original in (("train", train_examples), ("val", val_examples)):
        if len({example["id"] for example in original}) != len(original):
            raise ValueError("Training view ids must be unique")
        path = Path(params["data"][f"source_{split}"])
        source_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["id"]: row for row in source_rows}
        aux_by_id = {row["id"]: row for row in auxiliary[split]}
        if len(by_id) != len(source_rows):
            raise ValueError("Duplicate original source id")
        if not set(aux_by_id).issubset({row["id"] for row in original}):
            raise ValueError("Every auxiliary row must be included in its actual encoded split")
        encoded_rows, entries, samples = [], [], []
        for example in original:
            identity = example["id"]
            if identity not in by_id and identity not in aux_by_id:
                raise ValueError("Selected view id is absent from the original source")
            if mode == "json_genres":
                from src.tokenize_data import encode_example
                record = (auxiliary_record(aux_by_id[identity], system) if identity in aux_by_id
                          else json_record(by_id[identity], system))
                encoded = encode_example(tokenizer, record, source_config, max_length)
                meta = encoded.pop("_meta")
                if meta["truncated"] or meta["full_len"] > max_length or not meta["supervised"]:
                    raise ValueError(f"{identity}: JSON view was truncated or has no supervision")
                supervised = [token for token, label in zip(encoded["input_ids"], encoded["labels"]) if label != -100]
                decoded = tokenizer.decode(supervised, skip_special_tokens=False)
                if decoded != record["messages"][-1]["content"] + tokenizer.eos_token:
                    raise ValueError("JSON view labels do not supervise the complete answer and EOS only")
            else:
                encoded = copy.deepcopy(example)
                record, decoded = by_id[identity], None
                meta = {"full_len": len(encoded["input_ids"]), "supervised": sum(label != -100 for label in encoded["labels"][1:]),
                        "truncated": None, "original_source_tensor": True}
            if (encoded["id"] != identity or len(encoded["input_ids"]) != len(encoded["labels"])
                    or len(encoded["attention_mask"]) != len(encoded["input_ids"]) or encoded["labels"][0] != -100
                    or not any(label != -100 for label in encoded["labels"][1:])):
                raise ValueError("Malformed encoded training view")
            if any(type(token) is not int or token < 0 for token in encoded["input_ids"]):
                raise ValueError("Training view tokens must be finite nonnegative integers")
            if (any(type(label) is not int or (label < 0 and label != -100) for label in encoded["labels"])
                    or any(type(value) is not int or value not in (0, 1) for value in encoded["attention_mask"])):
                raise ValueError("Training view labels and attention masks must be finite valid integers")
            if any(label != -100 and label != token for token, label in zip(encoded["input_ids"], encoded["labels"])):
                raise ValueError("Training view labels differ from their target token ids")
            encoded_rows.append(encoded)
            entries.append({"id": identity, "input_ids_sha256": fingerprint(encoded["input_ids"]),
                            "attention_mask_sha256": fingerprint(encoded["attention_mask"]), "labels_sha256": fingerprint(encoded["labels"]),
                            "input_tokens": sum(encoded["attention_mask"]), "supervised_tokens": meta["supervised"], "meta": meta})
            sample_kind = "original" if identity in by_id else "aux_" + aux_by_id[identity]["kind"]
            sample_limit = 3 if sample_kind == "original" else 2
            if sum(sample.get("kind", "original") == sample_kind for sample in samples) < sample_limit:
                samples.append({"id": identity, "messages": record["messages"], "encoded": encoded,
                                "decoded_supervised": decoded, "meta": meta})
                if auxiliary_proof:
                    samples[-1]["kind"] = sample_kind
        prepared[split] = encoded_rows
        proof["splits"][split] = {
            "ids": [row["id"] for row in encoded_rows], "n": len(encoded_rows), "entries": entries,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "original_selected_tensors_sha256": fingerprint([row for row in original if row["id"] in by_id]),
            "encoded_sha256": fingerprint(encoded_rows),
            "input_tokens": sum(entry["input_tokens"] for entry in entries),
            "supervised_tokens": sum(entry["supervised_tokens"] for entry in entries),
            "min_tokens": min(len(row["input_ids"]) for row in encoded_rows),
            "max_tokens": max(len(row["input_ids"]) for row in encoded_rows),
            "truncated": 0 if mode == "json_genres" else None, "samples": samples,
        }
        if auxiliary_proof:
            proof["splits"][split]["original_ids"] = [row["id"] for row in original if row["id"] in by_id]
            proof["splits"][split]["auxiliary_ids"] = [row["id"] for row in original if row["id"] in aux_by_id]
    if set(proof["splits"]["train"]["ids"]) & set(proof["splits"]["val"]["ids"]):
        raise ValueError("Training view train and val ids overlap")
    return prepared["train"], prepared["val"], proof
