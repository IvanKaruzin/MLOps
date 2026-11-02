"""Deterministic SFT views derived from verified immutable HW4 source data."""

import copy
import hashlib
import json
from pathlib import Path

import yaml


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


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
    prepared = {}
    for split, original in (("train", train_examples), ("val", val_examples)):
        if len({example["id"] for example in original}) != len(original):
            raise ValueError("Training view ids must be unique")
        path = Path(params["data"][f"source_{split}"])
        source_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["id"]: row for row in source_rows}
        if len(by_id) != len(source_rows):
            raise ValueError("Duplicate original source id")
        encoded_rows, entries, samples = [], [], []
        for example in original:
            identity = example["id"]
            if identity not in by_id:
                raise ValueError("Selected view id is absent from the original source")
            if mode == "json_genres":
                from src.tokenize_data import encode_example
                record = json_record(by_id[identity], system)
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
            if len(samples) < 3:
                samples.append({"id": identity, "messages": record["messages"], "encoded": encoded,
                                "decoded_supervised": decoded, "meta": meta})
        prepared[split] = encoded_rows
        proof["splits"][split] = {
            "ids": [row["id"] for row in encoded_rows], "n": len(encoded_rows), "entries": entries,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "original_selected_tensors_sha256": fingerprint(original), "encoded_sha256": fingerprint(encoded_rows),
            "input_tokens": sum(entry["input_tokens"] for entry in entries),
            "supervised_tokens": sum(entry["supervised_tokens"] for entry in entries),
            "min_tokens": min(len(row["input_ids"]) for row in encoded_rows),
            "max_tokens": max(len(row["input_ids"]) for row in encoded_rows),
            "truncated": 0 if mode == "json_genres" else None, "samples": samples,
        }
    if set(proof["splits"]["train"]["ids"]) & set(proof["splits"]["val"]["ids"]):
        raise ValueError("Training view train and val ids overlap")
    return prepared["train"], prepared["val"], proof
