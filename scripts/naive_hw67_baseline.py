"""A diagnostic constant prior derived only from the actual training pool."""

from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def frequent_genres(source, identities):
    if not identities or len(identities) != len(set(identities)):
        raise ValueError("The training pool needs nonempty unique ids")
    sets = []
    for identity in identities:
        record = source[identity]
        genres = {part.strip().casefold() for part in record["messages"][-1]["content"].split(",")}
        if not genres or "" in genres:
            raise ValueError("The training source contains an empty genre")
        sets.append(genres)
    counts = Counter(genre for genres in sets for genre in genres)
    k = max(1, round(sum(map(len, sets)) / len(sets)))
    labels = sorted([genre for genre, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:k]])
    return labels, k, counts


def main():
    from src.common import load_params
    from src.experiment_train import expected_sampling_manifest
    from evaluation.contract import read_rows, write_json, fingerprint
    from evaluation.solution import score

    os.chdir(ROOT)
    params = load_params()
    identities = expected_sampling_manifest(params)["train"]["ids"]
    source = {row["id"]: row for row in read_rows(params["data"]["source_train"])}
    labels, k, counts = frequent_genres(source, identities)
    config = json.loads(Path("evaluation/params.json").read_text())
    rows = read_rows(config["golden"])
    raw = json.dumps({"answer": ", ".join(labels)}, ensure_ascii=False)
    outputs = [dict(id=row["id"], raw=raw, error=None) for row in rows]
    result = score(rows, outputs)
    result.update(
        kind="diagnostic_constant_prior",
        definition="Same top-k genre set for every film; k and frequencies use only the exact training pool, without test-based tuning.",
        training_rows=len(identities), training_ids_sha256=fingerprint(identities),
        golden_sha256=fingerprint(rows), k=k, predicted_genres=labels,
        source_genre_counts=dict(counts), used_for_gate=False,
    )
    write_json("metrics/naive-genre-baseline.json", result)
    print(json.dumps({"genres": labels, "k": k, "overall": result["overall"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
