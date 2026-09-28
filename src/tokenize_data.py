"""Токенизация JSONL: примеры для SFT, метрики и отчёт по датасету."""

import json
import warnings
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from src.collate import LABEL_PAD_ID
from src.config import load_params
from src.pack import pack_examples, packing_report
from src.prompt import build_chat_text, prompt_token_len, supervised_prefix
from src.schema import iter_examples

METRICS_PATH = Path("metrics/tokenize.json")
REPORT_PATH = Path("docs/tokenize_report.md")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"Нет {path}.\n"
            "Сюда кладётся ВАШ датасет из ДЗ 3 — выход стадии split, тот же формат "
            "(id, topic, messages). Курсовой срез собирает `make sample`, но parquet "
            "для него есть только у преподавателя."
        )
    return [record.model_dump() for record in iter_examples(path)]


def mask_prompt(input_ids: list[int], n_prompt: int) -> list[int]:
    """Оставить в лоссе только токены ответа и EOS."""
    if not 0 <= n_prompt <= len(input_ids):
        raise ValueError("граница маски за пределами последовательности")
    return [LABEL_PAD_ID] * n_prompt + input_ids[n_prompt:]


def encode_example(tokenizer, record: dict, params: dict, max_seq_len: int) -> dict:
    """Один пример -> input_ids / attention_mask / labels + служебная статистика."""
    if max_seq_len <= 0:
        raise ValueError("max_seq_len должен быть положительным")
    messages = record["messages"]
    full_text = build_chat_text(tokenizer, messages, params, add_generation_prompt=False)
    inference_text = build_chat_text(tokenizer, messages, params, add_generation_prompt=True)
    if not full_text.startswith(inference_text):
        raise ValueError(f"{record.get('id')}: inference-промпт не является префиксом обучения")
    prompt_text = supervised_prefix(full_text, messages, tokenizer.eos_token)

    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    if not input_ids or input_ids[-1] != tokenizer.eos_token_id:
        raise ValueError(f"{record.get('id')}: ответ не заканчивается токеном EOS")
    n_prompt, used_fallback = prompt_token_len(
        tokenizer, prompt_text, input_ids, encoded["offset_mapping"]
    )

    full_len = len(input_ids)
    truncated = full_len > max_seq_len
    if truncated:
        input_ids = input_ids[:max_seq_len]

    labels = mask_prompt(input_ids, min(n_prompt, len(input_ids)))
    return {
        "id": record.get("id"),
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "_meta": {
            "id": record.get("id"),
            "full_len": full_len,
            "prompt_len": n_prompt,
            "answer_len": full_len - n_prompt,
            "truncated": truncated,
            "bpe_fallback": used_fallback,
            "supervised": sum(1 for x in labels if x != LABEL_PAD_ID),
        },
    }


def describe(values: list[int]) -> dict:
    """Распределение длин: перцентили важнее среднего — хвост решает max_seq_len."""
    a = np.asarray(values)
    if not a.size:
        return {"count": 0, "mean": 0.0, "p50": 0, "p90": 0, "p99": 0, "max": 0}
    return {
        "count": int(a.size),
        "mean": round(float(a.mean()), 1),
        "p50": int(np.percentile(a, 50)),
        "p90": int(np.percentile(a, 90)),
        "p99": int(np.percentile(a, 99)),
        "max": int(a.max()),
    }


def truncation_stats(metas: list[dict], name: str, params: dict) -> dict:
    """Статистика обрезки по max_seq_len."""
    count = sum(bool(meta["truncated"]) for meta in metas)
    ratio = count / len(metas) if metas else 0.0
    limit = params["tokenize"]["truncated_warn_ratio"]
    if ratio > limit:
        warnings.warn(
            f"{name}: обрезано {count}/{len(metas)} примеров ({ratio:.1%}); "
            f"порог {limit:.1%}. Увеличьте tokenize.max_seq_len после анализа длин.",
            RuntimeWarning,
            stacklevel=2,
        )
    return {"truncated": count, "truncated_ratio": round(ratio, 6)}


def process_split(
    tokenizer, name: str, path: Path, params: dict
) -> tuple[list[dict], dict, list[dict]]:
    """Токенизировать сплит и собрать по нему статистику."""
    cfg = params["tokenize"]
    max_seq_len = cfg["max_seq_len"]
    records = read_jsonl(path)

    examples: list[dict] = []
    metas: list[dict] = []
    dropped = 0
    for record in records:
        encoded = encode_example(tokenizer, record, params, max_seq_len)
        # Статистика длин считается по ВСЕМ записям, включая выброшенные:
        # иначе доля обрезанных занижается ровно на самые длинные примеры.
        meta = encoded.pop("_meta")
        metas.append(meta)
        # Обрезка съела весь ответ: учить нечему, такой пример только шумит.
        if meta["supervised"] == 0:
            dropped += 1
            continue
        examples.append(encoded)

    stats = {
        "examples_in": len(records),
        "examples_kept": len(examples),
        "dropped_no_supervision": dropped,
        "length_tokens": describe([m["full_len"] for m in metas]),
        "prompt_tokens": describe([m["prompt_len"] for m in metas]),
        "answer_tokens": describe([m["answer_len"] for m in metas]),
        "bpe_boundary_fallback": sum(m["bpe_fallback"] for m in metas),
        "total_tokens": sum(len(e["input_ids"]) for e in examples),
        "supervised_tokens": sum(
            sum(label != LABEL_PAD_ID for label in example["labels"]) for example in examples
        ),
    }
    print(
        f"  {name}: {len(examples)} примеров, токенов {stats['total_tokens']} "
        f"(в лосс идёт {stats['supervised_tokens']}), p50/p90/max = "
        f"{stats['length_tokens']['p50']}/{stats['length_tokens']['p90']}/"
        f"{stats['length_tokens']['max']}"
    )
    stats.update(truncation_stats(metas, name, params))

    if params["packing"]["enabled"]:
        bins = pack_examples(examples, max_seq_len)
        stats["packing"] = packing_report(examples, bins, max_seq_len, params["packing"]["batch_size"])
    else:
        stats["packing"] = None
        bins = []

    return examples, stats, bins


def estimate_train_time(total_tokens: int, params: dict) -> dict:
    """Грубый прогноз времени обучения: токены x эпохи / пропускная способность."""
    cfg = params["train_estimate"]
    tps = cfg["tokens_per_sec"]
    if tps <= 0 or cfg["epochs"] <= 0:
        raise ValueError("train_estimate.epochs и tokens_per_sec должны быть положительными")
    seconds = total_tokens * cfg["epochs"] / tps
    return {
        "epochs": cfg["epochs"],
        "tokens_per_sec": tps,
        "tokens_per_epoch": total_tokens,
        "seconds": round(seconds, 1),
        "hours": round(seconds / 3600, 2),
        "note": (
            "tokens_per_sec — ориентир по скорости ГЕНЕРАЦИИ из конфигурации. "
            "Обучение считает также backward и шаг оптимизатора; по материалам "
            "курса оно может быть медленнее в 2–3 раза. Реальное время нужно "
            "измерить в первом обучающем прогоне."
        ),
    }


def mask_line(share: float) -> str:
    """Строка отчёта про долю токенов, попавших в лосс.

    Доля около единицы означает, что промпт не замаскирован: такой отчёт
    обязан сказать об этом прямо, а не молча показать красивое число.
    """
    if share > 0.99:
        return (
            "В лосс идут ПОЧТИ ВСЕ токены последовательности — похоже, промпт "
            "не замаскирован, и модель учится воспроизводить вопрос наравне с ответом."
        )
    return (
        f"Промпт занимает {1 - share:.0%} токенов: без маски лосса именно он "
        "и составил бы большую часть обучающего сигнала."
    )


def truncation_line(metrics: dict, train: dict) -> str:
    """Строка отчёта про обрезку — или честное признание, что её не считали."""
    warn = metrics["truncated_warn_ratio"]
    if "truncated_ratio" not in train:
        return (
            "Доля обрезанных НЕ ПОСЧИТАНА: стадия не знает, сколько ответов "
            f"потеряла на max_seq_len = {metrics['max_seq_len']}."
        )
    ratio = train["truncated_ratio"]
    verdict = "в норме" if ratio <= warn else "ВЫШЕ ПОРОГА"
    return f"Порог предупреждения: {warn:.1%}. Фактически обрезано (train): {ratio:.1%} — {verdict}."


def truncated_cell(s: dict) -> str:
    """Ячейка «обрезано». Если статистики нет — так и пишем, а не молчим."""
    if "truncated_ratio" not in s:
        return "НЕ СЧИТАЛАСЬ"
    return f"{s['truncated']} ({s['truncated_ratio']:.1%})"


def render_report(metrics: dict) -> str:
    """Отчёт по датасету — то, что читают глазами перед запуском обучения."""
    lines = [
        "# Отчёт стадии tokenize",
        "",
        "Сгенерирован `make tokenize`, руками не правится.",
        "",
        f"- модель: `{metrics['model']}`",
        f"- `max_seq_len`: {metrics['max_seq_len']}",
        f"- `enable_thinking`: {str(metrics['enable_thinking']).lower()}",
        f"- `padding_side`: {metrics['padding_side']}",
        "",
        "## Длины в токенах",
        "",
        "| сплит | сохранено / вход | p50 | p90 | p99 | max | обрезано | всего токенов | в лосс |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in metrics["splits"].items():
        L = s["length_tokens"]
        lines.append(
            f"| {name} | {s['examples_kept']} / {s['examples_in']} | {L['p50']} | {L['p90']} | {L['p99']} | "
            f"{L['max']} | {truncated_cell(s)} | "
            f"{s['total_tokens']} | {s['supervised_tokens']} |"
        )

    train = metrics["splits"]["train"]
    est = metrics["train_time_estimate"]
    share = (
        train["supervised_tokens"] / train["total_tokens"]
        if train["total_tokens"] else 0.0
    )
    if not train["examples_in"]:
        truncation_detail = "В train нет примеров; распределение длин и обрезку оценить нельзя."
    elif train["truncated"]:
        truncation_detail = (
            f"p99 длины — {train['length_tokens']['p99']} токенов, максимум — "
            f"{train['length_tokens']['max']} при `max_seq_len` {metrics['max_seq_len']}. "
            f"Обрезано {train['truncated']} примеров; у "
            f"{train['dropped_no_supervision']} обрезка съела весь ответ, и они выброшены."
        )
    else:
        truncation_detail = (
            f"p99 длины — {train['length_tokens']['p99']} токенов, максимум — "
            f"{train['length_tokens']['max']} при `max_seq_len` {metrics['max_seq_len']}. "
            "Все ответы сохранились целиком; примеров без обучающего сигнала: "
            f"{train['dropped_no_supervision']}."
        )
    lines += [
        "",
        mask_line(share) if train["total_tokens"] else "В train нет сохранённых токенов для оценки доли лосса.",
        "",
        "## Обрезка",
        "",
        truncation_line(metrics, train),
        "",
        truncation_detail,
        "",
        "## Граница маски и BPE",
        "",
        f"Запасной путь по символьным офсетам сработал на {train['bpe_boundary_fallback']} "
        "примерах train. Если BPE склеивает границу, смешанный токен маскируется; "
        "в лосс попадает первый токен, начинающийся не раньше ответа.",
        "",
        "## Прогноз времени обучения",
        "",
        f"Токенов за эпоху: {est['tokens_per_epoch']}, эпох: {est['epochs']}, "
        f"скорость: {est['tokens_per_sec']} ток/с.",
        "",
        f"Итого {est['seconds']:.0f} с ({est['hours']} ч).",
        "",
        est["note"],
        "",
    ]

    pack = train.get("packing")
    if pack:
        lines += [
            "## Packing",
            "",
            f"Диагностическая оценка: последовательностей до packing: {pack['sequences_before']}, "
            f"бинов по {metrics['max_seq_len']} токенов: {pack['sequences_after']} "
            f"(заполнение {pack['fill_ratio']:.0%}, до {pack['max_examples_per_bin']} "
            "примеров в бине).",
            "",
            f"Шагов оптимизатора при batch_size {pack['batch_size']}: "
            f"{pack['steps_before']} -> {pack['steps_after']} "
            f"(-{pack['steps_saved_ratio']:.0%}).",
            "",
            "Это только оценка, не обучающий датасет. Без блочно-диагональной "
            "маски внимания токены соседних примеров видят друг друга; "
            "обучение на таких бинах меняет задачу.",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    params = load_params()
    tokenizer = AutoTokenizer.from_pretrained(params["model"]["name"])
    tokenizer.padding_side = params["tokenize"]["padding_side"]
    if tokenizer.pad_token_id is None:
        raise ValueError("у токенизатора не задан pad_token_id")

    out_dir = Path(params["data"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for name, key in (("train", "train_jsonl"), ("val", "val_jsonl")):
        examples, stats, _bins = process_split(tokenizer, name, Path(params["data"][key]), params)
        torch.save(
            {
                "examples": examples,
                "model": params["model"]["name"],
                "max_seq_len": params["tokenize"]["max_seq_len"],
                "padding_side": tokenizer.padding_side,
                "pad_token_id": tokenizer.pad_token_id,
            },
            out_dir / f"{name}.pt",
        )
        # Packing без изолирующей маски внимания нельзя отдавать в обучение.
        (out_dir / f"{name}_packed.pt").unlink(missing_ok=True)
        splits[name] = stats

    metrics = {
        "model": params["model"]["name"],
        "enable_thinking": params["generate"].get("enable_thinking"),
        "max_seq_len": params["tokenize"]["max_seq_len"],
        "padding_side": tokenizer.padding_side,
        "truncated_warn_ratio": params["tokenize"]["truncated_warn_ratio"],
        "splits": splits,
        "train_time_estimate": estimate_train_time(splits["train"]["total_tokens"], params),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(metrics), encoding="utf-8")
    print(f"  -> {out_dir}/, {METRICS_PATH}, {REPORT_PATH}")


if __name__ == "__main__":
    main()
