#!/usr/bin/env python3
"""Дополнительная строгая проверка полноты ДЗ5 без изменения check.sh."""

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_comparison(result: dict, params: dict, fingerprint: str) -> None:
    for key in ("base", "adapter", "prompts"):
        values = result.get(key)
        require(isinstance(values, list) and len(values) == 5, f"compare.{key}: нужны ровно пять строк")
        require(all(isinstance(value, str) and value.strip() for value in values),
                f"compare.{key}: пустой или некорректный ответ")
    require(result["prompts"] == params["compare"]["prompts"], "compare: вопросы отличаются от конфига")
    require(result.get("system") == params["compare"]["system"], "compare: системная инструкция отличается")
    require(result.get("max_new_tokens") == params["compare"]["max_new_tokens"], "compare: лимит токенов отличается")
    require(result.get("enable_thinking") == params["generate"].get("enable_thinking"),
            "compare: режим рассуждений отличается")
    require(result.get("model") == params["model"]["name"], "compare: другая базовая модель")
    for key in ("expected", "source_ids"):
        require(result.get(key) == params["compare"].get(key), f"compare: {key} отличается от конфига")
    require(result.get("inputs_fingerprint") == fingerprint, "compare: устаревший отпечаток кода/адаптера/вопросов")


def validate_training(metrics: dict, params: dict, variant: dict, fingerprint: str,
                      train_examples: list[dict], val_count: int) -> None:
    name, cfg = variant["name"], params["train"]
    require(cfg.get("max_steps") is None, "train.max_steps должен быть null для сдаваемого полного прогона")
    require(cfg["eval_every"] == 10, "валидация должна выполняться каждые 10 шагов")
    steps = math.ceil(math.ceil(len(train_examples) / cfg["batch_size"]) / cfg["grad_accum"]) * cfg["epochs"]
    require(metrics.get("variant") == name, f"{name}: другое имя варианта")
    require(metrics.get("inputs_fingerprint") == fingerprint, f"{name}: устаревший отпечаток обучения")
    require(metrics.get("validation_limited") is False and metrics.get("validation_examples") == val_count,
            f"{name}: нужен весь val, не --val-limit")
    require(metrics.get("steps") == steps and metrics.get("planned_steps") == steps,
            f"{name}: неполное обучение, ожидается {steps} шагов")
    require(metrics.get("processed_examples") == len(train_examples) * cfg["epochs"],
            f"{name}: не все обучающие примеры обработаны")
    tokens = sum(sum(example["attention_mask"]) for example in train_examples) * cfg["epochs"]
    supervised = sum(sum(label != -100 for label in example["labels"][1:])
                     for example in train_examples) * cfg["epochs"]
    require(metrics.get("processed_tokens") == tokens and metrics.get("processed_supervised_tokens") == supervised,
            f"{name}: неверное число реально обработанных токенов")
    from src.plot import validate_run

    validate_run(metrics)
    require([point[0] for point in metrics["curve_train"]] == list(range(1, steps + 1)),
            f"{name}: неполная последовательность train loss")
    expected_val_steps = sorted({0, steps, *range(10, steps + 1, 10)})
    require([point[0] for point in metrics["curve_val"]] == expected_val_steps,
            f"{name}: нужны val step0/каждые10/final: {expected_val_steps}")
    base, final = metrics.get("base_val_loss"), metrics.get("final_val_loss")
    require(isinstance(base, (float, int)) and isinstance(final, (float, int))
            and math.isfinite(base) and math.isfinite(final) and 0 <= final < base,
            f"{name}: финальный val loss не лучше базы")
    require(abs(base - metrics["curve_val"][0][1]) <= 1e-6 and final == metrics["curve_val"][-1][1],
            f"{name}: итоговые val loss не совпадают с кривой")
    losses = [point[1] for point in metrics["curve_train"]]
    width = max(1, len(losses) // 5)
    require(not metrics.get("diverged") and sum(losses[-width:]) < sum(losses[:width]),
            f"{name}: train loss не снизился или обучение разошлось")
    require(metrics.get("freeze_first") == variant["freeze_first"], f"{name}: неверная заморозка")


def validate_adapter(adapter_dir: Path, metrics: dict, params: dict, variant: dict,
                     model_config, fingerprint: str) -> None:
    from safetensors import safe_open

    cfg = read_json(adapter_dir / "adapter_config.json")
    metadata = read_json(adapter_dir / "training_metadata.json")
    require(metadata.get("inputs_fingerprint") == fingerprint, f"{variant['name']}: устаревший адаптер")
    for key in ("model", "generate", "tokenize"):
        require(metadata.get(key) == params[key], f"{variant['name']}: training_metadata.{key} отличается")
    require(cfg["base_model_name_or_path"] == params["model"]["name"], "адаптер обучен на другой базе")
    for saved, current in (("r", "r"), ("lora_alpha", "alpha"), ("lora_dropout", "dropout")):
        require(cfg[saved] == params["lora"][current], f"адаптер: {saved} отличается от конфига")
    require(not cfg.get("modules_to_save"), "адаптер сохраняет полные модули словаря")
    targets = set(params["lora"]["target_modules"])
    require(set(cfg["target_modules"]) == targets, "адаптер: target_modules отличаются")
    layers = list(range(variant["freeze_first"], model_config.num_hidden_layers))
    require(cfg.get("layers_to_transform") == layers, "адаптер: фактический список layers_to_transform неверен")
    expected = {(layer, target, matrix) for layer in layers for target in targets for matrix in ("A", "B")}
    actual, parameters = set(), 0
    with safe_open(adapter_dir / "adapter_model.safetensors", framework="pt", device="cpu") as tensors:
        for key in tensors.keys():
            match = re.search(r"\.layers\.(\d+)\.(.+)\.lora_([AB])\.weight$", key)
            require(match is not None, f"адаптер содержит неожиданные веса: {key}")
            layer, module, matrix = match.groups()
            identity = (int(layer), module.split(".")[-1], matrix)
            require(identity not in actual, f"дублированная матрица LoRA: {key}")
            actual.add(identity)
            shape = tensors.get_slice(key).get_shape()
            require(len(shape) == 2 and shape[0 if matrix == "A" else 1] == cfg["r"],
                    f"некорректная форма LoRA: {key}: {shape}")
            parameters += math.prod(shape)
    require(actual == expected, "реальные матрицы LoRA не соответствуют слоям/проекциям сохранённого конфига")
    require(parameters == metrics["trainable_params"], "число реально сохранённых LoRA-параметров не совпадает")
    size_mb = sum(path.stat().st_size for path in adapter_dir.rglob("*") if path.is_file()) / 1048576
    require(0 < size_mb <= 100 and abs(round(size_mb, 2) - metrics["adapter_size_mb"]) < 0.005,
            "неверный размер адаптера или адаптер больше 100 МБ")
    from src.compare import load_adapter_tokenizer

    load_adapter_tokenizer(adapter_dir)


def validate_tracked_files(params: dict) -> None:
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=PROJECT_ROOT,
                             check=True, capture_output=True).stdout.decode().split("\0")
    models = Path(params["paths"]["models"]).as_posix().rstrip("/") + "/"
    metrics = Path(params["paths"]["metrics"]).as_posix().rstrip("/") + "/"
    forbidden = []
    for name in filter(None, tracked):
        if (name.startswith(models) or Path(name).suffix in (".safetensors", ".pt", ".bin")
                or (name.startswith(metrics) and re.fullmatch(r"(?:train|compare)_.*\.json", Path(name).name))):
            forbidden.append(name)
    require(not forbidden, "в git отслеживаются артефакты обучения: " + ", ".join(forbidden))


def main() -> int:
    os.chdir(PROJECT_ROOT)
    from transformers import AutoConfig
    from src.compare import inputs_fingerprint as compare_fingerprint, render_comparison
    from src.config import load_params
    from src.data import load_split
    from src.train import inputs_fingerprint as train_fingerprint

    try:
        params = load_params()
        train = load_split(params["data"]["train"])["examples"]
        val = load_split(params["data"]["val"])["examples"]
        fingerprint = train_fingerprint(params)
        model_config = AutoConfig.from_pretrained(params["model"]["name"], local_files_only=True)
        runs = {}
        for variant in params["variants"]:
            metrics = read_json(Path(params["paths"]["metrics"]) / f"train_{variant['name']}.json")
            validate_training(metrics, params, variant, fingerprint, train, len(val))
            adapter_dir = Path(metrics["adapter_dir"])
            validate_adapter(adapter_dir, metrics, params, variant, model_config, fingerprint)
            runs[variant["name"]] = metrics
            print(f"✓ {variant['name']}: полное обучение ({metrics['steps']} шагов), весь val, реальные LoRA-слои/веса")
        require(runs["freeze14"]["trainable_params"] < runs["all_layers"]["trainable_params"],
                "freeze14 не уменьшает число обучаемых параметров")
        result = read_json(Path(params["paths"]["metrics"]) / "compare_all_layers.json")
        require(Path(result["adapter_dir"]).resolve() == Path(runs["all_layers"]["adapter_dir"]).resolve(),
                "compare использует адаптер другого прогона")
        validate_comparison(result, params, compare_fingerprint(params, Path(result["adapter_dir"])))
        report = Path(params["paths"]["compare"]).read_text(encoding="utf-8")
        require(report == render_comparison(result, params["compare"]["max_new_tokens"]),
                "docs/compare.md не соответствует актуальным ответам")
        curves = Path(params["paths"]["curves"])
        require(curves.is_file() and curves.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", "нет графика PNG")
        require(Path("docs/defects.md").is_file() and Path("docs/defects.md").stat().st_size > 0,
                "нет docs/defects.md")
        validate_tracked_files(params)
        print("✓ Сравнение: ровно пять полных ответов базы/адаптера, свежий отпечаток и согласованный отчёт")
        print("✓ График, разбор дефектов и гигиена Git проверены")
        print("Дополнительная проверка ДЗ5 пройдена.")
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(f"Дополнительная проверка ДЗ5: ОШИБКА: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
