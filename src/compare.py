"""Сравнение базы и LoRA одним шаблоном и жадным декодированием."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_params
from src.prompt import build_chat_text
from src.runtime import resolve_device, resolve_dtype, set_seed


def inputs_fingerprint(params: dict, adapter_dir: Path) -> str:
    """Связать ответы с кодом, вопросами и точными переносимыми файлами."""
    digest = hashlib.sha256()
    for name in ("src/compare.py", "src/prompt.py", "src/runtime.py", "src/config.py", "uv.lock"):
        digest.update(name.encode())
        digest.update(Path(name).read_bytes())
    settings = {key: params[key] for key in ("compare", "generate", "model")}
    settings["seed"] = params.get("train", {}).get("seed", params.get("generate", {}).get("seed", 0))
    digest.update(json.dumps(settings, sort_keys=True).encode())
    files = sorted(path for path in adapter_dir.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"нет файлов адаптера в {adapter_dir}")
    for path in files:
        digest.update(path.relative_to(adapter_dir).as_posix().encode())
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()[:12]


def load_adapter_tokenizer(adapter_dir: Path):
    """Токенизатор — часть переносимого адаптера, не скрытая зависимость кеша."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("в токенизаторе адаптера не задан pad_token_id")
    if not tokenizer.chat_template:
        raise ValueError("в токенизаторе адаптера не сохранён шаблон чата")
    return tokenizer


@torch.inference_mode()
def generate(model, tok, prompts: list[str], system: str, params: dict, device) -> list[str]:
    """Сигнатура совпадает с независимой офлайн-проверкой из tests/check.sh."""
    seed = params.get("train", {}).get("seed", params.get("generate", {}).get("seed", 0))
    set_seed(seed)
    model.eval()
    answers = []
    for user in prompts:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = build_chat_text(tok, messages, params, add_generation_prompt=True)
        # Шаблон уже содержит служебные токены. Повторный BOS меняет вход
        # относительно tokenize и может незаметно исказить сравнение.
        inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        generated = model.generate(
            **inputs,
            max_new_tokens=params["compare"]["max_new_tokens"],
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
        )
        answer_ids = generated[0, inputs["input_ids"].shape[1] :]
        answers.append(tok.decode(answer_ids, skip_special_tokens=True).strip())
    return answers


def validate_compare(params: dict) -> None:
    cfg = params["compare"]
    prompts = cfg["prompts"]
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise ValueError("вопросы compare.prompts должны быть непустыми строками")
    if len(prompts) != 5 or len(set(prompts)) != 5:
        raise ValueError("compare.prompts должен содержать пять разных фиксированных вопросов")
    if not isinstance(cfg["system"], str) or not cfg["system"].strip():
        raise ValueError("compare.system должен быть непустой строкой")
    if not isinstance(cfg["max_new_tokens"], int) or cfg["max_new_tokens"] <= 0:
        raise ValueError("compare.max_new_tokens должен быть положительным целым числом")
    for key in ("expected", "source_ids"):
        values = cfg.get(key)
        if values is not None and (
            not isinstance(values, list) or len(values) != 5
            or any(not isinstance(value, str) or not value.strip() for value in values)
        ):
            raise ValueError(f"compare.{key}: нужны пять непустых строк")


def render_comparison(result: dict, max_new_tokens: int) -> str:
    """Показать точный вход, включая инструкцию, а не только обрезанный вопрос."""
    prompts, before, after = result["prompts"], result["base"], result["adapter"]
    if len(after) != len(prompts) or (before and len(before) != len(prompts)):
        raise ValueError("число ответов не совпадает с числом вопросов")
    lines = [
        "# Базовая модель против адаптера",
        "",
        f"База: `{result['model']}`. Адаптер: `{result['adapter_dir']}`.",
        f"Генерация жадная, до {max_new_tokens} токенов; режим рассуждений: "
        f"`{str(result['enable_thinking']).lower()}`.",
        "",
        "Пять примеров показывают изменение ответов после подключения адаптера. "
        "Они не являются статистической оценкой качества классификации.",
        "",
        "## Системная инструкция",
        "",
        result["system"],
        "",
    ]
    for i, (prompt, answer) in enumerate(zip(prompts, after, strict=True), 1):
        lines += [f"## {i}. Описание фильма", "", prompt, ""]
        if result.get("source_ids"):
            lines += [f"Источник: `{result['source_ids'][i - 1]}`.", ""]
        if result.get("expected"):
            lines += [f"**Эталонная метка датасета:** {result['expected'][i - 1]}", ""]
        if before:
            lines += ["**База:**", "", "> " + before[i - 1].replace("\n", "\n> "), ""]
        lines += ["**Адаптер:**", "", "> " + answer.replace("\n", "\n> "), ""]
    return "\n".join(line.rstrip() for line in "\n".join(lines).splitlines()) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="all_layers")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--only-adapter", action="store_true")
    parser.add_argument("--out", default=None, help="JSON для проверки переносимости адаптера")
    args = parser.parse_args()

    params = load_params()
    validate_compare(params)
    device = resolve_device(params["model"]["device"])
    dtype = resolve_dtype(params["model"]["dtype"])
    adapter_dir = Path(args.adapter_dir or Path(params["paths"]["models"]) / f"adapter_{args.variant}")
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    tokenizer = load_adapter_tokenizer(adapter_dir)
    cfg = params["compare"]
    metadata_path = adapter_dir / "training_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["generate"].get("enable_thinking") != params["generate"].get("enable_thinking"):
            raise ValueError("режим рассуждений отличается от обучения адаптера")
    fingerprint = inputs_fingerprint(params, adapter_dir)

    base = AutoModelForCausalLM.from_pretrained(
        adapter_config["base_model_name_or_path"], dtype=dtype
    ).to(device)
    before = [] if args.only_adapter else generate(
        base, tokenizer, cfg["prompts"], cfg["system"], params, device
    )
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False).to(device)
    after = generate(model, tokenizer, cfg["prompts"], cfg["system"], params, device)
    if fingerprint != inputs_fingerprint(params, adapter_dir):
        raise RuntimeError("код или файлы адаптера изменились во время сравнения")
    result = {
        "variant": args.variant,
        "adapter_dir": str(adapter_dir),
        "model": adapter_config["base_model_name_or_path"],
        "prompts": cfg["prompts"],
        "system": cfg["system"],
        "expected": cfg.get("expected"),
        "source_ids": cfg.get("source_ids"),
        "enable_thinking": params["generate"].get("enable_thinking"),
        "max_new_tokens": cfg["max_new_tokens"],
        "inputs_fingerprint": fingerprint,
        "base": before,
        "adapter": after,
    }
    output = Path(args.out) if args.out else Path(params["paths"]["metrics"]) / f"compare_{args.variant}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"-> {output}")
    if not args.out:
        report = Path(params["paths"]["compare"])
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render_comparison(result, cfg["max_new_tokens"]), encoding="utf-8")
        print(f"-> {report}")


if __name__ == "__main__":
    main()
