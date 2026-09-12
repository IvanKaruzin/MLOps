"""Анатомия модели: параметры по слоям, нормы активаций, профиль памяти.

    python -m src.inspect_model            полный разбор, отчёт в docs/
    python -m src.inspect_model --probe M  один режим замера памяти (служебный
                                           вызов из отдельного процесса)

Файл называется inspect_model.py, а не inspect.py: имя inspect занято
модулем стандартной библиотеки, и его перекрытие ломает импорты в чужом коде.
"""

import argparse
import gc
import json
import os
import platform
import sys
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model

from src.config import load_params
from src.model import build_prompt, load_model, set_seed

from src.memory import PeakMemory, resolve_device

# transformers читает safetensors в несколько потоков, и на связке
# pyo3 OnceLock + GIL загрузка иногда встаёт намертво: на этой машине
# примерно один процесс из шести не доживал до конца from_pretrained.
# Замер обязан быть воспроизводимым, поэтому читаем последовательно —
# на модели 0.6B это не стоит ничего (4.4 с против 4.5 с).
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

MODES = ("inference", "full_ft", "lora")

# Порядок задаёт порядок строк в таблице. Проверка идёт сверху вниз,
# поэтому «norm» стоит после проекций: в их именах слова norm нет.
GROUPS = (
    ("embed", ("embed_tokens",)),
    ("q_proj", ("q_proj",)),
    ("k_proj", ("k_proj",)),
    ("v_proj", ("v_proj",)),
    ("o_proj", ("o_proj",)),
    ("gate_proj", ("gate_proj",)),
    ("up_proj", ("up_proj",)),
    ("down_proj", ("down_proj",)),
    ("norm", ("norm",)),
    ("lm_head", ("lm_head",)),
)


# --------------------------------------------------------------------------
# 1. Параметры по типам модулей
# --------------------------------------------------------------------------

def group_of(name: str) -> str:
    """Тип модуля по имени параметра."""
    for group, marks in GROUPS:
        if any(mark in name for mark in marks):
            return group
    return "прочее"


def parameter_rows(model) -> list[dict]:
    """Все тензоры параметров модели.

    remove_duplicate=False — иначе в таблицу не попадёт lm_head.
    """
    rows = []
    seen = set()
    for name, param in model.named_parameters(remove_duplicate=False):
        rows.append({
            "name": name,
            "shape": tuple(param.shape),
            "numel": param.numel(),
            "tied": id(param) in seen,
        })
        seen.add(id(param))
    return rows


def group_table(rows: list[dict]) -> list[dict]:
    """Свод «тип модуля → shape → параметров → доля от всей модели».

    В params попадают только уникальные тензоры, в tied_params — то,
    что модуль переиспользует у соседа.
    """
    total = sum(r["numel"] for r in rows if not r["tied"])
    agg: dict[str, dict] = {}
    for row in rows:
        group = group_of(row["name"])
        item = agg.setdefault(group, {
            "group": group, "modules": 0, "shapes": [], "params": 0, "tied_params": 0,
        })
        item["modules"] += 1
        shape = "×".join(map(str, row["shape"]))
        if shape not in item["shapes"]:
            item["shapes"].append(shape)
        if row["tied"]:
            item["tied_params"] += row["numel"]
        else:
            item["params"] += row["numel"]

    order = [g for g, _ in GROUPS] + ["прочее"]
    table = [agg[g] for g in order if g in agg]
    for item in table:
        # В группе norm форм две (по голове и по hidden), показываем обе.
        item["shape"] = ", ".join(item.pop("shapes"))
        item["share"] = item["params"] / total
    return table


# --------------------------------------------------------------------------
# 2. Forward-hooks и нормы активаций
# --------------------------------------------------------------------------

def decoder_layers(model):
    """Список декодер-блоков. У Qwen3 это model.model.layers."""
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    return decoder.layers


def hook_targets(model) -> dict[str, int]:
    """Первый, средний и последний блок — по номерам, а не по именам."""
    n_layers = len(decoder_layers(model))
    return {"первый": 0, "средний": n_layers // 2, "последний": n_layers - 1}


@contextmanager
def forward_hooks(modules: dict):
    """Собирать нормы и гарантированно снять только собственные hooks."""
    store: dict[str, list[float]] = {}
    handles = []

    def make_hook(label: str):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            norms = hidden[0].detach().float().norm(dim=-1)
            if hidden.device.type == "mps":
                torch.mps.synchronize()
            store[label] = norms.cpu().tolist()
        return hook

    try:
        for label, module in modules.items():
            handles.append(module.register_forward_hook(make_hook(label)))
        yield store
    finally:
        for handle in handles:
            handle.remove()


def activation_norms(tokenizer, model, params: dict) -> dict:
    """L2-нормы скрытых состояний на выходе трёх блоков, по позициям токена."""
    layers = decoder_layers(model)
    targets = hook_targets(model)
    prompt = build_prompt(tokenizer, params, params["hooks"]["prompt"])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with forward_hooks({label: layers[i] for label, i in targets.items()}) as store, torch.inference_mode():
        model(**inputs)

    return {
        "layers": targets,
        "norms": {label: store[label] for label in targets},
        "n_tokens": inputs["input_ids"].shape[1],
    }


# --------------------------------------------------------------------------
# 3. Сколько параметров добавляет LoRA
# --------------------------------------------------------------------------

def verify_activations(tokenizer, model, params: dict) -> dict:
    before = sum(len(m._forward_hooks) for m in model.modules())
    first = activation_norms(tokenizer, model, params)
    after_first = sum(len(m._forward_hooks) for m in model.modules())
    second = activation_norms(tokenizer, model, params)
    after_second = sum(len(m._forward_hooks) for m in model.modules())
    assert before == after_first == after_second
    difference = 0.0
    for label in first["norms"]:
        a, b = torch.tensor(first["norms"][label]), torch.tensor(second["norms"][label])
        torch.testing.assert_close(a, b)
        difference = max(difference, (a - b).abs().max().item())
    first["hook_counts"] = [before, after_first, after_second]
    first["max_repeat_difference"] = difference
    return first


def lora_config(params: dict, cfg: dict) -> LoraConfig:
    """LoraConfig из params.yaml — ни r, ни target_modules в коде не зашиты."""
    return LoraConfig(
        r=cfg["r"],
        lora_alpha=params["lora"]["alpha_ratio"] * cfg["r"],
        lora_dropout=params["lora"]["dropout"],
        target_modules=list(cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )


def lora_params_formula(model, r: int, target_modules) -> int:
    """Своя формула: на каждый целевой Linear ровно r * (in_features + out_features).

    A имеет форму (r, in), B — (out, r), смещений у них нет. Вся арифметика
    LoRA умещается в эту строчку, и она обязана сойтись с peft до штуки.
    """
    targets = set(target_modules)
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in targets:
            total += r * (module.in_features + module.out_features)
    return total


def lora_report(model, params: dict) -> list[dict]:
    """Для каждого конфига: своя формула против peft.

    Адаптер снимается через unload(): дальше модель нужна чистой.
    """
    base_params = sum(p.numel() for p in model.parameters())
    result = []
    for cfg in params["lora"]["configs"]:
        expected = lora_params_formula(model, cfg["r"], cfg["target_modules"])

        peft_model = get_peft_model(model, lora_config(params, cfg))
        peft_model.print_trainable_parameters()
        trainable, total = peft_model.get_nb_trainable_parameters()
        model = peft_model.unload()
        model.requires_grad_(True)

        result.append({
            "name": cfg["name"],
            "r": cfg["r"],
            "target_modules": list(cfg["target_modules"]),
            "formula": expected,
            "peft": trainable,
            "match": expected == trainable,
            "total_with_adapter": total,
            "share_of_base": trainable / base_params,
        })
    return result


# --------------------------------------------------------------------------
# 4. Память в трёх режимах
# --------------------------------------------------------------------------

def measure_mode(mode: str, params: dict) -> dict:
    """Один режим: инференс / full fine-tune / LoRA.

    Обучение — ровно один шаг forward + backward + optimizer.step():
    пик памяти достигается уже на нём, гонять эпоху незачем.
    """
    device = resolve_device(params)
    params["model"]["device"] = str(device)
    set_seed(params["generate"]["seed"])

    started = time.perf_counter()
    loss = None

    with PeakMemory(device) as peak:
        _, model = load_model(params)
        weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        peak.checkpoint()
        ids = torch.randint(
            0, model.config.vocab_size,
            (params["memory"]["batch_size"], params["memory"]["seq_len"]),
            device=model.device,
        )
        if mode == "inference":
            model.eval()
            with torch.inference_mode():
                model(input_ids=ids, use_cache=False)
        else:
            if mode == "lora":
                model = get_peft_model(model, lora_config(params, params["lora"]["configs"][0]))
            model.train()
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=float(params["memory"]["lr"]),
            )
            output = model(input_ids=ids, labels=ids, use_cache=False)
            peak.checkpoint()
            output.loss.backward()
            peak.checkpoint()
            optimizer.step()
            peak.checkpoint()
            optimizer.zero_grad(set_to_none=True)
            loss = round(output.loss.detach().item(), 4)

    result = peak.result()
    result.update(
        mode=mode,
        pid=os.getpid(),
        weights_mb=round(weights_bytes / 1024 ** 2, 1),
        device=str(device),
        seq_len=params["memory"]["seq_len"],
        batch_size=params["memory"]["batch_size"],
        seconds=round(time.perf_counter() - started, 1),
        loss=loss,
    )
    return result


def memory_profile(params: dict) -> list[dict]:
    """Профиль памяти в трёх режимах.

    memory.repeats задаёт число прогонов на режим; берётся худший (максимум).
    """
    repeats = max(1, int(params["memory"].get("repeats", 1)))
    results = []
    for mode in MODES:
        runs = []
        for _ in range(repeats):
            completed = subprocess.run(
                [sys.executable, "-m", "src.inspect_model", "--probe", mode, "--params-stdin"],
                input=json.dumps(params), text=True, capture_output=True, check=True,
                cwd=Path(__file__).resolve().parent.parent, timeout=600,
            )
            runs.append(json.loads(completed.stdout.strip().splitlines()[-1]))
        worst = max(runs, key=lambda item: item["peak_mb"])
        worst["repeats"] = repeats
        worst["pids"] = [item["pid"] for item in runs]
        worst["peak_mb_runs"] = [item["peak_mb"] for item in runs]
        results.append(worst)
        gc.collect()
    return results


# --------------------------------------------------------------------------
# 5. Условия, без которых цифры замера ничего не значат
# --------------------------------------------------------------------------

def environment(params: dict, memory: list[dict]) -> dict:
    """Всё, что нужно, чтобы чужой замер можно было сравнить со своим.

    Расхождение в полтора раза между двумя машинами — норма, а не ошибка,
    но только если написано, чем эти машины отличались. Метрики памяти берутся
    из самих замеров, а не из предположений: что реально сработало в дочернем
    процессе, то и уходит в отчёт.
    """
    def unique(field: str) -> str:
        values = dict.fromkeys(str(item.get(field) or "") for item in memory)
        return ", ".join(value for value in values if value)

    return {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "macos": platform.mac_ver()[0],
        "processor": (subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
                      if sys.platform == "darwin" else platform.processor()),
        "ram_gib": (int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)) / 1024 ** 3
                    if sys.platform == "darwin" else None),
        "system": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "device": params["model"]["device"],
        "dtype": params["model"]["dtype"],
        "seq_len": params["memory"]["seq_len"],
        "batch_size": params["memory"]["batch_size"],
        "repeats": max(1, int(params["memory"].get("repeats", 1))),
        "memory_metric": unique("metric_source"),
        "rss_metric": unique("rss_source"),
    }


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Разбор модели: параметры, активации, память")
    parser.add_argument("--probe", choices=MODES, help="служебный режим: замерить память и выйти")
    parser.add_argument("--params-only", action="store_true", help="проверить только параметры")
    parser.add_argument("--hooks-only", action="store_true", help="два прогона hooks и график")
    parser.add_argument("--params-stdin", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    params = json.load(sys.stdin) if args.params_stdin else load_params()
    set_seed(params["generate"]["seed"])
    params["model"]["device"] = str(resolve_device(params))

    if args.probe:
        print(json.dumps(measure_mode(args.probe, params), ensure_ascii=False))
        return

    if args.params_only:
        _, model = load_model(params)
        table = group_table(parameter_rows(model))
        total = sum(item["params"] for item in table)
        direct = sum(p.numel() for p in model.parameters())
        assert total == direct, (total, direct)
        print(json.dumps({"params_total": total, "params_direct": direct,
                          "params_by_group": table}, ensure_ascii=False, indent=2))
        return

    if args.hooks_only:
        from src.report import plot_activations

        tokenizer, model = load_model(params)
        result = verify_activations(tokenizer, model, params)
        plot_activations(result, params["hooks"]["plot"])
        print(json.dumps(result, ensure_ascii=False))
        return

    # Импорт здесь, а не наверху: matplotlib не нужен в служебных --probe
    # процессах, а тянется он заметно дольше остального.
    from src.report import write_report

    memory = memory_profile(params)
    tokenizer, model = load_model(params)
    rows = parameter_rows(model)
    table = group_table(rows)
    total = sum(item["params"] for item in table)
    assert total == sum(p.numel() for p in model.parameters())

    report = {
        "model": params["model"]["name"],
        "dtype": params["model"]["dtype"],
        "device": params["model"]["device"],
        "environment": environment(params, memory),
        "config": {
            key: getattr(model.config, key)
            for key in ("num_hidden_layers", "hidden_size", "intermediate_size",
                        "num_attention_heads", "num_key_value_heads", "head_dim",
                        "vocab_size", "tie_word_embeddings")
        },
        "params_total": total,
        "params_direct": sum(p.numel() for p in model.parameters()),
        "params_by_group": table,
        "activations": verify_activations(tokenizer, model, params),
        "lora": lora_report(model, params),
        "memory": memory,
    }

    peaks = {item["mode"]: item["peak_mb"] for item in memory}
    assert peaks["full_ft"] > peaks["lora"] > peaks["inference"], peaks
    assert all(item["peak_mb"] >= item["weights_mb"] for item in memory)
    assert all(item["match"] for item in report["lora"])
    Path(params["report"]["json"]).parent.mkdir(exist_ok=True)
    Path(params["report"]["json"]).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(report, params)

    env = report["environment"]
    print(f"\nПараметров: {total:,} (по таблице) / {report['params_direct']:,} (напрямую)"
          .replace(",", " "))
    print(f"Условия: {env['platform']}, device {env['device']}, dtype {env['dtype']}, "
          f"seq_len {env['seq_len']}, прогонов на режим {env['repeats']}, "
          f"torch {env['torch']}, transformers {env['transformers']}")
    for mode in report["memory"]:
        print(f"  {mode['mode']:<10} пик {mode['peak_mb']:>8.1f} МБ  "
              f"({mode['metric']}: {mode['metric_source']})")
    print(f"\nОтчёт: {params['report']['markdown']}, график: {params['hooks']['plot']}")


if __name__ == "__main__":
    main()
