"""Сборка docs/anatomy.md и графика норм активаций."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без дисплея: скрипт должен работать и в CI

import matplotlib.pyplot as plt  # noqa: E402  (backend выбирается до импорта)

MODE_TITLES = {
    "inference": "инференс",
    "full_ft": "full fine-tune",
    "lora": "LoRA (r=8, q/v)",
}


def thousands(n: int) -> str:
    """Число с неразрывными пробелами по разрядам."""
    return f"{n:,}".replace(",", " ")


def plot_activations(activations: dict, path: str) -> None:
    """Две панели: норма по позициям токена и средняя норма по трём блокам."""
    labels = list(activations["norms"])
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4), width_ratios=(2, 1))

    for label in labels:
        values = activations["norms"][label]
        ax_left.plot(values, linewidth=1.4,
                     label=f"{label} (слой {activations['layers'][label]})")
    ax_left.set_xlabel("позиция токена")
    ax_left.set_yscale("log")   # без лога всё придавит выброс massive activations
    ax_left.set_ylabel("‖h‖₂ (лог. шкала)")
    ax_left.set_title("Норма скрытого состояния по позициям")
    ax_left.legend(fontsize=9)
    ax_left.grid(alpha=0.3)

    means = [sum(activations["norms"][x]) / len(activations["norms"][x]) for x in labels]
    ax_right.bar(labels, means, color=["#4c78a8", "#f58518", "#54a24b"])
    ax_right.set_ylabel("средняя ‖h‖₂")
    ax_right.set_title("Средняя норма по блоку")
    ax_right.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    Path(path).parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def markdown_report(report: dict, params: dict) -> str:
    env, config = report["environment"], report["config"]
    memory = {item["mode"]: item for item in report["memory"]}
    base = memory["inference"]
    weights = base["weights_mb"]
    lines = [
        f"# Анатомия {report['model']}", "",
        "Результаты `make inspect`. Единица памяти в отчёте — МиБ (2²⁰ байт);",
        "ключи JSON с суффиксом `_mb` используют ту же единицу.", "",
        "## 1. Конфигурация", "",
        "| Параметр | Значение |", "|---|---:|",
    ]
    lines += [f"| {name} | {value} |" for name, value in config.items()]
    ratio = config["num_attention_heads"] / config["num_key_value_heads"]
    lines += ["", f"GQA: {config['num_attention_heads']} query-голов на "
              f"{config['num_key_value_heads']} KV-голов. При одинаковой длине контекста "
              f"KV-cache в {ratio:g} раза меньше, чем с отдельными K/V для каждой query-головы.", "",
              "## 2. Условия замера", "",
              "| Условие | Значение |", "|---|---|"]
    conditions = {
        "время UTC": env["measured_at"], "платформа": env["platform"],
        "ОС": f"macOS {env['macos']}" if env["macos"] else env["system"],
        "процессор": env["processor"], "RAM, ГиБ": env["ram_gib"],
        "модель": report["model"], "устройство": env["device"], "dtype": env["dtype"],
        "seq_len × batch": f"{env['seq_len']} × {env['batch_size']}",
        "прогонов на режим": env["repeats"], "seed": params["generate"]["seed"],
        "Python": env["python"], "torch": env["torch"],
        "transformers": env["transformers"], "peft": env["peft"],
        "память устройства": env["memory_metric"], "память процесса": env["rss_metric"],
    }
    lines += [f"| {name} | {value} |" for name, value in conditions.items()]
    lines += ["", "Каждый режим и каждый повтор выполняются в отдельном subprocess.",
              "Родитель загружает модель для таблиц и hooks только после завершения probes.",
              "Время включает загрузку модели и один шаг режима. Вход — случайные token IDs,",
              "`use_cache=False` во всех трёх режимах. Инференс — один forward в inference_mode;",
              "full FT и LoRA — forward, backward и один шаг AdamW.",
              f"AdamW: lr={params['memory']['lr']}, остальные параметры — значения PyTorch по умолчанию.",
              "На MPS sampler читает driver_allocated_memory каждые 10 мс; дополнительно",
              "синхронизирует устройство и снимает значения после загрузки, forward, backward",
              "и optimizer.step, до удаления градиентов. MPS-метрика включает кэш Metal",
              "и буферы MPSGraph. Это выборочный максимум: короткий всплеск между выборками",
              "может быть пропущен. CUDA использует собственный peak counter после reset.",
              "На CPU RSS — high-water mark за жизнь свежего процесса, включая импорты.", "",
              "## 3. Параметры по типам модулей", "",
              "| Группа | Модулей | Shape | Параметров | Доля | Разделяет тензор |",
              "|---|--:|---|--:|--:|--:|"]
    for row in report["params_by_group"]:
        lines.append(f"| {row['group']} | {row['modules']} | {row['shape']} | "
                     f"{thousands(row['params'])} | {100 * row['share']:.4f}% | "
                     f"{thousands(row['tied_params'])} |")
    lines += [f"| **Итого** | | | **{thousands(report['params_total'])}** | **100%** | |", "",
              f"Прямая сумма уникальных параметров: **{thousands(report['params_direct'])}**.",
              "Идентичность определена по объекту Parameter, как в model.parameters().",
              "Равные по значению независимые тензоры считаются отдельно; повторные имена",
              "одного Parameter отражаются в колонке связи и не увеличивают итог.", "",
              "## 4. Forward-hooks и нормы активаций", "",
              f"Промпт: «{params['hooks']['prompt']}». После chat template — "
              f"{report['activations']['n_tokens']} токенов.", "",
              f"![Нормы активаций]({Path(params['hooks']['plot']).name})", "",
              "| Блок | Индекс | Средняя L2-норма | Максимум | Позиция максимума (с 0) |",
              "|---|--:|--:|--:|--:|"]
    activations = report["activations"]
    for label, index in activations["layers"].items():
        norms = activations["norms"][label]
        lines.append(f"| {label} | {index} | {sum(norms) / len(norms):.2f} | "
                     f"{max(norms):.2f} | {norms.index(max(norms))} |")
    lines += ["", "Нормы сняты на выходе decoder blocks, до финальной нормализации модели.",
              "График показывает масштабы residual stream в трёх выбранных точках.",
              "Большая L2-норма сама по себе не доказывает attention sink: для этого",
              "нужно отдельно исследовать attention weights. Residual-связи тоже",
              "не гарантируют монотонный рост нормы на каждом слое или позиции.",
              f"Число hooks до и после двух прогонов: {activations['hook_counts']}; "
              f"максимальное расхождение повторных норм: {activations['max_repeat_difference']:g}.",
              "Очистка через finally проверена также при исключении, с сохранением чужого hook.", "",
              "## 5. LoRA-арифметика", "",
              "Для каждого целевого Linear: `r × (in_features + out_features)`.",
              "A имеет размер r × in, B — out × r. Смещения не обучаются.",
              "«Все линейные» здесь — семь decoder-проекций из конфигурации; lm_head исключён.", "",
              "| Конфиг | Своя формула | PEFT | Совпало | Доля от базы |",
              "|---|--:|--:|:--:|--:|"]
    for item in report["lora"]:
        lines.append(f"| {item['name']} | {thousands(item['formula'])} | "
                     f"{thousands(item['peft'])} | {'да' if item['match'] else 'нет'} | "
                     f"{item['share_of_base'] * 100:.4f}% |")
    lines += ["", "Доли рассчитаны от уникальных параметров базовой модели.",
              "PEFT печатает долю от базы вместе с адаптером, поэтому она немного меньше.",
              "Число 0,192% из задания относится к Qwen3-0.6B, а не к этой конфигурации.", "",
              "## 6. Память в трёх режимах", "",
              "| Режим | Пик устройства, МиБ | Пик RSS, МиБ | × к инференсу | Секунд | Loss | PID |",
              "|---|--:|--:|--:|--:|--:|---|"]
    for mode, item in memory.items():
        loss = "—" if item["loss"] is None else f"{item['loss']:.4f}"
        lines.append(f"| {MODE_TITLES[mode]} | {item['peak_mb']:.1f} | "
                     f"{item['peak_rss_mb']:.1f} | {item['peak_mb'] / base['peak_mb']:.3f} | "
                     f"{item['seconds']:.1f} | {loss} | {item['pids']} |")
    rss = [item["peak_rss_mb"] for item in memory.values()]
    lines += ["", f"Уникальные bf16-веса: {weights:.1f} МиБ. Каждый измеренный пик выше этой границы.",
              f"Грубая оценка full FT: веса {weights:.1f} + градиенты {weights:.1f} + "
              f"два состояния AdamW {2 * weights:.1f} = {4 * weights:.1f} МиБ до активаций.",
              "Эта оценка предполагает состояния того же dtype, как в данном AdamW;",
              "оптимизатор с fp32 master weights потребовал бы другой оценки.",
              f"Фактически full FT / LoRA = {memory['full_ft']['peak_mb'] / memory['lora']['peak_mb']:.3f}.",
              "В LoRA градиенты и состояния оптимизатора создаются только для адаптеров.",
              "Промежуточные активации для backward остаются, поэтому LoRA дороже инференса.",
              f"Разброс RSS трёх процессов — {max(rss) - min(rss):.1f} МиБ: "
              "RSS не заменяет память MPS. Значения разных метрик нельзя складывать,",
              "так как на Apple Silicon они относятся к общей физической памяти.", "",
              "## 7. Четыре дефекта и подтверждение исправлений", ""]
    baseline_path = Path("docs/hw2-baseline.json")
    if baseline_path.exists():
        import json

        old = json.loads(baseline_path.read_text(encoding="utf-8"))
        if old["model"]["name"] == report["model"]:
            lines += [f"Исходный каркас: [численные симптомы](hw2-baseline.json). "
                      f"Сумма параметров {thousands(old['broken_params_total'])}, "
                      f"hooks {old['broken_hook_counts']}, full FT "
                      f"{old['broken_full_ft']['peak_mb']:.1f} МиБ через "
                      f"{old['broken_full_ft']['metric_source']}. Это ошибочные результаты для сравнения.", ""]
    tied = sum(row["tied_params"] for row in report["params_by_group"])
    lines += [
        "### 1. Tied embeddings учитывались дважды", "",
        f"Каркас помечал каждый параметр tied=False. Повторно добавлялись {thousands(tied)} "
        f"параметров. Теперь итог {thousands(report['params_total'])} равен прямой сумме; "
        "lm_head сохранён отдельной строкой с нулевым собственным вкладом.", "",
        "### 2. Forward-hooks не снимались", "",
        "Каркас терял handles после register_forward_hook. Каждый вызов оставлял ещё три hooks. "
        f"Теперь handles снимаются в finally: {activations['hook_counts']}; "
        f"расхождение повторных норм {activations['max_repeat_difference']:g}. "
        "Само накопление hooks не обязано менять нормы; надёжный симптом — число регистраций.", "",
        "### 3. Вместо пика снимался остаток после gc.collect", "",
        "Один снимок после вычислений мог пропустить память градиентов и временных буферов. "
        "Теперь sampler действует от загрузки до завершения шага и сохраняет максимум; "
        "на границе optimizer.step замер выполняется до zero_grad. "
        "Регрессионный тест с расходом 10 → 100 → 10 сохраняет пик 100 и завершает sampler "
        "даже при исключении. Эта проверка отделяет ошибку момента замера от ошибки метрики.", "",
        "### 4. На ускорителе использовался RSS", "",
        f"Функция возвращала ru_maxrss под подписью «аллокатор mps». Теперь источник — "
        f"`{base['metric_source']}`. Full FT: {memory['full_ft']['peak_mb']:.1f} МиБ устройства "
        f"против {memory['full_ft']['peak_rss_mb']:.1f} МиБ RSS. "
        "Порядок full FT > LoRA > inference и нижняя граница размера весов проверяются кодом.", "",
        "Дополнительно устранён запуск режимов в одном процессе: PID в таблице различны. "
        "Настройки передаются дочернему процессу через stdin, включая изменения в памяти вызывающего кода.", "",
        "## 8. Проверки и источники", "",
        "`make check` — семь проверок ДЗ-2; `make check-all` — обе домашние работы. "
        "Регрессионные тесты: `uv run python -m unittest discover -s tests -p 'test_*.py'`.", "",
        "- [PyTorch: память Metal](https://docs.pytorch.org/docs/stable/generated/torch.mps.driver_allocated_memory.html)",
        "- [PyTorch: пик CUDA](https://docs.pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html)",
        "- [PyTorch: параметры и hooks](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)", "",
    ]
    return "\n".join(lines)


def write_report(report: dict, params: dict) -> None:
    plot_activations(report["activations"], params["hooks"]["plot"])
    path = Path(params["report"]["markdown"])
    path.parent.mkdir(exist_ok=True)
    path.write_text(markdown_report(report, params), encoding="utf-8")
