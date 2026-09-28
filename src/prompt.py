"""Единая сборка диалога для обучения и генерации."""

from typing import Any


def _template_kwargs(params: dict) -> dict:
    """Передавать переключатель режима рассуждений из общей конфигурации."""
    value = params.get("generate", {}).get("enable_thinking")
    return {} if value is None else {"enable_thinking": value}


def split_messages(messages: list[dict]) -> tuple[list[dict], dict]:
    """Вернуть контекст и последний ответ ассистента."""
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("последняя реплика обучающего диалога должна быть ответом ассистента")
    return messages[:-1], messages[-1]


def build_chat_text(
    tokenizer: Any,
    messages: list[dict],
    params: dict,
    add_generation_prompt: bool,
) -> str:
    """Собрать текст одним шаблоном модели в обоих режимах.

    Инференс принимает и готовый диалог с последним ответом ассистента,
    и ещё не завершённый диалог с последним сообщением пользователя.
    Обучающий текст заканчивается именно EOS: некоторые шаблоны добавляют
    после него разделительный перевод строки, который не должен идти в лосс.
    """
    if not messages:
        raise ValueError("диалог пуст")
    if add_generation_prompt:
        context = messages[:-1] if messages[-1].get("role") == "assistant" else messages
        if not context:
            raise ValueError("нет сообщений для генерации")
    else:
        split_messages(messages)
        context = messages

    text = tokenizer.apply_chat_template(
        context,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **_template_kwargs(params),
    )
    if add_generation_prompt:
        return text

    eos = tokenizer.eos_token
    if not eos:
        raise ValueError("у токенизатора не задан EOS-токен")
    end = text.rfind(eos)
    if end < 0 or text[end + len(eos) :].strip():
        raise ValueError("шаблон обучения не завершает ответ явным EOS-токеном")
    return text[: end + len(eos)]


def supervised_prefix(full_text: str, messages: list[dict], eos_token: str) -> str:
    """Часть обучающей строки, которую нужно исключить из лосса.

    Проверка точного суффикса не даёт незаметно включить служебные токены
    шаблона в ответ. Например, шаблон Qwen3 может вставить пустой think-блок
    между заголовком ассистента и содержимым ответа.
    """
    _, answer = split_messages(messages)
    content = answer["content"]
    if not content:
        raise ValueError("ответ ассистента пуст")
    suffix = content + eos_token
    if not full_text.endswith(suffix):
        raise ValueError("шаблон не заканчивается точным ответом ассистента и EOS")
    return full_text[: -len(suffix)]


def prompt_token_len(
    tokenizer: Any,
    prompt_text: str,
    full_ids: list[int],
    full_offsets: list[tuple[int, int]],
) -> tuple[int, bool]:
    """Найти безопасную границу маски при возможной склейке BPE.

    Если токен пересекает границу промпта и ответа, он остаётся под маской.
    Первый токен лосса тогда начинается не раньше ответа. При полностью
    склеенном коротком ответе вернётся длина всей строки: пример будет
    отброшен как лишённый обучающего сигнала.
    """
    if len(full_offsets) != len(full_ids):
        raise ValueError("число офсетов не совпадает с числом токенов")
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    n_prompt = len(prompt_ids)
    if full_ids[:n_prompt] == prompt_ids:
        return n_prompt, False

    boundary = len(prompt_text)
    for i, (start, _end) in enumerate(full_offsets):
        if start >= boundary:
            return i, True
    return len(full_ids), True
