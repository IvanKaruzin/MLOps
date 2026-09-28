# План выполнения ДЗ-4

## Задача

Встроить токенизацию датасета ДЗ-3 в единый проект, исправить учебные дефекты `hw4-broken`, сохранить воспроизводимость DVC и работу ДЗ-1–3. Получить токенизированные train/val, метрики, отчёт, девять пройденных проверок ДЗ-4 и обновлённый README. После повторной проверки удалить `hw4-broken`, сделать коммиты и отправить их в `origin/main`.

## Структура работ

Исследование требований и исходного состояния; интеграция кода и конфигурации; проверка поведения; получение артефактов; документация и повторный аудит; публикация.

## Выполнение

### Stage 1: Зафиксировать контракт задания и исходное состояние

**Что сделать:**

- Прочитать все три HTML-файла `course_tasks/hw4`, схемы лекции и исходную самопроверку.
- Зафиксировать вход ДЗ-3 (4800 train и 599 val для v2), отпечаток исходного `check.sh` и карту переноса из `hw4-broken`.
- Вести выполнение в `vibe/HW4-plan-track.md`.

**Файлы:** `vibe/HW4-plan.md`, `vibe/HW4-plan-track.md`.

**Проверка:** `shasum -a 256 hw4-broken/tests/check.sh`; `git status --short`.

### Stage 2: Интегрировать токенизацию в общий проект

**Что сделать:**

- Перенести `prompt.py`, `tokenize_data.py`, `collate.py`, `pack.py` в `src/`, а `make_sample.py` в `scripts/`; оставить один корневой загрузчик конфигурации.
- В `src/prompt.py` применять `apply_chat_template` в train и inference через `generate.enable_thinking=false`; в `src/model.py` вызывать общий путь.
- Маскировать prompt и padding значением `-100`, вычислять границу с fallback по offsets, учить только ответ и EOS, отбрасывать записи без supervision.
- Добавить левый динамический padding, статистику длин и обрезки, прогноз времени, отчёт. Packing оставить выключенным без изоляции внимания между примерами.
- Объединить настройки в корневом `params.yaml`, добавить `tokenize` в `dvc.yaml` и цели в `Makefile`. Сохранить `tests/check.sh` ДЗ-4 байт-в-байт, перенести старую проверку ДЗ-3 в `tests/check_hw3.sh`.

**Файлы:** `src/prompt.py`, `src/tokenize_data.py`, `src/collate.py`, `src/pack.py`, `src/model.py`, `src/config.py`, `scripts/make_sample.py`, `params.yaml`, `dvc.yaml`, `Makefile`, `tests/check.sh`, `tests/check_hw3.sh`.

**Проверка:** `python3 -m compileall -q src scripts`; `shasum -a 256 tests/check.sh`; `git diff --check`.

### Stage 3: Проверить новую логику на граничных случаях

**Что сделать:**

- Проверить побайтовый префикс train/inference, маску prompt и EOS, BPE-склейку, разные длины с левым padding и zero-supervision после обрезки.
- Запустить штатную проверку ДЗ-4 на реальных train/val ДЗ-3; исправить найденные ошибки.

**Файлы:** `tests/test_hw4_unit.py` и только необходимые исправления файлов Stage 2.

**Проверка:** `UV_CACHE_DIR=/private/tmp/mlops-uv-cache uv run python -m unittest discover -s tests -p 'test_hw4*.py'`; `UV_CACHE_DIR=/private/tmp/mlops-uv-cache make check`.

### Stage 4: Получить воспроизводимые артефакты

**Что сделать:**

- Выбрать `max_seq_len` по фактическим p99/max своего датасета; зафиксировать порог обрезки и скорость из замера ДЗ-1.
- Запустить DVC DAG с новой стадией, сохранить `dvc.lock`, `metrics/tokenize.json`, `docs/tokenize_report.md`.
- Проверить согласованность чисел и чистый `dvc status`; убедиться, что тензоры находятся под DVC и исключены из Git.

**Файлы:** `params.yaml`, `dvc.lock`, `metrics/tokenize.json`, `docs/tokenize_report.md`.

**Проверка:** `UV_CACHE_DIR=/private/tmp/mlops-uv-cache DVC_SITE_CACHE_DIR=/private/tmp/mlops-dvc-site-cache uv run dvc repro`; `UV_CACHE_DIR=/private/tmp/mlops-uv-cache DVC_SITE_CACHE_DIR=/private/tmp/mlops-dvc-site-cache uv run dvc status`; `git ls-files data/`.

### Stage 5: Описать результат и найденные дефекты

**Что сделать:**

- Дополнить `docs/defects.md` четырьмя разделами ДЗ-4: симптом, причина, исправление и фактическое число для каждого.
- Обновить `README.md` с новым DAG, запуском, результатами, проверками и оговоркой об оценке времени обучения.
- Обновить сценарий демонстрации девяти проверок и отпечатка скрипта.

**Файлы:** `docs/defects.md`, `README.md`, `docs/screencast.md`.

**Проверка:** `UV_CACHE_DIR=/private/tmp/mlops-uv-cache make check`; `git diff --check`.

### Stage 6: Выполнить независимый повторный аудит и регрессию

**Что сделать:**

- Другим агентом перечитать diff, требования и фактические артефакты; исправить каждое подтверждённое замечание.
- Выполнить `make check-all` и повторно сверить метрики, выходы DVC, отпечаток проверки и состав Git.
- Проверить восстановление публичного source snapshot в свежем клоне без локального DVC remote; добавить `make bootstrap-source` с SHA-256 гейтом и проверить его отдельно.
- Удалить `hw4-broken/` после переноса всех нужных файлов; добавить `course_tasks/hw4/`.

**Файлы:** все изменённые файлы, `scripts/fetch_source.py`, `tests/test_fetch_source.py`, `hw4-broken/`, `course_tasks/hw4/`.

**Проверка:** `UV_CACHE_DIR=/private/tmp/mlops-uv-cache DVC_SITE_CACHE_DIR=/private/tmp/mlops-dvc-site-cache make check-all` с доступом к MPS; `UV_CACHE_DIR=/private/tmp/mlops-uv-cache uv run python -m unittest discover -s tests -p 'test_fetch_source.py'`; `UV_CACHE_DIR=/private/tmp/mlops-uv-cache DVC_SITE_CACHE_DIR=/private/tmp/mlops-dvc-site-cache make bootstrap-source`; `UV_CACHE_DIR=/private/tmp/mlops-uv-cache DVC_SITE_CACHE_DIR=/private/tmp/mlops-dvc-site-cache uv run dvc status`; `git diff --check`; `git status --short`.

### Stage 7: Зафиксировать и отправить изменения

**Что сделать:**

- Сделать коммиты по ходу выполнения: план, интеграция и проверка, документация и финальные исправления.
- Проверить `git log` и отправить `main` в `origin`.

**Файлы:** индекс и история Git.

**Проверка:** `git status --short --branch`; `git log -3 --oneline`; `git push origin main`.
