# Progress tracker for HW4-plan.md

Status format: [ ] - not processed, [X] - completed

## Execution Stages

- [X] Stage 1: Зафиксировать контракт задания и исходное состояние (2026-09-28: изучены HTML, SVG и checker; SHA256 начинается с `f2002c8c9519`).
- [X] Stage 2: Интегрировать токенизацию в общий проект (2026-09-28: единый `src/`, `params.yaml`, `dvc.yaml`, `Makefile`, исходный checker без изменений).
- [X] Stage 3: Проверить новую логику на граничных случаях (2026-09-28: 5/5 unittest и 9/9 секций `make check`; checker №8 будет дополнен содержательным разбором на Stage 5).
- [X] Stage 4: Получить воспроизводимые артефакты (2026-09-28: `dvc repro` и `dvc status` успешны, 0% усечений в train и val, 51 919 supervised токенов train).
- [X] Stage 5: Описать результат и найденные дефекты (2026-09-28: README, сценарий записи и четыре разбора с фактическими числами; `make check` прошёл).
- [X] Stage 6: Выполнить независимый повторный аудит и регрессию (2026-09-28: `make check-all` прошёл с доступом к MPS; `dvc status` чистый; bootstrap без remote проверен в клоне и 2/2 unit-теста; `hw4-broken` удалён).
- [ ] Stage 7: Зафиксировать и отправить изменения.

## Notes

- Рабочая ветка `main` на старте совпадала с `origin/main`; исходные папки ДЗ-4 были untracked.
- Визуальная схема лекции ошибочно показывает обучаемый перевод строки после EOS. Сохраняем только ответ и EOS в `labels`.
- Packing не требуется для ДЗ-4; без блочной маски он позволил бы токенам разных примеров видеть друг друга.
- Для DVC в этом sandbox нужен `DVC_SITE_CACHE_DIR=/private/tmp/mlops-dvc-site-cache`; для uv — `UV_CACHE_DIR=/private/tmp/mlops-uv-cache`.
