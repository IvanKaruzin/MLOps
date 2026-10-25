"""Контракт файлов оценки и проверка утечки; работает без GPU и MLflow."""

import hashlib
import json
import math
from pathlib import Path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Повторяющийся JSON-ключ: {key}')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'Недопустимая JSON-константа: {value}')


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError('JSON-число должно быть конечным')
    return parsed


def strict_json_loads(value):
    return json.loads(value, object_pairs_hook=_unique_object, parse_constant=_reject_constant,
                      parse_float=_finite_float)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return strict_json_loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _validate_ids(rows, label):
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{label}: нужен непустой список строк')
    ids = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id'].strip():
            raise ValueError(f'{label}: нужны непустые строковые id')
        ids.append(row['id'])
    if len(ids) != len(set(ids)):
        raise ValueError(f'{label}: повторяющиеся id')


def read_rows(path):
    rows = [strict_json_loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    _validate_ids(rows, str(path))
    return rows


def _validate_strings(values, label):
    if (not isinstance(values, list) or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(values) != len(set(values))):
        raise ValueError(f'{label}: нужны уникальные непустые строки')


def validate_golden(rows, training, behavioral):
    for label, items in [('golden', rows), ('training', training), ('behavioral', behavioral)]:
        _validate_ids(items, label)
    if len(rows) < 50:
        raise ValueError('Golden set: минимум 50 примеров')
    for row in rows + behavioral + training:
        if not isinstance(row.get('input'), str) or not row['input'].strip():
            raise ValueError('Нет input')
        if not isinstance(row.get('group_id'), str) or not row['group_id'].strip():
            raise ValueError('Нет group_id')
    for row in rows + behavioral:
        _validate_strings(row.get('slices'), 'slices')
        _validate_strings(row.get('choices'), 'choices')
        if (not isinstance(row.get('expected'), str) or row['expected'] not in row['choices']
                or not isinstance(row.get('source'), str) or not row['source'].strip()):
            raise ValueError('Нужны проверенный expected и строковый source')
    def normalize(value):
        return ' '.join(value.casefold().split())
    golden_inputs = [normalize(row['input']) for row in rows]
    if len(golden_inputs) != len(set(golden_inputs)):
        raise ValueError('В golden есть точные дубли вопросов с разными id')
    train_inputs = {normalize(row['input']) for row in training}
    train_groups = {row['group_id'] for row in training}
    leaks = [row['id'] for row in rows + behavioral
             if normalize(row['input']) in train_inputs or row['group_id'] in train_groups]
    if leaks:
        raise ValueError(f'Пересечение train/evaluation: {leaks[:5]}')
    slices = {s for row in rows for s in row['slices']}
    if len(slices) < 3:
        raise ValueError('Нужны минимум три среза golden set')
    if any(row.get('type') not in ('format', 'paraphrase', 'refusal') for row in behavioral):
        raise ValueError('Неизвестный тип behavioral-кейса')
    kinds = {row['type'] for row in behavioral}
    if len(behavioral) < 3 or not {'format', 'paraphrase', 'refusal'} <= kinds:
        raise ValueError('Behavioral: format, paraphrase и refusal, минимум три кейса')
    if {r['id'] for r in rows} & {r['id'] for r in behavioral}:
        raise ValueError('ID golden и behavioral должны различаться')
    by_id = {row['id']: row for row in rows + behavioral}
    for row in behavioral:
        # paired_with необязателен для совместимости исходного формата. Если
        # автор указал пару, нельзя молча потерять её при редактировании данных.
        if 'paired_with' in row:
            if not isinstance(row['paired_with'], str) or not row['paired_with'].strip():
                raise ValueError('paired_with должен быть непустым строковым id')
            paired = by_id.get(row['paired_with'])
            if paired is None or paired is row or paired['group_id'] != row['group_id']:
                raise ValueError('Некорректная behavioral-пара paired_with/group_id')
    return dict(exact_or_group_intersections=0, n_train_index=len(training),
                n_golden=len(rows), n_behavioral=len(behavioral))
