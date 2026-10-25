"""Строгая оценка ответов и fail-closed правило допуска кандидата."""

import math

from evaluation.contract import strict_json_loads


def _index(items, label):
    if not isinstance(items, list) or not items:
        raise ValueError(f'{label}: нужен непустой список')
    result = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id'].strip():
            raise ValueError(f'{label}: нужен строковый id')
        if item['id'] in result:
            raise ValueError(f'{label}: повторяющийся id {item["id"]}')
        result[item['id']] = item
    return result


def _names(value, label):
    if (not isinstance(value, list) or not value
            or any(not isinstance(x, str) or not x.strip() for x in value)
            or len(set(value)) != len(value)):
        raise ValueError(f'{label}: нужны уникальные непустые строки')
    return value


def _metrics(items):
    n = len(items)
    correct = sum(item['correct'] for item in items)
    valid = sum(item['valid'] for item in items)
    errors = sum(item['error'] is not None for item in items)
    return dict(n=n, correct=correct, valid=valid, errors=errors,
                accuracy=correct / n, format_rate=valid / n)


def score(rows, outputs):
    """Считать каждый вопрос, включая неверный формат и generation errors.

    Нарушение состава id — ошибка контракта. Невалидный ответ модели остаётся
    в per_item с valid=False. JSON с повторяющимися ключами невалиден.
    """
    row_index, output_index = _index(rows, 'rows'), _index(outputs, 'outputs')
    if set(row_index) != set(output_index):
        raise ValueError('Состав id rows и outputs различается')
    per_item = []
    for row in rows:
        choices = _names(row.get('choices'), 'choices')
        slices = _names(row.get('slices'), 'slices')
        if not isinstance(row.get('expected'), str) or row['expected'] not in choices:
            raise ValueError('expected должен входить в choices')
        out = output_index[row['id']]
        if 'raw' not in out or 'error' not in out:
            raise ValueError('В outputs нужны raw и error')
        raw, error = out['raw'], out['error']
        if raw is not None and not isinstance(raw, str):
            raise ValueError('raw должен быть строкой или null')
        if error is not None and (not isinstance(error, str) or not error.strip()):
            raise ValueError('error должен быть непустой строкой или null')
        parsed = None
        if isinstance(raw, str):
            try:
                parsed = strict_json_loads(raw)
            except (ValueError, TypeError, RecursionError):
                pass
        valid = (error is None and isinstance(parsed, dict) and set(parsed) == {'answer'}
                 and isinstance(parsed['answer'], str) and parsed['answer'] in choices)
        per_item.append(dict(id=row['id'], raw=raw, parsed=parsed, valid=bool(valid),
                             correct=bool(valid and parsed['answer'] == row['expected']),
                             error=error, slices=list(slices)))
    slice_names = sorted({s for item in per_item for s in item['slices']})
    return dict(overall=_metrics(per_item),
                slices={s: _metrics([item for item in per_item if s in item['slices']]) for s in slice_names},
                per_item=per_item)


def _validate_counts(actual, expected, label):
    if not isinstance(actual, dict):
        raise ValueError(f'{label}: отсутствуют метрики')
    for key in ('n', 'correct', 'valid', 'errors'):
        if type(actual.get(key)) is not int or actual[key] != expected[key]:
            raise ValueError(f'{label}: {key} не соответствует per_item')
    for key in ('accuracy', 'format_rate'):
        value = actual.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1
                or not math.isclose(value, expected[key], rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f'{label}: некорректная {key}')


def _validate_score(report, label):
    if not isinstance(report, dict):
        raise ValueError(f'{label}: нет отчёта')
    indexed = _index(report.get('per_item'), label + '.per_item')
    items = list(indexed.values())
    for item in items:
        if not {'raw', 'parsed', 'valid', 'correct', 'error', 'slices'} <= item.keys():
            raise ValueError(f'{label}: неполный per_item')
        _names(item['slices'], label + '.slices')
        if type(item['valid']) is not bool or type(item['correct']) is not bool:
            raise ValueError(f'{label}: valid/correct должны быть bool')
        if item['error'] is not None and (not isinstance(item['error'], str) or not item['error'].strip()):
            raise ValueError(f'{label}: некорректная ошибка генерации')
        if item['correct'] and not item['valid'] or item['error'] is not None and item['valid']:
            raise ValueError(f'{label}: противоречивый per_item')
        raw, parsed = item['raw'], None
        if raw is not None and not isinstance(raw, str):
            raise ValueError(f'{label}: raw должен быть строкой или null')
        if isinstance(raw, str):
            try:
                parsed = strict_json_loads(raw)
            except (ValueError, TypeError, RecursionError):
                pass
        if parsed != item['parsed']:
            raise ValueError(f'{label}: parsed не соответствует raw')
        if item['valid'] and (not isinstance(parsed, dict) or set(parsed) != {'answer'}
                              or not isinstance(parsed['answer'], str)):
            raise ValueError(f'{label}: valid при невалидном JSON')
    _validate_counts(report.get('overall'), _metrics(items), label + '.overall')
    slices = report.get('slices')
    names = {s for item in items for s in item['slices']}
    if not isinstance(slices, dict) or set(slices) != names:
        raise ValueError(f'{label}: состав срезов не соответствует per_item')
    for name in sorted(names):
        _validate_counts(slices[name], _metrics([item for item in items if name in item['slices']]),
                         f'{label}.slices.{name}')
    return indexed


def validate_model_spec(model):
    if not isinstance(model, dict) or model.get('kind') not in ('base', 'adapter', 'fixture'):
        raise ValueError('Не указана конкретная модель')
    # Учебные unit/integration fixtures явно отделены от реальных спецификаций.
    if model['kind'] == 'fixture':
        return
    if not isinstance(model.get('name'), str) or not model['name'].strip():
        raise ValueError('Не указано имя модели')
    if model['kind'] == 'base':
        revision = model.get('revision')
        if not isinstance(revision, str) or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError('База должна иметь точную commit-ревизию')
    else:
        version, run = model.get('version'), model.get('run_id')
        if (not isinstance(version, str) or not version.isdecimal() or int(version) < 1
                or not isinstance(run, str) or not run.strip()
                or model.get('uri') != f'models:/{model["name"]}/{version}'):
            raise ValueError('Неполная спецификация версии адаптера')


def quality_gate(baseline, candidate, critical_slices):
    """Допустить совместимого кандидата без регрессии; иначе ValueError."""
    _names(critical_slices, 'critical_slices')
    try:
        indexes, behaviors = [], []
        for label, report in [('baseline', baseline), ('candidate', candidate)]:
            if not isinstance(report, dict) or report.get('status') != 'complete':
                raise ValueError(f'{label}: оценка не завершена')
            validate_model_spec(report.get('model'))
            for key in ('protocol_sha256', 'golden_sha256', 'baseline_spec_sha256'):
                if not isinstance(report.get(key), str) or not report[key].strip():
                    raise ValueError(f'{label}: отсутствует {key}')
            indexes.append(_validate_score(report, label))
            behaviors.append(_validate_score(report.get('behavioral'), label + '.behavioral'))
            if report['overall']['n'] < 50 or len(report['slices']) < 3:
                raise ValueError('Golden должен содержать минимум 50 вопросов и 3 среза')
            if len(behaviors[-1]) < 3:
                raise ValueError('Нужно минимум 3 behavioral-кейса')
            if report['overall']['errors'] or report['behavioral']['overall']['errors']:
                raise ValueError(f'{label}: есть ошибки генерации')
        for key in ('protocol_sha256', 'golden_sha256', 'baseline_spec_sha256'):
            if baseline[key] != candidate[key]:
                raise ValueError(f'Несовместимый {key}')
        for label, a, b in [('golden', *indexes), ('behavioral', *behaviors)]:
            if set(a) != set(b) or any(set(a[i]['slices']) != set(b[i]['slices']) for i in a):
                raise ValueError(f'{label}: различаются id или принадлежность срезам')
        if not set(critical_slices) <= set(candidate['slices']):
            raise ValueError('Критический срез отсутствует')
        for metric in ('accuracy', 'format_rate'):
            if candidate['overall'][metric] < baseline['overall'][metric]:
                raise ValueError(f'Регрессия общей {metric}')
        for name in critical_slices:
            if candidate['slices'][name]['accuracy'] < baseline['slices'][name]['accuracy']:
                raise ValueError(f'Регрессия критического среза {name}')
        behavior = candidate['behavioral']['overall']
        if behavior['correct'] != behavior['n'] or behavior['valid'] != behavior['n']:
            raise ValueError('Не все обязательные behavioral-кейсы пройдены')
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError('Неполный или некорректный отчёт оценки') from exc
    return dict(passed=True, baseline_model=baseline['model'], candidate_model=candidate['model'],
                protocol_sha256=candidate['protocol_sha256'], golden_sha256=candidate['golden_sha256'],
                baseline_spec_sha256=candidate['baseline_spec_sha256'], critical_slices=list(critical_slices),
                n=candidate['overall']['n'], behavioral_n=behavior['n'],
                accuracy_delta=candidate['overall']['accuracy'] - baseline['overall']['accuracy'],
                format_rate_delta=candidate['overall']['format_rate'] - baseline['overall']['format_rate'])
