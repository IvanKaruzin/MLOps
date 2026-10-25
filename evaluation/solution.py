"""Строгая оценка ответов и fail-closed правило допуска кандидата."""

import math

from evaluation.contract import strict_json_loads, target_for, parse_genre_answer, normalize_genre


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


MULTILABEL_CONVENTION = {
    'accuracy': 'exact equality of canonical genre sets; REFUSE is separate atomic behavior',
    'invalid_prediction': 'empty genre set, format failure and incorrect exact answer',
    'macro_f1': 'all taxonomy genres, including zero-support genres; zero_division=0',
    'refusal': 'REFUSE is not a genre; correct refusal is measured by behavioral exact accuracy',
}


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _label_metrics(tp, fp, fn):
    return dict(tp=tp, fp=fp, fn=fn, support=tp + fn,
                precision=_ratio(tp, tp + fp), recall=_ratio(tp, tp + fn),
                f1=_ratio(2 * tp, 2 * tp + fp + fn))


def _metrics(items, labels=None):
    n = len(items)
    correct = sum(item['correct'] for item in items)
    valid = sum(item['valid'] for item in items)
    errors = sum(item['error'] is not None for item in items)
    result = dict(n=n, correct=correct, valid=valid, errors=errors,
                  accuracy=correct / n, format_rate=valid / n)
    if labels is not None:
        per_label = {}
        for label in labels:
            tp = sum(label in item['expected_labels'] and label in item['predicted_labels'] for item in items)
            fp = sum(label not in item['expected_labels'] and label in item['predicted_labels'] for item in items)
            fn = sum(label in item['expected_labels'] and label not in item['predicted_labels'] for item in items)
            per_label[label] = _label_metrics(tp, fp, fn)
        tp, fp, fn = (sum(value[key] for value in per_label.values()) for key in ('tp', 'fp', 'fn'))
        combined = _label_metrics(tp, fp, fn)
        result.update({key: combined[key] for key in ('tp', 'fp', 'fn', 'support', 'precision', 'recall')})
        result.update(exact_set_accuracy=result['accuracy'], micro_f1=combined['f1'],
                      macro_f1=sum(value['f1'] for value in per_label.values()) / len(labels), per_label=per_label)
    return result


def score(rows, outputs):
    """Строгий JSON; MCQ либо свободное множество канонических жанров.

    Невалидная multilabel-генерация остаётся в знаменателе, даёт пустое
    predicted_labels и все FN эталона. REFUSE не входит в жанровой F1.
    """
    row_index, output_index = _index(rows, 'rows'), _index(outputs, 'outputs')
    if set(row_index) != set(output_index):
        raise ValueError('Состав id rows и outputs различается')
    targets = []
    for row in rows:
        _names(row.get('choices'), 'choices')
        targets.append(target_for(row))
    if len({target['task'] for target in targets}) != 1:
        raise ValueError('Нельзя смешивать задачи в одном отчёте')
    multilabel = targets[0]['task'] == 'multilabel'
    labels = targets[0]['labels'] if multilabel else None
    if multilabel and any(target['labels'] != labels for target in targets):
        raise ValueError('Таксономия жанров должна совпадать для всех вопросов')
    per_item = []
    for row, target in zip(rows, targets):
        slices = _names(row.get('slices'), 'slices')
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
                 and isinstance(parsed['answer'], str))
        extras = {}
        if multilabel:
            predicted, refused = [], False
            if valid:
                try:
                    predicted, refused = parse_genre_answer(parsed['answer'], labels)
                except ValueError:
                    valid = False
            if not valid:
                predicted, refused = [], False
            expected, expected_refusal = target['expected_labels'], target['expected_refusal']
            correct = valid and expected == predicted and expected_refusal == refused
            expected_set, predicted_set = set(expected), set(predicted)
            extras = dict(task='multilabel', expected_labels=expected, predicted_labels=predicted,
                          expected_refusal=expected_refusal, predicted_refusal=refused,
                          tp=len(expected_set & predicted_set), fp=len(predicted_set - expected_set),
                          fn=len(expected_set - predicted_set), support=len(expected_set))
        else:
            valid = valid and parsed['answer'] in row['choices']
            correct = valid and parsed['answer'] == row['expected']
        per_item.append(dict(id=row['id'], raw=raw, parsed=parsed, valid=bool(valid),
                             correct=bool(correct), error=error, slices=list(slices), **extras))
    slice_names = sorted({s for item in per_item for s in item['slices']})
    result = dict(overall=_metrics(per_item, labels),
                  slices={s: _metrics([item for item in per_item if s in item['slices']], labels) for s in slice_names},
                  per_item=per_item)
    if multilabel:
        result.update(task='multilabel', labels=list(labels), metric_convention=dict(MULTILABEL_CONVENTION))
    return result

def _validate_counts(actual, expected, label):
    if not isinstance(actual, dict):
        raise ValueError(f'{label}: отсутствуют метрики')
    count_keys = ('n', 'correct', 'valid', 'errors') if 'n' in expected else ()
    count_keys += tuple(key for key in ('tp', 'fp', 'fn', 'support') if key in expected)
    for key in count_keys:
        if type(actual.get(key)) is not int or actual[key] != expected[key]:
            raise ValueError(f'{label}: {key} не соответствует per_item')
    for key in ('accuracy', 'format_rate', 'exact_set_accuracy', 'precision', 'recall', 'micro_f1', 'macro_f1', 'f1'):
        if key not in expected:
            continue
        value = actual.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1
                or not math.isclose(value, expected[key], rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f'{label}: некорректная {key}')
    if 'per_label' in expected:
        per_label = actual.get('per_label')
        if not isinstance(per_label, dict) or set(per_label) != set(expected['per_label']):
            raise ValueError(f'{label}: подменены или пропущены жанровые классы')
        for genre in expected['per_label']:
            _validate_counts(per_label[genre], expected['per_label'][genre], f'{label}.per_label.{genre}')

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
    labels = None
    if report.get('task') == 'multilabel':
        labels = _names(report.get('labels'), label + '.labels')
        if labels != sorted(normalize_genre(genre) for genre in labels) or 'refuse' in labels:
            raise ValueError(f'{label}: таксономия не каноническая')
        reconstructed = []
        for item in items:
            for flag in ('expected_refusal', 'predicted_refusal'):
                if type(item.get(flag)) is not bool:
                    raise ValueError(f'{label}: нужен флаг {flag}')
            for key in ('expected_labels', 'predicted_labels'):
                values = item.get(key)
                if (not isinstance(values, list) or any(not isinstance(value, str) for value in values)
                        or values != sorted(set(values)) or not set(values) <= set(labels)):
                    raise ValueError(f'{label}: некорректный {key}')
            if item['expected_refusal'] == bool(item['expected_labels']):
                raise ValueError(f'{label}: некорректное сочетание эталонных жанров и отказа')
            for key in ('tp', 'fp', 'fn', 'support'):
                if type(item.get(key)) is not int:
                    raise ValueError(f'{label}: некорректный per_item.{key}')
            reconstructed.append(dict(id=item['id'], task='multilabel', choices=labels + ['REFUSE'],
                                      expected='REFUSE' if item['expected_refusal'] else ', '.join(item['expected_labels']),
                                      slices=item['slices']))
        recalculated = score(reconstructed, [dict(id=item['id'], raw=item['raw'], error=item['error']) for item in items])
        if report.get('metric_convention') != recalculated['metric_convention'] or report['per_item'] != recalculated['per_item']:
            raise ValueError(f'{label}: multilabel per_item/правило метрик не соответствует raw и эталонам')
    elif (report.get('task', 'mcq') != 'mcq' or 'labels' in report
          or any('expected_labels' in item or item.get('task') == 'multilabel' for item in items)
          or any(key in report.get('overall', {}) for key in ('micro_f1', 'macro_f1', 'per_label', 'exact_set_accuracy'))):
        raise ValueError(f'{label}: отсутствует или несовместима задача отчёта')
    _validate_counts(report.get('overall'), _metrics(items, labels), label + '.overall')
    slices = report.get('slices')
    names = {s for item in items for s in item['slices']}
    if not isinstance(slices, dict) or set(slices) != names:
        raise ValueError(f'{label}: состав срезов не соответствует per_item')
    for name in sorted(names):
        _validate_counts(slices[name], _metrics([item for item in items if name in item['slices']], labels),
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
        if (baseline.get('task', 'mcq') != candidate.get('task', 'mcq')
                or baseline.get('labels') != candidate.get('labels')):
            raise ValueError('Несовместимая задача или таксономия жанров')
        multilabel = candidate.get('task') == 'multilabel'
        if any(report.get('behavioral', {}).get('task', 'mcq') != report.get('task', 'mcq')
               or report.get('behavioral', {}).get('labels') != report.get('labels') for report in (baseline, candidate)):
            raise ValueError('Behavioral использует другую задачу или таксономию')
        for key in ('protocol_sha256', 'golden_sha256', 'baseline_spec_sha256'):
            if baseline[key] != candidate[key]:
                raise ValueError(f'Несовместимый {key}')
        for label, a, b in [('golden', *indexes), ('behavioral', *behaviors)]:
            if set(a) != set(b) or any(set(a[i]['slices']) != set(b[i]['slices']) for i in a):
                raise ValueError(f'{label}: различаются id или принадлежность срезам')
            if multilabel and any(a[i]['expected_labels'] != b[i]['expected_labels']
                                  or a[i]['expected_refusal'] != b[i]['expected_refusal'] for i in a):
                raise ValueError(f'{label}: различаются эталонные жанры/отказ')
        if not set(critical_slices) <= set(candidate['slices']):
            raise ValueError('Критический срез отсутствует')
        metrics = ('accuracy', 'format_rate', 'micro_f1', 'macro_f1') if multilabel else ('accuracy', 'format_rate')
        for metric in metrics:
            if candidate['overall'][metric] < baseline['overall'][metric]:
                raise ValueError(f'Регрессия общей {metric}')
        for name in critical_slices:
            for metric in (('accuracy', 'micro_f1') if multilabel else ('accuracy',)):
                if candidate['slices'][name][metric] < baseline['slices'][name][metric]:
                    raise ValueError(f'Регрессия критического среза {name}: {metric}')
        behavior = candidate['behavioral']['overall']
        if behavior['correct'] != behavior['n'] or behavior['valid'] != behavior['n']:
            raise ValueError('Не все обязательные behavioral-кейсы пройдены')
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError('Неполный или некорректный отчёт оценки') from exc
    result = dict(passed=True, baseline_model=baseline['model'], candidate_model=candidate['model'],
                protocol_sha256=candidate['protocol_sha256'], golden_sha256=candidate['golden_sha256'],
                baseline_spec_sha256=candidate['baseline_spec_sha256'], critical_slices=list(critical_slices),
                n=candidate['overall']['n'], behavioral_n=behavior['n'],
                accuracy_delta=candidate['overall']['accuracy'] - baseline['overall']['accuracy'],
                format_rate_delta=candidate['overall']['format_rate'] - baseline['overall']['format_rate'])

    if multilabel:
        result.update(micro_f1_delta=candidate['overall']['micro_f1'] - baseline['overall']['micro_f1'],
                      macro_f1_delta=candidate['overall']['macro_f1'] - baseline['overall']['macro_f1'],
                      task='multilabel', labels=candidate['labels'])
    return result
