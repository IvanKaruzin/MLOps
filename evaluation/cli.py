"""Готовая обвязка: студент реализует score и quality_gate в solution.py."""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

from evaluation.contract import fingerprint, read_json, read_rows, validate_golden, write_json
from evaluation.solution import quality_gate, score, validate_model_spec

BASELINE = Path('evaluation/baseline.json')
CANDIDATE = Path('metrics/candidate-model.json')
SELECTION = Path('metrics/candidate-selection.json')
EXPLICIT_SELECTION = 'explicit_finished_training_run'


def tracking():
    from mlflow import MlflowClient
    from src.common import connect, load_params
    p = load_params()
    connect(p)
    return p, MlflowClient()


def registry_version(client, spec):
    """Связать registry URI с реальным model-пакетом заявленного training run."""
    validate_model_spec(spec)
    if spec['kind'] != 'adapter':
        raise ValueError('Ожидается версия адаптера')
    version = client.get_model_version(spec['name'], spec['version'])
    run = client.get_run(spec['run_id'])
    allowed_sources = {f'runs:/{spec["run_id"]}/model',
                       run.info.artifact_uri.rstrip('/') + '/model'}
    if (version.run_id != spec['run_id'] or version.status != 'READY'
            or not isinstance(version.source, str) or version.source.rstrip('/') not in allowed_sources):
        raise ValueError('Версия реестра не соответствует model-пакету training run')
    return version


def manifest_for(spec):
    import yaml
    _, client = tracking()
    registry_version(client, spec)
    root = Path(client.download_artifacts(spec['run_id'], 'model'))
    info = yaml.safe_load((root / 'MLmodel').read_text())
    adapter = (root / info['flavors']['python_function']['artifacts']['adapter']['path']).resolve()
    if not adapter.is_relative_to(root.resolve()):
        raise ValueError('Некорректный путь адаптера')
    manifest = read_json(adapter / 'manifest.json')
    validate_model_spec(dict(kind='base', name=manifest['model']['name'], revision=manifest['model']['revision']))
    return manifest, adapter


def explicit_training_run(client, params, run_id, reason):
    """Validate one user-selected model without claiming a completed fair sweep."""
    from src.common import provenance
    from src.promote import gate as loss_gate, validate_sampling, validate_training_view, validate_checkpoints
    from src.sweep import verify_source_proof
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('Для явного выбора нужна непустая причина --reason')
    run = client.get_run(run_id)
    parent_id = run.data.tags.get('mlflow.parentRunId')
    if (run.info.run_id != run_id or run.info.status != 'FINISHED'
            or run.data.tags.get('kind') not in ('candidate', 'development')
            or not isinstance(parent_id, str) or not parent_id or parent_id == run_id):
        raise ValueError('Явно выбрать можно только завершённый training child')
    parent = client.get_run(parent_id)
    if (parent.info.run_id != parent_id or parent.info.status not in ('FINISHED', 'KILLED')
            or parent.info.experiment_id != run.info.experiment_id):
        raise ValueError('Родитель должен быть завершён или явно остановлен в том же experiment')
    current = {key: value for key, value in provenance(params).items() if key not in ('git_sha', 'git_dirty')}
    if any(item.data.tags.get(key) != value for item in (parent, run) for key, value in current.items()):
        raise ValueError('Версии явного кандидата или родителя устарели')
    configurations = {item['name']: item for item in params['sweep']}
    configuration = configurations.get(run.data.tags.get('candidate'))
    if (configuration is None or run.data.tags.get('base_revision') != params['model']['revision']
            or run.data.metrics.get('steps') != params['train']['max_steps']):
        raise ValueError('База, конфигурация или бюджет явного кандидата не совпадают')
    logged = {'r': configuration['r'], 'lr': configuration['lr'],
              'target_modules': ','.join(configuration['target_modules']),
              **{f'train.{key}': value for key, value in params['train'].items()},
              'model': params['model']['name'], 'dtype': params['model']['dtype'],
              'sampling.method': params['sampling']['method'], 'sampling.seed': params['sampling']['seed'],
              'alpha_over_r': params['lora']['alpha_over_r'], 'dropout': params['lora']['dropout']}
    if any(run.data.params.get(key) != str(value) for key, value in logged.items()):
        raise ValueError('Сохранённые параметры явного кандидата отличаются от конфигурации')
    with tempfile.TemporaryDirectory(prefix='hw67-explicit-selection-') as directory:
        def artifact(identity, name):
            destination = Path(directory) / identity
            destination.mkdir(exist_ok=True)
            return read_json(client.download_artifacts(identity, name, str(destination)))
        if artifact(run_id, 'config.json') != params:
            raise ValueError('Серверный config.json отличается от текущего конфига')
        source = artifact(run_id, 'source-link-check.json')
        source_sha = hashlib.sha256(json.dumps(source, sort_keys=True, allow_nan=False).encode()).hexdigest()
        verify_source_proof(params, current, source)
        if (run.data.tags.get('source_link_sha256') != source_sha
                or artifact(parent_id, 'source-link-check.json') != source):
            raise ValueError('Source/tensor proof кандидата и родителя не совпадает')
        sampling = validate_sampling(client, params, [run])
        validate_training_view(client, params, [run], sampling)
        validate_checkpoints(client, params, [run], sampling)
        checkpoint = artifact(run_id, 'checkpoint.json')
    history = client.get_metric_history(run_id, 'train_loss')
    if (len(history) != params['train']['max_steps']
            or {point.step for point in history} != set(range(1, params['train']['max_steps'] + 1))
            or any(not math.isfinite(point.value) or point.value < 0 for point in history)):
        raise ValueError('История train loss не подтверждает полный бюджет')
    curve = sorted(client.get_metric_history(run_id, 'val_loss'), key=lambda point: point.step)
    expected = checkpoint['checkpoints']
    if (len(curve) != len(expected)
            or any(point.step != saved['step'] or not math.isclose(point.value, saved['val_loss'], rel_tol=1e-9, abs_tol=1e-9)
                   for point, saved in zip(curve, expected))):
        raise ValueError('Серверная val curve не соответствует checkpoint proof')
    after = {key: value for key, value in provenance(params).items() if key not in ('git_sha', 'git_dirty')}
    if after != current:
        raise ValueError('Входы изменились во время проверки явного кандидата')
    loss_gate(run, params['registry']['max_val_ratio'])
    evidence = dict(schema_version=1, mode=EXPLICIT_SELECTION, reason=reason.strip(),
                    training_run_id=run_id, parent_run_id=parent_id, parent_status=parent.info.status,
                    source_sweep_cancelled=parent.info.status == 'KILLED', fair_sweep_verified=False,
                    selection_scope='one_finished_child_not_a_sweep_winner', provenance=current,
                    configuration=configuration, training_steps=params['train']['max_steps'],
                    source_link_sha256=source_sha, sampling_sha256=run.data.tags['sampling_sha256'],
                    training_view_sha256=run.data.tags['training_view_sha256'],
                    checkpoint_sha256=fingerprint(checkpoint), adapter_sha256=checkpoint['adapter_sha256'])
    return run, evidence

def candidate_model():
    from src.promote import gate as loss_gate, select
    p, client = tracking()
    spec = read_json(CANDIDATE)
    validate_model_spec(spec)
    if spec['kind'] != 'adapter' or spec['name'] != p['registry']['name']:
        raise ValueError('Неверный реестр кандидата')
    version = registry_version(client, spec)
    alias = client.get_model_version_by_alias(spec['name'], 'candidate')
    if (version.run_id != spec['run_id'] or str(alias.version) != spec['version']
            or alias.run_id != spec['run_id']):
        raise ValueError('Кандидат/alias не соответствует закреплённой версии реестра')
    selection_mode = version.tags.get('selection.mode')
    if SELECTION.exists() or selection_mode == EXPLICIT_SELECTION:
        evidence = read_json(SELECTION)
        if (selection_mode != EXPLICIT_SELECTION or evidence.get('model') != spec
                or version.tags.get('selection.sha256') != fingerprint(evidence)):
            raise ValueError('Явный выбор не соответствует закреплённой версии и evidence')
        _, verified = explicit_training_run(client, p, spec['run_id'], evidence.get('reason'))
        if evidence != dict(verified, model=spec):
            raise ValueError('Явный выбор или его доказательства изменились после регистрации')
    else:
        best = select(client, p, read_json('metrics/sweep.json'))
        if best.info.run_id != spec['run_id']:
            raise ValueError('Выбран другой/устаревший кандидат; выполните make candidate')
        loss_gate(best, p['registry']['max_val_ratio'])
    return spec


def prepare_candidate(training_run_id=None, reason=None):
    from src.promote import gate as loss_gate, promote, select
    p, client = tracking()
    evidence = None
    if training_run_id is not None:
        import mlflow
        best, evidence = explicit_training_run(client, p, training_run_id, reason)
        # A stopped sweep remains stopped; only this validated model is registered.
        ratio = loss_gate(best, p['registry']['max_val_ratio'])
        version = mlflow.register_model(f'runs:/{best.info.run_id}/model', p['registry']['name'])
        for key, value in {'val_ratio': ratio, 'gate.max_val_ratio': p['registry']['max_val_ratio'],
                           'validation': 'passed'}.items():
            client.set_model_version_tag(version.name, version.version, key, str(value))
    else:
        if reason is not None:
            raise ValueError('--reason используется только с --training-run-id')
        best = select(client, p, read_json('metrics/sweep.json'))
        version = promote(client, best, p['registry']['name'], 'candidate', p['registry']['max_val_ratio'])
    spec = dict(kind='adapter', name=version.name, version=str(version.version), run_id=best.info.run_id,
                uri=f'models:/{version.name}/{version.version}')
    if evidence is not None:
        evidence['model'] = spec
        client.set_model_version_tag(version.name, version.version, 'selection.mode', EXPLICIT_SELECTION)
        client.set_model_version_tag(version.name, version.version, 'selection.sha256', fingerprint(evidence))
        client.set_registered_model_alias(version.name, 'candidate', version.version)
        write_json(SELECTION, evidence)
    elif SELECTION.exists():
        SELECTION.unlink()
    write_json(CANDIDATE, spec)
    print(json.dumps(spec, ensure_ascii=False, indent=2))


def init_baseline():
    if BASELINE.exists():
        raise ValueError('Baseline уже закреплён. Обновление только make baseline-update после принятия')
    manifest, _ = manifest_for(candidate_model())
    write_json(BASELINE, dict(kind='base', name=manifest['model']['name'],
                             revision=manifest['model']['revision']))
    print('Baseline: исходная модель с точной ревизией. Сохраните evaluation/baseline.json в Git')



def inputs():
    config = read_json('evaluation/params.json')
    if (not isinstance(config.get('system'), str) or not config['system'].strip()
            or type(config.get('seed')) is not int
            or type(config.get('max_new_tokens')) is not int or config['max_new_tokens'] < 1
            or type(config.get('enable_thinking')) is not bool):
        raise ValueError('Неполный протокол генерации')
    critical = config.get('critical_slices')
    if (not isinstance(critical, list) or not critical
            or any(not isinstance(s, str) or not s.strip() for s in critical)
            or len(set(critical)) != len(critical)):
        raise ValueError('Нужен минимум один именованный критический срез')
    rows = read_rows(config['golden'])
    training = read_rows(config['training_index'])
    behavioral = read_rows(config['behavioral'])
    proof = validate_golden(rows, training, behavioral)
    if not set(config['critical_slices']) <= {s for r in rows for s in r['slices']}:
        raise ValueError('Критический срез отсутствует')
    baseline = read_json(BASELINE)
    validate_model_spec(baseline)
    if baseline['kind'] not in ('base', 'adapter'):
        raise ValueError('Baseline должен быть реальной моделью')
    protocol = dict(config=config,
                    golden=rows, behavioral=behavioral, training_index=training,
                    evaluation_code={p.name: p.read_text() for p in sorted(Path('evaluation').glob('*.py'))})
    return config, rows, behavioral, baseline, proof, fingerprint(protocol)


def generate(spec, common_manifest, config, rows):
    """Одинаковый eval prompt и decoding. База и адаптер загружаются последовательно."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.runtime import resolve_device, resolve_dtype, set_seed
    manifest = common_manifest
    adapter = None
    if spec['kind'] == 'adapter':
        manifest, adapter = manifest_for(spec)
    model_cfg = manifest['model']
    expected_base = common_manifest['model']
    actual_name = spec['name'] if spec['kind'] == 'base' else model_cfg['name']
    actual_revision = spec['revision'] if spec['kind'] == 'base' else model_cfg['revision']
    if (actual_name, actual_revision) != (expected_base['name'], expected_base['revision']):
        raise ValueError('Baseline и candidate должны использовать одну закреплённую базу')
    base_name = spec['name'] if spec['kind'] == 'base' else model_cfg['name']
    revision = spec['revision'] if spec['kind'] == 'base' else model_cfg['revision']
    if len(revision) != 40:
        raise ValueError('Нужна точная commit-ревизия базы')
    device = resolve_device(os.environ.get('HW6_DEVICE', expected_base['device']))
    dtype = resolve_dtype(os.environ.get('HW6_DTYPE', expected_base['dtype']))
    set_seed(config['seed'])
    # Один закреплённый tokenizer/chat template для всех сравниваемых ролей.
    tokenizer = AutoTokenizer.from_pretrained(expected_base['name'], revision=expected_base['revision'])
    model = AutoModelForCausalLM.from_pretrained(base_name, revision=revision, dtype=dtype)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model = model.to(device).eval()
    outputs = []
    for row in rows:
        started = time.monotonic()
        raw, error = '', None
        try:
            text = tokenizer.apply_chat_template(
                [{'role': 'system', 'content': config['system']}, {'role': 'user', 'content': row['input']}],
                tokenize=False, add_generation_prompt=True, enable_thinking=config['enable_thinking'])
            batch = tokenizer(text, return_tensors='pt').to(device)
            with torch.inference_mode():
                out = model.generate(**batch, max_new_tokens=config['max_new_tokens'], do_sample=False,
                                     pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id))
            raw = tokenizer.decode(out[0, batch['input_ids'].shape[1]:], skip_special_tokens=True)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        outputs.append(dict(id=row['id'], raw=raw, error=error,
                            latency_seconds=time.monotonic() - started))
    del model, tokenizer
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    return outputs


def answer_diff(anchor, other):
    a = {item['id']: item for item in anchor['per_item']}
    b = {item['id']: item for item in other['per_item']}
    if set(a) != set(b):
        raise ValueError('Нельзя сравнить ответы с разными id')
    return dict(improved=[i for i in a if not a[i]['correct'] and b[i]['correct']],
                regressed=[i for i in a if a[i]['correct'] and not b[i]['correct']])


def evaluate():
    import mlflow
    config, rows, behavioral, baseline, proof, protocol_hash = inputs()
    candidate = candidate_model()
    models = dict(baseline=baseline, candidate=candidate)
    manifest, _ = manifest_for(candidate)
    reports = {}
    for role, spec in models.items():
        outputs = generate(spec, manifest, config, rows + behavioral)
        by_id = {o['id']: o for o in outputs}
        if len(by_id) != len(outputs) or set(by_id) != {r['id'] for r in rows + behavioral}:
            raise ValueError(f'{role}: generation вернула несовместимые id')
        write_json(f'metrics/{role}-outputs.json', outputs)
        report = score(rows, [by_id[r['id']] for r in rows])
        report.update(status='complete', role=role, model=spec, protocol_sha256=protocol_hash,
                      golden_sha256=fingerprint(rows), baseline_spec_sha256=fingerprint(baseline),
                      behavioral=score(behavioral, [by_id[r['id']] for r in behavioral]), leakage=proof,
                      outputs_sha256=fingerprint(outputs))
        reports[role] = report
    diff = answer_diff(reports['baseline'], reports['candidate'])
    mlflow.set_experiment('hw67-evaluation')
    with mlflow.start_run(run_name='same-protocol-model-comparison') as run:
        for role, report in reports.items():
            report['evaluation_run_id'] = run.info.run_id
            write_json(f'metrics/{role}-eval.json', report)
            mlflow.log_metrics({f'{role}.{key}': value for key, value in report['overall'].items()
                                if type(value) in (int, float)})
        mlflow.set_tags(dict(kind='evaluation', candidate_run_id=candidate['run_id'],
                             protocol_sha256=protocol_hash, golden_sha256=fingerprint(rows),
                             roles=','.join(models)))
        write_json('metrics/eval-diff.json', diff)
        write_json('metrics/evaluation-models.json', dict(evaluation_run_id=run.info.run_id,
                                                        protocol_sha256=protocol_hash, models=models))
        # При повторном запуске не включаем артефакты старых ролей/negative-*.
        for role in reports:
            mlflow.log_artifact(f'metrics/{role}-eval.json', 'evaluation')
            mlflow.log_artifact(f'metrics/{role}-outputs.json', 'evaluation')
        for path in ('metrics/eval-diff.json', 'metrics/evaluation-models.json',
                     'evaluation/params.json', str(BASELINE)):
            mlflow.log_artifact(path, 'evaluation')
        mlflow.log_dict(proof, 'evaluation/leakage.json')
    print('Отчёты всех ролей и сырые ответы сохранены. Далее: make eval; только после PASS - make promote')


def validated_reports():
    config, rows, behavioral, baseline_spec, _, protocol_hash = inputs()
    models = dict(baseline=baseline_spec, candidate=candidate_model())
    reports = {role: read_json(f'metrics/{role}-eval.json') for role in models}
    for role, report in reports.items():
        if report.get('status') != 'complete':
            raise ValueError(f'{role}: оценка не завершена')
        if report.get('role') != role or report.get('model') != models[role]:
            raise ValueError(f'{role}: в отчёте другая роль или модель')
        if (report.get('protocol_sha256') != protocol_hash or report.get('golden_sha256') != fingerprint(rows)
                or report.get('baseline_spec_sha256') != fingerprint(baseline_spec)):
            raise ValueError('Изменились данные/протокол/модели/код оценки; повторите make evaluate')
        outputs = read_json(f'metrics/{role}-outputs.json')
        if fingerprint(outputs) != report.get('outputs_sha256'):
            raise ValueError(f'{role}: сырые ответы изменены после оценки')
        by_id = {o['id']: o for o in outputs}
        if len(by_id) != len(outputs) or set(by_id) != {r['id'] for r in rows + behavioral}:
            raise ValueError('Изменились id сырых ответов')
        recalculated = score(rows, [by_id[r['id']] for r in rows])
        if any(report.get(key) != value for key, value in recalculated.items()):
            raise ValueError(f'{role}: метрики/эталоны/таксономия не соответствуют сырым ответам')
        if report.get('behavioral') != score(behavioral, [by_id[r['id']] for r in behavioral]):
            raise ValueError(f'{role}: behavioral-отчёт не соответствует ответам')
    evaluation_id = reports['candidate'].get('evaluation_run_id')
    if (not isinstance(evaluation_id, str) or not evaluation_id.strip()
            or any(report.get('evaluation_run_id') != evaluation_id for report in reports.values())):
        raise ValueError('Отчёты должны относиться к одному конкретному evaluation run')
    expected_manifest = dict(evaluation_run_id=evaluation_id, protocol_sha256=protocol_hash, models=models)
    if read_json('metrics/evaluation-models.json') != expected_manifest:
        raise ValueError('Манифест моделей не соответствует оценке')
    if read_json('metrics/eval-diff.json') != answer_diff(reports['baseline'], reports['candidate']):
        raise ValueError('Diff baseline/candidate не соответствует ответам')
    return reports


def current_reports():
    reports = validated_reports()
    return reports['baseline'], reports['candidate']

def report_evidence(baseline, candidate):
    """Вызывать только после успешной проверки raw, протокола и реестра."""
    return dict(reports_verified=True, evaluation_run_id=candidate['evaluation_run_id'],
                baseline_model=baseline['model'], candidate_model=candidate['model'],
                protocol_sha256=candidate['protocol_sha256'], golden_sha256=candidate['golden_sha256'],
                baseline_spec_sha256=candidate['baseline_spec_sha256'],
                baseline_outputs_sha256=baseline['outputs_sha256'],
                candidate_outputs_sha256=candidate['outputs_sha256'])


def record_gate(result, path, kind):
    """Сохранить как PASS, так и отказ; ошибка логирования блокирует выпуск."""
    write_json(path, result)
    import mlflow
    tracking()
    mlflow.set_experiment('hw67-evaluation')
    with mlflow.start_run(run_name=kind) as run:
        result['gate_run_id'] = run.info.run_id
        tags = dict(kind=kind, decision='pass' if result.get('passed') else 'reject',
                    reports_verified=str(result.get('reports_verified', False)).lower())
        if result.get('reports_verified') is True:
            tags.update(evaluation_run_id=result['evaluation_run_id'],
                        candidate_run_id=result['candidate_model']['run_id'],
                        candidate_model_uri=result['candidate_model']['uri'],
                        protocol_sha256=result['protocol_sha256'],
                        golden_sha256=result['golden_sha256'],
                        baseline_spec_sha256=result['baseline_spec_sha256'])
        mlflow.set_tags(tags)
        write_json(path, result)
        mlflow.log_artifact(path, 'evaluation')
    return result


def gate():
    evidence = dict(reports_verified=False)
    try:
        baseline, candidate = current_reports()
        evidence = report_evidence(baseline, candidate)
        result = quality_gate(baseline, candidate, read_json('evaluation/params.json')['critical_slices'])
        if not isinstance(result, dict) or result.get('passed') is not True:
            raise ValueError('Гейт не подтвердил passed=True')
    except Exception as exc:
        result = dict(passed=False, reason=str(exc), **evidence)
        try:
            record_gate(result, 'metrics/quality-gate.json', 'quality-gate')
        except Exception as logging_error:
            # Исходная причина отказа сохраняется даже при недоступном tracking.
            result['tracking_log_error'] = type(logging_error).__name__
            write_json('metrics/quality-gate.json', result)
        raise ValueError(str(exc)) from exc
    result.update(evidence)
    record_gate(result, 'metrics/quality-gate.json', 'quality-gate')
    return candidate


def release():
    candidate = gate()
    spec = candidate['model']
    _, client = tracking()
    version = registry_version(client, spec)
    if (version.tags.get('selection.mode') == EXPLICIT_SELECTION
            or (SELECTION.exists() and read_json(SELECTION).get('mode') == EXPLICIT_SELECTION)):
        raise ValueError('Явный single-run выбор не подтверждает обязательную приёмку полного sweep; champion запрещён')
    # Смена champion выполняется только после успешного gate, candidate остаётся отдельным alias.
    client.set_registered_model_alias(spec['name'], 'champion', spec['version'])
    write_json('metrics/released.json', dict(model=spec, evaluation_run_id=candidate['evaluation_run_id'],
                                           protocol_sha256=candidate['protocol_sha256']))
    print('champion -> ' + spec['uri'])


def update_baseline():
    candidate = gate()
    spec = candidate['model']
    _, client = tracking()
    champion = client.get_model_version_by_alias(spec['name'], 'champion')
    if str(champion.version) != spec['version'] or champion.run_id != spec['run_id']:
        raise ValueError('Сначала примите эту версию: make promote')
    old = read_json(BASELINE)
    write_json(f'evaluation/baseline-history/{fingerprint(old)}.json', old)
    write_json(BASELINE, spec)
    print('Baseline обновлён явно; старый сохранён. Следующая оценка сравнит с принятой версией')


def negative():
    baseline, candidate = current_reports()
    config = read_json('evaluation/params.json')
    rows = read_rows(config['golden'])
    behavioral = read_rows(config['behavioral'])
    # Это самостоятельная согласованная копия raw и пересчитанного отчёта.
    # Исходные ответы, реальные веса и оба alias не изменяются.
    broken_outputs = [dict(id=r['id'], raw='INVALID', error=None) for r in rows + behavioral]
    broken = copy.deepcopy(candidate)
    broken.update(score(rows, broken_outputs[:len(rows)]))
    broken['behavioral'] = score(behavioral, broken_outputs[len(rows):])
    broken['outputs_sha256'] = fingerprint(broken_outputs)
    write_json('metrics/negative-outputs.json', broken_outputs)
    write_json('metrics/negative-eval.json', broken)
    try:
        quality_gate(baseline, broken, config['critical_slices'])
    except ValueError as exc:
        result = dict(rejected=True, passed=False, reason=str(exc),
                      source_outputs_sha256=candidate['outputs_sha256'],
                      corrupted_outputs_sha256=broken['outputs_sha256'],
                      **report_evidence(baseline, candidate),
                      note='Испорчена копия raw; реальные веса и alias не менялись')
        record_gate(result, 'metrics/negative-gate.json', 'negative-gate')
        import mlflow
        with mlflow.start_run(run_id=result['gate_run_id']):
            mlflow.log_artifact('metrics/negative-outputs.json', 'evaluation')
            mlflow.log_artifact('metrics/negative-eval.json', 'evaluation')
        print('PASS: испорченные ответы отклонены, alias не менялся')
        return
    raise ValueError('FAIL: испорченные ответы прошли gate')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['candidate', 'baseline-init', 'evaluate', 'gate', 'promote', 'baseline-update', 'negative'])
    ap.add_argument('--training-run-id')
    ap.add_argument('--reason')
    args = ap.parse_args()
    if (args.training_run_id is not None or args.reason is not None) and args.command != 'candidate':
        ap.error('--training-run-id/--reason разрешены только для candidate')
    if args.training_run_id is not None and (args.reason is None or not args.reason.strip()):
        ap.error('candidate --training-run-id требует явную непустую --reason')
    if args.reason is not None and args.training_run_id is None:
        ap.error('--reason требует --training-run-id')
    commands = {'candidate': prepare_candidate, 'baseline-init': init_baseline, 'evaluate': evaluate,
                'gate': gate, 'promote': release, 'baseline-update': update_baseline, 'negative': negative}
    try:
        if args.command == 'candidate':
            prepare_candidate(args.training_run_id, args.reason)
        else:
            commands[args.command]()
    except (ValueError, KeyError, FileNotFoundError, NotImplementedError) as exc:
        raise SystemExit(f'Отказ: {exc}') from exc


if __name__ == '__main__':
    main()
