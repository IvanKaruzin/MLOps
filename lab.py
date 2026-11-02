"""Shared release operations and compatible lightweight STUB laboratory CLI.

Production orchestration is flow.py. Both paths use these same operations;
no training is performed. Workspaces hold cache, attempts and pointers.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import release


canonical = release.canonical
digest = release.digest


def read(path):
    def reject(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      parse_constant=reject, object_pairs_hook=unique)


def write(path, value):
    release.write_json_atomic(path, value)


def workspace_file(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Workspace file escapes workspace')
    return path


def initialize(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError('Рабочий каталог уже существует. Выберите новый --workspace.')
    root.mkdir(parents=True)
    runtime = {'base_revision': 'fixture-only', 'tokenizer_sha256': 'a' * 64,
               'template_sha256': 'b' * 64,
               'generation': {'max_new_tokens': 32, 'do_sample': False}, 'dtype': 'float32'}
    for version in ('A', 'B'):
        artifact = root / f'model-{version}.txt'
        artifact.write_text(f'STUB ONLY — version {version}\n', encoding='utf-8')
        files = {artifact.name: release.file_sha256(artifact)}
        manifest = {'schema_version': 1, 'model_id': f'fixture-{version}', 'synthetic': True,
                    'artifact_uri': artifact.name, 'model_sha256': digest(canonical(files)),
                    'protocol_sha256': 'd' * 64, 'golden_sha256': 'e' * 64,
                    'runtime_sha256': digest(canonical(runtime)), 'runtime': runtime, 'file_hashes': files}
        report = {key: manifest[key] for key in ('model_id', *release.HASH_FIELDS)}
        report.update(status='complete', passed=True, evaluation_run_id=f'fixture-eval-{version}',
                      metrics={'accuracy': .8, 'format_rate': 1.0}, synthetic=True)
        write(root / f'manifest-{version}.json', manifest)
        write(root / f'report-{version}.json', report)
    write(root / 'accepted.json', read(root / 'manifest-A.json'))
    report = read(root / 'report-B.json')
    report.update(status='running', passed='false')
    write(root / 'report-incomplete.json', report)
    report = read(root / 'report-B.json')
    report['passed'] = False
    write(root / 'report-reject.json', report)
    write(root / 'config.json', {'tokenizer_revision': 'r1', 'template_sha256': 'b' * 64,
                                'max_length': 128, 'code_revision': 'prepare-v2', 'format': 'text'})
    (root / 'input.txt').write_text('Первый пример\nВторой пример\n', encoding='utf-8')
    print('Создан синтетический стенд STUB ONLY:', root)


def preparation_inputs(root):
    root = Path(root)
    config = read(root / 'config.json')
    for key in ('tokenizer_revision', 'template_sha256', 'code_revision'):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f'Preparation config requires {key}')
    if type(config.get('max_length')) is not int or config['max_length'] <= 0:
        raise ValueError('Preparation max_length must be a positive integer')
    payload = (root / 'input.txt').read_bytes()
    return config, payload, release.cache_key_for_bytes(payload, config)


def preparation_key(root):
    return preparation_inputs(root)[2]


def prepare(root):
    """Validate UTF-8 records and summarize lexical tokens, not HF tokenization."""
    root = Path(root)
    config, payload, key = preparation_inputs(root)
    cache = root / 'cache'
    cache.mkdir(exist_ok=True)
    entry = cache / f'{key}.json'
    if entry.exists():
        envelope = read(entry)
        result = envelope.get('result', {})
        if (envelope.get('cache_key') != key or envelope.get('result_sha256') != digest(canonical(result))
                or result.get('input_sha256') != digest(payload) or result.get('config') != config):
            raise ValueError('Corrupted persisted preparation cache')
        print('CACHE_HIT', result['input_sha256'])
        return result
    lines = [line for line in payload.decode('utf-8').splitlines() if line.strip()]
    if not lines:
        raise ValueError('Preparation input is empty')
    texts, identities = [], set()
    for line in lines:
        if config.get('format', 'text') == 'jsonl':
            record = json.loads(line)
            if (not isinstance(record, dict) or not isinstance(record.get('id'), str)
                    or not record['id'] or record['id'] in identities
                    or not isinstance(record.get('input'), str) or not record['input'].strip()):
                raise ValueError('Preparation JSONL requires unique ids and nonempty inputs')
            identities.add(record['id'])
            texts.append(record['input'])
        else:
            texts.append(line)
    lexical = [text.split() for text in texts]
    selected = [tokens[:config['max_length']] for tokens in lexical]
    result = {'input_sha256': digest(payload), 'rows': len(texts), 'config': config,
              'cache_key': key, 'stage': 'utf8_record_validation_and_lexical_summary',
              'lexical_tokens': sum(map(len, selected)),
              'truncated_rows': sum(len(tokens) > config['max_length'] for tokens in lexical),
              'lexical_summary_sha256': digest(canonical(selected))}
    write(entry, {'cache_key': key, 'result_sha256': digest(canonical(result)), 'result': result})
    print('PREPARED', result['input_sha256'])
    return result


def pointer_sha(root):
    path = Path(root) / 'accepted.json'
    return release.file_sha256(path) if path.exists() else None


def release_identity(manifest, report):
    return digest(canonical({'manifest': manifest, 'admission': report}))


def read_journal(path):
    path = Path(path)
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError('Journal records must be objects')
    return records


def write_journal(path, records):
    payload = b''.join(canonical(record) + b'\n' for record in records)
    release.atomic_write_bytes(path, payload)


def record_attempt(root, attempt_id, state, before, **details):
    path = Path(root) / 'attempts.jsonl'
    records = read_journal(path)
    records.append({'attempt_id': attempt_id, 'state': state,
                    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                    'accepted_before_sha256': before, 'accepted_after_sha256': pointer_sha(root), **details})
    write_journal(path, records)


def check_admission(root, manifest, report):
    release.verify_artifact(manifest, root)
    if manifest.get('synthetic') is True and report.get('synthetic') is True:
        if (not manifest.get('model_id', '').startswith('fixture-')
                or manifest.get('runtime', {}).get('base_revision') != 'fixture-only'):
            raise ValueError('Synthetic admission is restricted to the explicit STUB laboratory')
    else:
        from hw8910.runtime import verify_release_evidence
        verify_release_evidence(manifest, report, Path(root))
    release.validate_release(manifest, report)


def recover_publication(root):
    """Complete a previously validated one-writer intent before a new release."""
    root = Path(root)
    pending = root / 'publication-intent.json'
    if not pending.exists():
        return False
    intent = read(pending)
    manifest, report = intent['manifest'], intent['report']
    check_admission(root, manifest, report)
    identity = release_identity(manifest, report)
    if identity != intent.get('release_id'):
        raise ValueError('Pending release identity is corrupted')
    record = intent.get('record')
    bound = {'release_id': identity, 'model_id': manifest['model_id'],
             'model_sha256': manifest['model_sha256'], 'runtime_sha256': manifest['runtime_sha256'],
             'evaluation_run_id': report['evaluation_run_id'], 'report_sha256': digest(canonical(report))}
    if (not isinstance(record, dict) or any(record.get(key) != value for key, value in bound.items())
            or not release.is_digest(record.get('prepared'))
            or not isinstance(record.get('timestamp_utc'), str) or not record['timestamp_utc']):
        raise ValueError('Pending journal record is not bound to the validated release')
    journal = root / 'publications.jsonl'
    records = read_journal(journal)
    identities = [item.get('release_id') for item in records]
    if any(not release.is_digest(value) for value in identities) or len(identities) != len(set(identities)):
        raise ValueError('Duplicate release identities in publication journal')
    release.publish(manifest, report, root / 'accepted.json')
    if identity not in identities:
        records.append(record)
        write_journal(journal, records)
    pending.unlink()
    return True


def publish_release(root, manifest, report, prepared):
    root = Path(root)
    recover_publication(root)
    check_admission(root, manifest, report)
    identity = release_identity(manifest, report)
    journal = root / 'publications.jsonl'
    records = read_journal(journal)
    identities = [item.get('release_id') for item in records]
    if any(not release.is_digest(value) for value in identities) or len(identities) != len(set(identities)):
        raise ValueError('Duplicate release identities in publication journal')
    already_current = (root / 'accepted.json').exists() and read(root / 'accepted.json') == manifest
    if identity in identities and already_current:
        return {'status': 'already_published', 'release_id': identity, 'model_id': manifest['model_id']}
    record = {'release_id': identity, 'model_id': manifest['model_id'],
              'model_sha256': manifest['model_sha256'], 'runtime_sha256': manifest['runtime_sha256'],
              'evaluation_run_id': report['evaluation_run_id'], 'report_sha256': digest(canonical(report)),
              'prepared': prepared['input_sha256'], 'timestamp_utc': datetime.now(timezone.utc).isoformat()}
    write(root / 'publication-intent.json', {'release_id': identity, 'manifest': manifest,
                                          'report': report, 'record': record})
    release.publish(manifest, report, root / 'accepted.json')
    if identity not in identities:
        records.append(record)
        write_journal(journal, records)
    (root / 'publication-intent.json').unlink()
    return {'status': 'published', 'release_id': identity, 'model_id': manifest['model_id']}


def run(root, report_name, fail_after_prepare=False, manifest_name='manifest-B.json'):
    root = Path(root).resolve()
    attempt_id, before = str(uuid.uuid4()), pointer_sha(root)
    try:
        prepared = prepare(root)
        if fail_after_prepare:
            raise RuntimeError('Инъекция сбоя после подготовки')
        manifest = read(workspace_file(root, manifest_name))
        report = read(workspace_file(root, report_name))
        check_admission(root, manifest, report)
        result = publish_release(root, manifest, report, prepared)
        record_attempt(root, attempt_id, result['status'], before, **result,
                       prepared_key=prepared['cache_key'])
        print(result['status'].upper(), manifest['model_id'])
        return result
    except Exception as exc:
        record_attempt(root, attempt_id, 'rejected' if isinstance(exc, ValueError) else 'failed', before,
                       error=f'{type(exc).__name__}: {exc}')
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path('.lab'))
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init')
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--report', default='report-B.json')
    run_parser.add_argument('--manifest', default='manifest-B.json')
    run_parser.add_argument('--fail-after-prepare', action='store_true')
    args = parser.parse_args()
    if args.command == 'init':
        initialize(args.workspace)
    else:
        run(args.workspace, args.report, args.fail_after_prepare, args.manifest)


if __name__ == '__main__':
    main()
