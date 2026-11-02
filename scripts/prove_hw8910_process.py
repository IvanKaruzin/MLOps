"""Reproduce Prefect cache/recovery/publication evidence without model weights."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
import lab
import release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=REPOSITORY / 'docs/hw8910/evidence')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    records, logs = [], []
    with tempfile.TemporaryDirectory(prefix='hw8910-process-proof-') as directory:
        temporary = Path(directory)
        home, workspace, recovery = temporary / 'prefect', temporary / 'lab', temporary / 'recovery'
        home.mkdir()
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PREFECT_HOME=str(home),
                           PREFECT_API_URL=f'http://127.0.0.1:{port}/api',
                           PREFECT_SERVER_ANALYTICS_ENABLED='false', PREFECT_LOGGING_LEVEL='INFO',
                           PREFECT_SERVER_EPHEMERAL_ENABLED='false')
        environment.pop('PREFECT_API_KEY', None)
        server_log = temporary / 'server.log'
        with server_log.open('w', encoding='utf-8') as server_stream:
            server = subprocess.Popen([sys.executable, '-B', '-m', 'prefect', 'server', 'start',
                                       '--host', '127.0.0.1', '--port', str(port),
                                       '--no-ui', '--no-scheduler', '--analytics-off'],
                                      cwd=temporary, env=environment,
                                      stdout=server_stream, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 90
                while True:
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health', timeout=1) as response:
                            if response.status == 200:
                                break
                    except (OSError, urllib.error.URLError):
                        pass
                    if server.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError('Prefect server failed to start: ' + server_log.read_text()[-3000:])
                    time.sleep(.25)

                def step(label, command, root, expected_success=True):
                    cwd = temporary / 'cwd' / label
                    cwd.mkdir(parents=True)
                    before = lab.pointer_sha(root)
                    started = time.monotonic()
                    process = subprocess.run(command, cwd=cwd, env=environment,
                                             text=True, capture_output=True, timeout=150)
                    attempts = lab.read_journal(root / 'attempts.jsonl') if root.exists() else []
                    record = dict(label=label, command=command, cwd=str(cwd), exit_code=process.returncode,
                                  expected_success=expected_success,
                                  elapsed_seconds=time.monotonic() - started,
                                  accepted_before_sha256=before, accepted_after_sha256=lab.pointer_sha(root),
                                  cache_entries=len(list((root / 'cache').glob('*.json'))),
                                  publication_count=len(lab.read_journal(root / 'publications.jsonl')),
                                  last_attempt=attempts[-1] if attempts else None)
                    records.append(record)
                    logs.append('\n### ' + label + '\n' + json.dumps(record, ensure_ascii=False, sort_keys=True)
                                + '\nSTDOUT\n' + process.stdout + '\nSTDERR\n' + process.stderr)
                    if (process.returncode == 0) is not expected_success:
                        raise AssertionError(f'{label}: unexpected exit {process.returncode}\n{process.stderr[-3000:]}')
                    print(label, 'exit', process.returncode, 'publications', record['publication_count'], flush=True)
                    return record

                def flow_command(root, *extra):
                    return [sys.executable, '-B', str(REPOSITORY / 'flow.py'), '--workspace', str(root), *extra]

                step('01-init', [sys.executable, '-B', str(REPOSITORY / 'lab.py'),
                                 '--workspace', str(workspace), 'init'], workspace)
                failed = step('02-fail-after-prepare', flow_command(workspace, '--fail-after-prepare'),
                              workspace, expected_success=False)
                assert failed['accepted_before_sha256'] == failed['accepted_after_sha256']
                assert failed['last_attempt']['task_states']['prepare'].lower() == 'completed'
                cached = step('03-recover-cache', flow_command(workspace), workspace)
                assert cached['last_attempt']['task_states']['prepare'].lower() == 'cached'
                assert cached['publication_count'] == 1
                repeated = step('04-repeat-release', flow_command(workspace), workspace)
                assert repeated['publication_count'] == 1
                assert repeated['accepted_before_sha256'] == repeated['accepted_after_sha256']
                (workspace / 'input.txt').write_text('Первый пример\nВторой пример\nНовый пример\n', encoding='utf-8')
                changed = step('05-changed-input', flow_command(workspace), workspace)
                assert changed['last_attempt']['task_states']['prepare'].lower() == 'completed'
                assert changed['cache_entries'] == 2 and changed['publication_count'] == 1
                config = lab.read(workspace / 'config.json')
                config['code_revision'] += '-changed'
                lab.write(workspace / 'config.json', config)
                configured = step('06-changed-config', flow_command(workspace), workspace)
                assert configured['last_attempt']['task_states']['prepare'].lower() == 'completed'
                assert configured['cache_entries'] == 3 and configured['publication_count'] == 1
                report = lab.read(workspace / 'report-B.json')
                report['evaluation_run_id'] += '-rechecked'
                lab.write(workspace / 'report-B.json', report)
                reevaluated = step('07-new-admission', flow_command(workspace), workspace)
                assert reevaluated['publication_count'] == 2
                rejected = step('08-rejected-report', flow_command(workspace, '--report', 'report-reject.json'),
                                workspace, expected_success=False)
                assert rejected['accepted_before_sha256'] == rejected['accepted_after_sha256']
                assert rejected['publication_count'] == 2
                step('09-recovery-init', [sys.executable, '-B', str(REPOSITORY / 'lab.py'),
                                          '--workspace', str(recovery), 'init'], recovery)
                fault_code = '\n'.join([
                    'import sys', f'sys.path.insert(0, {str(REPOSITORY)!r})',
                    'from unittest.mock import patch', 'import release, flow',
                    'original = release.publish', 'def crash(*args, **kwargs):',
                    '    original(*args, **kwargs)', "    raise RuntimeError('INJECTED_AFTER_POINTER')",
                    "with patch.object(release, 'publish', side_effect=crash):",
                    f'    flow.release_flow({str(recovery)!r})'])
                pointer_fault = step('10-pointer-journal-fault', [sys.executable, '-B', '-c', fault_code],
                                     recovery, expected_success=False)
                assert pointer_fault['accepted_before_sha256'] != pointer_fault['accepted_after_sha256']
                assert (recovery / 'publication-intent.json').exists()
                assert pointer_fault['publication_count'] == 0
                recovered = step('11-pointer-journal-recovery', flow_command(recovery), recovery)
                assert recovered['publication_count'] == 1
                assert not (recovery / 'publication-intent.json').exists()
                replay = step('12-recovered-repeat', flow_command(recovery), recovery)
                assert replay['publication_count'] == 1
                assert replay['accepted_before_sha256'] == replay['accepted_after_sha256']
                for label, root in [('main', workspace), ('recovery', recovery)]:
                    for filename in ('attempts.jsonl', 'publications.jsonl'):
                        (out / f'process-{label}-{filename}').write_bytes((root / filename).read_bytes())
            finally:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)
                (out / 'process-runs.log').write_text(''.join(logs), encoding='utf-8')
                (out / 'process-runs.jsonl').write_bytes(b''.join(release.canonical(record) + b'\n' for record in records))

    summary = dict(passed=True, synthetic=True, model_execution=False,
                   timestamp_utc=datetime.now(timezone.utc).isoformat(),
                   source_commit='5b8efbc', framework='Prefect',
                   prefect_version=importlib.metadata.version('prefect'),
                   subprocess_runs=len(records),
                   flow_run_ids=[record['last_attempt']['flow_run_id'] for record in records
                                 if record['last_attempt'] and record['last_attempt'].get('flow_run_id')],
                   server_stopped=True, workspace_removed=True,
                   preparation_cache_verified=True, input_and_config_invalidation_verified=True,
                   publication_idempotency_verified=True, admission_identity_verified=True,
                   pointer_journal_recovery_verified=True, rejection_preserves_pointer=True,
                   source_sha256={name: release.file_sha256(REPOSITORY / name)
                                  for name in ('release.py', 'lab.py', 'flow.py')})
    lab.write(out / 'process-summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
