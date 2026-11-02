"""Four-stage local Prefect release flow using the shared release operations."""
import argparse
import json
from pathlib import Path
import uuid

from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from prefect.runtime import flow_run

import lab
import release


def preparation_cache_key(context, parameters):
    return lab.preparation_key(Path(parameters['workspace']))


@task(name='prepare-input', cache_key_fn=preparation_cache_key, persist_result=True, retries=0)
def prepare_input(workspace):
    return lab.prepare(Path(workspace))


@task(name='obtain-and-verify-artifact', retries=0, persist_result=True, cache_policy=NO_CACHE)
def obtain_artifact(workspace, manifest_name):
    root = Path(workspace)
    path = lab.workspace_file(root, manifest_name)
    manifest = lab.read(path)
    release.verify_artifact(manifest, root)
    return {'manifest_path': str(path), 'manifest_sha256': release.file_sha256(path),
            'model_id': manifest['model_id'], 'artifact_uri': manifest['artifact_uri']}


@task(name='check-full-admission', retries=0, persist_result=True, cache_policy=NO_CACHE)
def check_admission(workspace, descriptor, report_name):
    root = Path(workspace)
    path = Path(descriptor['manifest_path'])
    if release.file_sha256(path) != descriptor['manifest_sha256']:
        raise ValueError('Manifest changed after artifact verification')
    manifest = lab.read(path)
    report_path = lab.workspace_file(root, report_name)
    report = lab.read(report_path)
    lab.check_admission(root, manifest, report)
    return {**descriptor, 'report_path': str(report_path),
            'report_sha256': release.file_sha256(report_path),
            'evaluation_run_id': report['evaluation_run_id']}


@task(name='publish-accepted-pointer', retries=0, persist_result=True, cache_policy=NO_CACHE)
def publish_accepted(workspace, descriptor, prepared):
    for name in ('manifest', 'report'):
        if release.file_sha256(descriptor[f'{name}_path']) != descriptor[f'{name}_sha256']:
            raise ValueError(f'{name} changed after admission')
    return lab.publish_release(Path(workspace), lab.read(descriptor['manifest_path']),
                               lab.read(descriptor['report_path']), prepared)


@flow(name='hw8910-model-release', log_prints=True, retries=0)
def release_flow(workspace, report_name='report-B.json', manifest_name='manifest-B.json',
                 fail_after_prepare=False):
    root = Path(workspace).resolve()
    attempt_id, before = str(uuid.uuid4()), lab.pointer_sha(root)
    states, futures = {}, {}
    prefect_id = str(flow_run.id)
    try:
        prepared_future = prepare_input.submit(str(root))
        futures['prepare'] = prepared_future
        prepared = prepared_future.result()
        states['prepare'] = prepared_future.state.name
        if fail_after_prepare:
            raise RuntimeError('Инъекция сбоя после подготовки')
        artifact_future = obtain_artifact.submit(str(root), manifest_name)
        futures['artifact'] = artifact_future
        descriptor = artifact_future.result()
        states['artifact'] = artifact_future.state.name
        admission_future = check_admission.submit(str(root), descriptor, report_name)
        futures['admission'] = admission_future
        descriptor = admission_future.result()
        states['admission'] = admission_future.state.name
        publication_future = publish_accepted.submit(str(root), descriptor, prepared)
        futures['publication'] = publication_future
        result = publication_future.result()
        states['publication'] = publication_future.state.name
        lab.record_attempt(root, attempt_id, result['status'], before, **result,
                           execution='prefect', flow_run_id=prefect_id,
                           task_states=states, prepared_key=prepared['cache_key'])
        return {**result, 'flow_run_id': prefect_id, 'task_states': states,
                'prepared_key': prepared['cache_key']}
    except Exception as exc:
        states.update({name: future.state.name for name, future in futures.items()})
        lab.record_attempt(root, attempt_id, 'rejected' if isinstance(exc, ValueError) else 'failed', before,
                           execution='prefect', flow_run_id=prefect_id, task_states=states,
                           error=f'{type(exc).__name__}: {exc}')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--report', default='report-B.json')
    parser.add_argument('--manifest', default='manifest-B.json')
    parser.add_argument('--fail-after-prepare', action='store_true')
    args = parser.parse_args()
    result = release_flow(str(args.workspace), args.report, args.manifest, args.fail_after_prepare)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
