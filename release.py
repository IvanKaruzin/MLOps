"""Strict release metadata, artifact integrity and atomic pointer publication."""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import unquote, urlsplit


HASH_FIELDS = ('model_sha256', 'protocol_sha256', 'golden_sha256', 'runtime_sha256')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def cache_key_for_bytes(payload, config):
    if not isinstance(config, dict):
        raise ValueError('Preparation config must be an object')
    return digest(canonical({'data_sha256': digest(payload), 'config': config}))


def cache_key(data_path: Path, config: dict) -> str:
    return cache_key_for_bytes(Path(data_path).read_bytes(), config)


def is_digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def validate_release(manifest: dict, report: dict) -> None:
    """Metadata only: artifact I/O and the full quality gate are separate."""
    if not isinstance(manifest, dict) or not isinstance(report, dict):
        raise ValueError('Manifest and report must be objects')
    if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1:
        raise ValueError('Unsupported manifest schema_version')
    for key in ('model_id', 'artifact_uri'):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f'Manifest requires a nonempty {key}')
    for key in HASH_FIELDS:
        if not is_digest(manifest.get(key)):
            raise ValueError(f'Manifest requires lowercase SHA-256 {key}')
    runtime = manifest.get('runtime')
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError('Manifest requires a nonempty runtime object')
    try:
        runtime_sha = digest(canonical(runtime))
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Runtime is not canonical finite JSON') from exc
    if runtime_sha != manifest['runtime_sha256']:
        raise ValueError('Runtime content does not match runtime_sha256')
    if report.get('status') != 'complete' or report.get('passed') is not True:
        raise ValueError('Candidate evaluation is incomplete or rejected')
    if not isinstance(report.get('evaluation_run_id'), str) or not report['evaluation_run_id'].strip():
        raise ValueError('Report requires a nonempty evaluation_run_id')
    for key in ('model_id', *HASH_FIELDS):
        if report.get(key) != manifest[key]:
            raise ValueError(f'Report is bound to a different {key}')
    metrics = report.get('metrics')
    if not isinstance(metrics, dict):
        raise ValueError('Report requires metrics')
    for key in ('accuracy', 'format_rate'):
        value = metrics.get(key)
        if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
            raise ValueError(f'Invalid report metric {key}')


def verify_artifact(manifest, workspace):
    """Rehash declared real files and reject unsafe paths/aggregate mismatches."""
    files = manifest.get('file_hashes')
    if not isinstance(files, dict) or not files:
        raise ValueError('Artifact requires a nonempty file_hashes map')
    uri = manifest.get('artifact_uri')
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError('Artifact URI is missing')
    if uri.startswith('file://'):
        parsed = urlsplit(uri)
        if parsed.netloc not in ('', 'localhost') or parsed.query or parsed.fragment:
            raise ValueError('Only local file artifact URIs are supported')
        location = Path(unquote(parsed.path))
    elif '://' in uri or uri.startswith(('models:', 'runs:')):
        raise ValueError('Resolve artifact URI to a verified local directory first')
    else:
        location = Path(uri)
    if not location.is_absolute():
        root = Path(workspace).resolve()
        location = (root / location).resolve()
        if not location.is_relative_to(root):
            raise ValueError('Relative artifact URI escapes workspace')
    location = location.resolve(strict=True)
    base = location if location.is_dir() else location.parent
    actual = {}
    for name, expected in files.items():
        if (not isinstance(name, str) or not name or '\\' in name
                or PurePosixPath(name).is_absolute()
                or any(part in ('', '.', '..') for part in name.split('/')) or not is_digest(expected)):
            raise ValueError('Unsafe artifact path or invalid file digest')
        path = (base / name).resolve(strict=True)
        if not path.is_relative_to(base) or not path.is_file():
            raise ValueError('Artifact file escapes its root or is not a file')
        actual[name] = file_sha256(path)
        if actual[name] != expected:
            raise ValueError(f'Artifact file hash changed: {name}')
    if location.is_file() and location.name not in files:
        raise ValueError('File artifact URI is absent from file_hashes')
    if digest(canonical(actual)) != manifest.get('model_sha256'):
        raise ValueError('Actual artifact files do not match model_sha256')
    return actual


def atomic_write_bytes(target, payload):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and not target.is_symlink() and target.read_bytes() == payload:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', prefix=f'.{target.name}.', suffix='.tmp',
                                         dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json_atomic(target, value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         indent=2, allow_nan=False).encode('utf-8') + b'\n'
    atomic_write_bytes(target, payload)


def publish(manifest: dict, report: dict, target: Path) -> None:
    validate_release(manifest, report)
    write_json_atomic(target, manifest)
