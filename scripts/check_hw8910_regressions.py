"""Run identical boundary tests on the committed prototype and repaired project."""

import argparse
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile


PROJECT = Path(__file__).resolve().parents[1]
SEED_TEST = "hw8910-broken/tests/test_release.py"
EXPECTED_INTEGRATION = {
    "test_cache_survives_fresh_process_and_invalidates_content_and_config",
    "test_incomplete_foreign_and_rejected_reports_preserve_pointer_bytes",
    "test_atomic_replace_error_preserves_pointer_and_cleans_temporary_file",
    "test_release_is_idempotent_and_recovers_pointer_journal_failure",
    "test_http_ready_and_generation_wait_for_actual_warmup",
    "test_http_loaded_version_changes_only_after_restart_and_supports_rollback",
    "test_http_invalid_requests_do_not_invoke_backend",
}


def run_suite(code_root, runner, pattern):
    command = [sys.executable, "-B", "-m", "unittest", "discover", "-s",
               str(PROJECT / "tests_hw8910"), "-p", pattern, "-v"]
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                   "PYTHONPATH": str(code_root), "HW8910_CODE_ROOT": str(code_root)}
    result = subprocess.run(command, cwd=runner, env=environment,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
    output = result.stdout
    ran = re.search(r"Ran (\d+) tests?", output)
    failures = sorted(set(re.findall(r"^FAIL: (test_\w+)", output, re.MULTILINE)))
    errors = sorted(set(re.findall(r"^ERROR: (test_\w+)", output, re.MULTILINE)))
    infrastructure_failure = any(marker in output for marker in
        ("ModuleNotFoundError", "ImportError:", "unrecognized arguments:", "can't open file"))
    summary = dict(returncode=result.returncode, tests=int(ran[1]) if ran else None,
                   failed_test_methods=failures, error_test_methods=errors,
                   assertion_failures=int(re.search(r"FAILED \(failures=(\d+)", output)[1])
                   if re.search(r"FAILED \(failures=(\d+)", output) else 0,
                   infrastructure_failure=infrastructure_failure,
                   log_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest())
    return summary, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ref", default="5b8efbc")
    parser.add_argument("--fixed-root", type=Path, default=PROJECT)
    parser.add_argument("--seed-only", action="store_true")
    parser.add_argument("--out", type=Path, help="Explicit directory for the required red/green proof logs")
    args = parser.parse_args(argv)
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.source_ref}^{{commit}}"], cwd=PROJECT, text=True).strip()
    starter = subprocess.check_output(["git", "show", f"{commit}:{SEED_TEST}"], cwd=PROJECT)
    if starter != (PROJECT / "tests_hw8910/test_release.py").read_bytes():
        raise ValueError("The original 13 starter tests were changed")
    test_paths = ["tests_hw8910/test_release.py", "tests_hw8910/test_integration.py", "tests_hw8910/helpers.py"]
    test_hashes = {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in test_paths}
    archive = subprocess.check_output(["git", "archive", commit, "hw8910-broken"], cwd=PROJECT)
    summaries, logs, code_hashes, code_snapshots = {}, {}, {}, {}
    with tempfile.TemporaryDirectory(prefix="hw8910-regressions-") as directory:
        root = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(root, filter="data")
        targets = [("seed", root / "hw8910-broken")]
        if not args.seed_only:
            targets.append(("fixed", args.fixed_root.resolve()))
        for label, code_root in targets:
            snapshot = {name: (code_root / name).read_bytes() for name in ("release.py", "lab.py", "service.py")}
            before = {name: hashlib.sha256(content).hexdigest() for name, content in snapshot.items()}
            for suite, pattern in (("starter", "test_release.py"), ("integration", "test_integration.py")):
                runner = root / f"runner-{label}-{suite}"
                runner.mkdir()
                summary, output = run_suite(code_root, runner, pattern)
                key = f"{label}-{suite}"
                summaries[key], logs[key] = summary, output
                print(f"{key}: tests={summary['tests']} exit={summary['returncode']} "
                      f"failed_methods={len(summary['failed_test_methods'])} errors={len(summary['error_test_methods'])}")
            after = {name: hashlib.sha256((code_root / name).read_bytes()).hexdigest() for name in before}
            if after != before:
                raise ValueError(f"{label} code changed during regression verification")
            code_hashes[label] = after
            code_snapshots[label] = snapshot
    if test_hashes != {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in test_paths}:
        raise ValueError("Test code changed between the seed and fixed runs")
    seed = summaries["seed-integration"]
    red_proven = (seed["returncode"] != 0 and seed["tests"] == len(EXPECTED_INTEGRATION)
                  and set(seed["failed_test_methods"]) == EXPECTED_INTEGRATION
                  and not seed["error_test_methods"] and not seed["infrastructure_failure"]
                  and summaries["seed-starter"]["tests"] == 13
                  and summaries["seed-starter"]["returncode"] != 0)
    green_proven = None if args.seed_only else all(
        summaries[f"fixed-{suite}"]["returncode"] == 0
        and summaries[f"fixed-{suite}"]["tests"] == count
        for suite, count in (("starter", 13), ("integration", len(EXPECTED_INTEGRATION))))
    fixes = None
    if not args.seed_only:
        fixes = "".join("".join(difflib.unified_diff(
            code_snapshots["seed"][name].decode("utf-8").splitlines(keepends=True),
            code_snapshots["fixed"][name].decode("utf-8").splitlines(keepends=True),
            fromfile=f"a/hw8910-broken/{name}", tofile=f"b/{name}"))
            for name in ("release.py", "lab.py", "service.py"))
    proof = dict(source_commit=commit, source_path="hw8910-broken", fixed_root=str(args.fixed_root.resolve()),
                 starter_tests_sha256=hashlib.sha256(starter).hexdigest(),
                 test_files_sha256=test_hashes,
                 tested_code_sha256=code_hashes,
                 same_test_files_for_both_versions=True,
                 fixes_diff_sha256=None if fixes is None else hashlib.sha256(fixes.encode("utf-8")).hexdigest(),
                 expected_regression_test_methods=sorted(EXPECTED_INTEGRATION),
                 seed_regressions_proven=red_proven, fixed_regressions_proven=green_proven,
                 suites=summaries, backend_kind="stub", real_model_loaded=False)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        for key, output in logs.items():
            (args.out / f"{key}.log").write_text(output, encoding="utf-8")
        if fixes is not None:
            (args.out / "fixes.patch").write_text(fixes, encoding="utf-8")
        (args.out / "regressions.json").write_text(
            json.dumps(proof, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(seed_regressions_proven=red_proven, fixed_regressions_proven=green_proven)))
    return 0 if red_proven and (args.seed_only or green_proven) else 1


if __name__ == "__main__":
    raise SystemExit(main())
