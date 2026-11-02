"""The same seven real-boundary regressions run against seed and fixed code."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from helpers import lab, read_json, request, run_python, server, wait_ready, write_json


class ReleaseIntegration(unittest.TestCase):
    def observe(self, event, **values):
        print("OBSERVATION " + json.dumps({"test": self._testMethodName, "event": event, **values},
                                         ensure_ascii=False, sort_keys=True), flush=True)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="hw8910-integration-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cwd = self.root / "unrelated-cwd"
        self.cwd.mkdir()

    def initialize(self, name="workspace"):
        workspace = self.root / name
        result = lab(workspace, self.cwd, "init")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return workspace

    def successful_run(self, workspace, *arguments):
        result = lab(workspace, self.cwd, "run", *arguments)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_cache_survives_fresh_process_and_invalidates_content_and_config(self):
        """Failure/retry reuses preparation, but input and every config change do not."""
        workspace = self.initialize()
        first = lab(workspace, self.cwd, "run", "--fail-after-prepare")
        self.assertNotEqual(first.returncode, 0, "The requested failure after preparation must exit nonzero")
        self.assertIn("PREPARED", first.stdout, first.stdout + first.stderr)
        second_cwd = self.root / "retry-cwd"
        second_cwd.mkdir()
        retry = lab(workspace, second_cwd, "run")
        self.assertEqual(retry.returncode, 0, retry.stdout + retry.stderr)
        self.observe("fresh_process_retry", first_exit=first.returncode, retry_exit=retry.returncode,
                     retry_output=retry.stdout.strip())
        self.assertIn("CACHE_HIT", retry.stdout, "A fresh process must reuse the persisted result")

        (workspace / "input.txt").write_text("Changed bytes at the same path\n", encoding="utf-8")
        changed = self.successful_run(workspace)
        self.observe("input_changed_same_path", output=changed.stdout.strip())
        self.assertIn("PREPARED", changed.stdout,
                      "Changing input bytes at an unchanged path must invalidate preparation: " + changed.stdout)
        expected_sha = hashlib.sha256((workspace / "input.txt").read_bytes()).hexdigest()
        self.assertIn(expected_sha, changed.stdout)
        for key in ("tokenizer_revision", "template_sha256", "max_length", "code_revision"):
            with self.subTest(config_field=key):
                config = read_json(workspace / "config.json")
                previous = config[key]
                if key == "max_length":
                    value = previous + 1
                elif key == "template_sha256":
                    value = hashlib.sha256((previous + "-integration-change").encode()).hexdigest()
                else:
                    value = previous + "-integration-change"
                self.assertNotEqual(value, previous, "The invalidation scenario must actually change the setting")
                config[key] = value
                write_json(workspace / "config.json", config)
                updated = self.successful_run(workspace)
                self.observe("config_changed", field=key, output=updated.stdout.strip())
                self.assertIn("PREPARED", updated.stdout, f"Changing {key} must invalidate preparation: {updated.stdout}")

    def test_incomplete_foreign_and_rejected_reports_preserve_pointer_bytes(self):
        """Each rejection uses a new workspace and a separate actual release process."""
        for index, report_name in enumerate(("report-incomplete.json", "report-A.json", "report-reject.json")):
            with self.subTest(report=report_name):
                workspace = self.initialize(f"reject-{index}")
                target = workspace / "accepted.json"
                before = target.read_bytes()
                rejected = lab(workspace, self.cwd, "run", "--report", report_name)
                self.observe("report_decision", report=report_name, exit_code=rejected.returncode,
                             accepted_before_sha256=hashlib.sha256(before).hexdigest(),
                             accepted_after_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
                self.assertNotEqual(rejected.returncode, 0,
                                    f"Incomplete/foreign/refused report {report_name} must fail, observed: {rejected.stdout}")
                self.assertEqual(target.read_bytes(), before, "Rejection must preserve the exact accepted bytes")
                self.assertEqual(read_json(target)["model_id"], "fixture-A")

    def test_atomic_replace_error_preserves_pointer_and_cleans_temporary_file(self):
        """A real child process injects a replacement failure without damaging the disk."""
        workspace = self.initialize()
        code = """
import json, sys
from pathlib import Path
from unittest.mock import patch
import release
root = Path(sys.argv[1])
target = root / 'accepted.json'
before = target.read_bytes()
failed = False
with patch('os.replace', side_effect=OSError('INJECTED_REPLACE_FAILURE')):
    try:
        release.publish(json.loads((root/'manifest-B.json').read_text()),
                        json.loads((root/'report-B.json').read_text()), target)
    except OSError:
        failed = True
print(json.dumps({'replace_failure_propagated': failed,
                  'old_pointer_preserved': target.read_bytes() == before,
                  'temporary_files': [str(p.name) for p in root.glob('*.tmp')]}))
"""
        result = run_python(["-c", code, workspace], self.cwd)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        evidence = json.loads(result.stdout.strip().splitlines()[-1])
        self.observe("replace_error", **evidence)
        self.assertTrue(evidence["replace_failure_propagated"], f"Atomic replacement must be attempted: {evidence}")
        self.assertTrue(evidence["old_pointer_preserved"], f"Failed replacement changed accepted bytes: {evidence}")
        self.assertEqual(evidence["temporary_files"], [], evidence)

    def test_release_is_idempotent_and_recovers_pointer_journal_failure(self):
        """A repeated release and an interrupted pointer/journal write each emit one event."""
        workspace = self.initialize("repeat")
        self.successful_run(workspace)
        self.successful_run(workspace)
        records = [json.loads(line) for line in (workspace / "publications.jsonl").read_text().splitlines()]
        self.observe("repeated_release", attempts=2, publication_events=len(records))
        self.assertEqual(len(records), 1, f"Two attempts created duplicate publication events: {records}")

        report = read_json(workspace / "report-B.json")
        report["evaluation_run_id"] = "fixture-eval-B-rechecked"
        write_json(workspace / "report-B.json", report)
        self.successful_run(workspace)
        records = (workspace / "publications.jsonl").read_text().splitlines()
        self.observe("new_evaluation_same_model", publication_events=len(records))
        self.assertEqual(len(records), 2, "A new evaluation of the same model must have a distinct release identity")
        manifest = read_json(workspace / "manifest-B.json")
        manifest["runtime"]["template_sha256"] = "f" * 64
        canonical = json.dumps(manifest["runtime"], sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8")
        manifest["runtime_sha256"] = hashlib.sha256(canonical).hexdigest()
        report["runtime_sha256"] = manifest["runtime_sha256"]
        write_json(workspace / "manifest-B.json", manifest)
        write_json(workspace / "report-B.json", report)
        self.successful_run(workspace)
        self.observe("new_runtime_same_model", publication_events=len((workspace / "publications.jsonl").read_text().splitlines()))
        self.assertEqual(len((workspace / "publications.jsonl").read_text().splitlines()), 3,
                         "The same model with a newly evaluated runtime is a different release")

        recovery = self.initialize("recovery")
        code = """
import sys
from pathlib import Path
from unittest.mock import patch
import lab, release
original = release.publish
def fail_after_pointer(*args, **kwargs):
    original(*args, **kwargs)
    raise RuntimeError('INJECTED_AFTER_POINTER')
with patch.object(release, 'publish', fail_after_pointer):
    lab.run(Path(sys.argv[1]), 'report-B.json')
"""
        crashed = run_python(["-c", code, recovery], self.cwd)
        self.assertNotEqual(crashed.returncode, 0, "Pointer/journal fault must be observable to the calling process")
        self.assertIn("INJECTED_AFTER_POINTER", crashed.stderr)
        self.assertEqual(read_json(recovery / "accepted.json")["model_id"], "fixture-B")
        self.successful_run(recovery)
        self.successful_run(recovery)
        recovered = [json.loads(line) for line in (recovery / "publications.jsonl").read_text().splitlines()]
        self.observe("pointer_journal_recovery", crash_exit=crashed.returncode,
                     loaded_pointer=read_json(recovery / "accepted.json")["model_id"], publication_events=len(recovered))
        self.assertEqual(len(recovered), 1, f"Recovery/retry duplicated the committed publication: {recovered}")

    def test_http_ready_and_generation_wait_for_actual_warmup(self):
        """Liveness is independent from readiness and does not permit early computation."""
        workspace = self.initialize()
        with server(workspace, self.cwd, warmup_seconds=2) as url:
            self.assertEqual(request(url, "/health")[0], 200)
            _, before = request(url, "/debug")
            self.assertFalse(before["warmup_finished"], "The test must observe the real warmup interval")
            status, body = request(url, "/ready")
            self.observe("during_warmup", ready_status=status, debug=before)
            self.assertEqual(status, 503, f"Ready became green before warmup: {body}")
            status, body = request(url, "/generate", {"prompt": "A film description", "max_new_tokens": 32})
            self.assertEqual(status, 503, f"Generation was served before warmup: {body}")
            _, after = request(url, "/debug")
            self.assertEqual(after["calls"], before["calls"], "Unready requests must not invoke the backend")
            wait_ready(url)
            self.assertEqual(request(url, "/generate", {"prompt": "A film description", "max_new_tokens": 32})[0], 200)

    def test_http_loaded_version_changes_only_after_restart_and_supports_rollback(self):
        """Publish B while A is loaded, restart onto B, then atomically restore A and reload."""
        workspace = self.initialize()
        with server(workspace, self.cwd) as url:
            wait_ready(url)
            _, initial = request(url, "/generate", {"prompt": "Version probe", "max_new_tokens": 32})
            self.assertEqual(initial["model_id"], "fixture-A")
            self.observe("initial_loaded_A", response=initial)
            self.assertIn("fixture-A", initial["answer"])
            self.successful_run(workspace)
            self.assertEqual(read_json(workspace / "accepted.json")["model_id"], "fixture-B")
            _, still_a = request(url, "/generate", {"prompt": "Version probe", "max_new_tokens": 32})
            self.observe("accepted_B_while_A_loaded", accepted_id=read_json(workspace / "accepted.json")["model_id"], response=still_a)
            self.assertEqual(still_a["model_id"], "fixture-A",
                             f"Metadata must identify loaded A, not newly accepted B: {still_a}")
            self.assertIn("fixture-A", still_a["answer"])
            self.assertEqual(still_a["runtime_sha256"], initial["runtime_sha256"])
        with server(workspace, self.cwd) as url:
            wait_ready(url)
            _, loaded_b = request(url, "/generate", {"prompt": "Version probe", "max_new_tokens": 32})
            self.observe("restarted_onto_B", response=loaded_b)
            self.assertEqual(loaded_b["model_id"], "fixture-B")
            self.assertIn("fixture-B", loaded_b["answer"])

        rollback = """
import json, sys
from pathlib import Path
import release
root = Path(sys.argv[1])
release.publish(json.loads((root/'manifest-A.json').read_text()),
                json.loads((root/'report-A.json').read_text()), root/'accepted.json')
"""
        result = run_python(["-c", rollback, workspace], self.cwd)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with server(workspace, self.cwd) as url:
            wait_ready(url)
            _, loaded_a = request(url, "/generate", {"prompt": "Version probe", "max_new_tokens": 32})
            self.observe("rollback_and_restart_A", response=loaded_a)
            self.assertEqual(loaded_a["model_id"], "fixture-A")
            self.assertIn("fixture-A", loaded_a["answer"])

    def test_http_invalid_requests_do_not_invoke_backend(self):
        """Invalid shape, prompt, types, bounds and generation options fail before computation."""
        workspace = self.initialize()
        invalid = [
            {"prompt": "", "max_new_tokens": 32},
            {"prompt": "   ", "max_new_tokens": 32},
            {"prompt": None, "max_new_tokens": 32},
            {"prompt": [], "max_new_tokens": 32},
            {"prompt": "valid", "max_new_tokens": 0},
            {"prompt": "valid", "max_new_tokens": 100000},
            {"prompt": "valid", "max_new_tokens": "32"},
            {"prompt": "valid", "max_new_tokens": True},
            {"prompt": "valid", "max_new_tokens": 32, "temperature": 0.7},
        ]
        with server(workspace, self.cwd) as url:
            before = wait_ready(url)
            for payload in invalid:
                with self.subTest(payload=payload):
                    status, body = request(url, "/generate", payload)
                    _, debug = request(url, "/debug")
                    self.observe("invalid_request", http_status=status,
                                 calls_before=before["calls"], calls_after=debug["calls"])
                    self.assertTrue(400 <= status < 500, f"Invalid request was accepted: {payload!r}; {body}")
                    self.assertEqual(debug["calls"], before["calls"], "Rejected input invoked the backend")
            _, final = request(url, "/debug")
            self.assertEqual(final["calls"], before["calls"])
            status, body = request(url, "/generate", {"prompt": "A valid film description", "max_new_tokens": 32})
            self.assertEqual(status, 200, body)
            self.assertIn("STUB ONLY", body["answer"])
            self.assertEqual(body["model_id"], "fixture-A")
            self.observe("valid_request", http_status=status, response=body)
            self.assertEqual(request(url, "/debug")[1]["calls"], before["calls"] + 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
