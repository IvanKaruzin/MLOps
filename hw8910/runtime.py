"""Bind HW67 evidence to immutable serving assets; never trains a model."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPOSITORY = Path(__file__).resolve().parents[1]
EVIDENCE_FILES = (
    "metrics/baseline-eval.json", "metrics/candidate-eval.json",
    "metrics/baseline-outputs.json", "metrics/candidate-outputs.json",
    "metrics/evaluation-models.json", "metrics/eval-diff.json",
    "evaluation/params.json", "evaluation/baseline.json",
)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    from evaluation.contract import strict_json_loads
    return strict_json_loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    """Workspace metadata is durable; model files are never written by this module."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def frozen_evaluation():
    """Reuse the complete raw-verified HW67 gate, including its honest rejection."""
    from evaluation.cli import validated_reports
    from evaluation.solution import quality_gate
    reports = validated_reports()
    params = read(REPOSITORY / "evaluation/params.json")
    try:
        gate = quality_gate(reports["baseline"], reports["candidate"], params["critical_slices"])
    except ValueError as exc:
        gate = dict(passed=False, reason=str(exc))
    evidence = {path: file_sha(REPOSITORY / path) for path in EVIDENCE_FILES}
    return reports, gate, evidence


def hf_assets(spec):
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(spec["name"], revision=spec["revision"], local_files_only=True))
    # HF snapshot links resolve into this same repository's blobs directory.
    repository = snapshot.parent.parent
    names = ("config.json", "generation_config.json", "model.safetensors", *TOKENIZER_FILES)
    hashes = {str((snapshot / name).relative_to(repository)): file_sha(snapshot / name) for name in names}
    return repository, snapshot, hashes


def evaluation_runtime(baseline, model_manifest):
    config = read(REPOSITORY / "evaluation/params.json")
    model = model_manifest["model"]
    if ((baseline["name"], baseline["revision"]) != (model["name"], model["revision"])
            or model_manifest["generation"]["system"] != config["system"]
            or model_manifest["generation"]["max_new_tokens"] != config["max_new_tokens"]
            or model["enable_thinking"] != config["enable_thinking"]):
        raise ValueError("Training package and evaluated generation conditions differ")
    repository, snapshot, hashes = hf_assets(baseline)
    tokenizer_hashes = {name: file_sha(snapshot / name) for name in TOKENIZER_FILES}
    template = read(snapshot / "tokenizer_config.json")["chat_template"]
    if not isinstance(template, str) or not template:
        raise ValueError("Expected the pinned tokenizer's explicit chat template")
    runtime = dict(base_model=baseline["name"], base_revision=baseline["revision"],
        tokenizer_sha256=sha(canonical(tokenizer_hashes)), template_sha256=sha(template.encode()),
        generation=dict(system=config["system"], do_sample=False, seed=config["seed"],
                        max_new_tokens=config["max_new_tokens"], enable_thinking=config["enable_thinking"]),
        dtype=model["dtype"], device=model["device"])
    return runtime, repository, hashes


def manifest(model_id, artifact_uri, hashes, runtime, report, **extra):
    return dict(schema_version=1, model_id=model_id, artifact_uri=str(artifact_uri),
        model_sha256=sha(canonical(hashes)), file_hashes=hashes,
        protocol_sha256=report["protocol_sha256"], golden_sha256=report["golden_sha256"],
        runtime=runtime, runtime_sha256=sha(canonical(runtime)), **extra)


def prepare_workspace(workspace):
    """Read an existing training result, materialize metadata, retain the baseline.

    Baseline bootstrap is explicitly allowed by the assignment. It is neither a
    candidate quality PASS nor an MLflow champion operation.
    """
    if Path.cwd().resolve() != REPOSITORY:
        raise ValueError("Prepare from the project root to validate the original HW67 inputs")
    from evaluation.cli import manifest_for
    from release import verify_artifact
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    reports, gate, evidence = frozen_evaluation()
    base, candidate = reports["baseline"], reports["candidate"]
    model_manifest, downloaded_adapter = manifest_for(candidate["model"])
    runtime, repository, base_hashes = evaluation_runtime(base["model"], model_manifest)
    # Do not persist MLflow's temporary download path in a durable manifest.
    adapter = workspace / "artifacts" / "candidate5"
    adapter.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        source = downloaded_adapter / name
        target = adapter / name
        if target.exists() and file_sha(target) == file_sha(source):
            continue
        temporary = target.with_name(f".{name}.tmp")
        try:
            shutil.copyfile(source, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    candidate_hashes = {name: file_sha(adapter / name) for name in
                        ("adapter_model.safetensors", "adapter_config.json")}
    candidate_manifest = manifest(candidate["model"]["run_id"], adapter, candidate_hashes, runtime, candidate,
        backend_kind="adapter", model_spec=candidate["model"], synthetic=False,
        training_run_id=candidate["model"]["run_id"],
        base_artifact_uri=str(repository), base_file_hashes=base_hashes)
    base_id = f"{base['model']['name']}@{base['model']['revision']}"
    baseline_manifest = manifest(base_id, repository, base_hashes, runtime, base,
        backend_kind="base", model_spec=base["model"], synthetic=False,
        acceptance_basis="initial_pinned_baseline_when_candidate_rejected",
        source_evaluation_run_id=base["evaluation_run_id"], evidence_files=evidence)
    verify_artifact(candidate_manifest, workspace)
    verify_artifact(baseline_manifest, workspace)
    release_report = {key: candidate_manifest[key] for key in
                      ("model_id", "model_sha256", "protocol_sha256", "golden_sha256", "runtime_sha256")}
    release_report.update(status="complete", passed=gate["passed"], reason=gate.get("reason"),
        evaluation_run_id=candidate["evaluation_run_id"], synthetic=False,
        metrics={key: candidate["overall"][key] for key in ("accuracy", "format_rate")},
        quality_gate=gate, evidence_files=evidence,
        training_run_id=candidate["model"]["run_id"])
    accepted = workspace / "accepted.json"
    if accepted.exists() and read(accepted) != baseline_manifest:
        raise ValueError("Workspace already has a different accepted model; refusing to bootstrap over it")
    if not accepted.exists():
        if gate["passed"]:
            raise ValueError("This baseline bootstrap route is only for the recorded rejected candidate")
        save(accepted, baseline_manifest)
        save(workspace / "baseline-bootstrap.json", dict(basis=baseline_manifest["acceptance_basis"],
            candidate_passed=False, accepted_model_id=base_id, evaluation_run_id=base["evaluation_run_id"],
            accepted_sha256=file_sha(accepted), note="Initial baseline, not a fabricated candidate PASS or champion"))
    save(workspace / "manifest-baseline.json", baseline_manifest)
    save(workspace / "manifest-candidate.json", candidate_manifest)
    save(workspace / "release-report.json", release_report)
    preparation_code = {name: file_sha(REPOSITORY / name) for name in ("lab.py", "release.py")}
    config = dict(tokenizer_revision=runtime["base_revision"], template_sha256=runtime["template_sha256"],
                  max_length=512, code_revision=sha(canonical(preparation_code)), format="jsonl",
                  tokenizer_sha256=runtime["tokenizer_sha256"])
    save(workspace / "config.json", config)
    # A real schema/UTF-8 preparation input, kept separate from the frozen golden.
    (workspace / "input.txt").write_bytes((REPOSITORY / "data/eval/training-index.jsonl").read_bytes())
    return dict(workspace=str(workspace), accepted_model_id=base_id,
        candidate_model_id=candidate_manifest["model_id"], candidate_passed=gate["passed"],
        release_report_sha256=file_sha(workspace / "release-report.json"), accepted_sha256=file_sha(accepted))


def verify_release_evidence(model_manifest, report, workspace):
    """Recompute a real decision; a bool or val_loss alone can never admit it."""
    # Prefect has an isolated lightweight environment. Verify existing HW67
    # evidence in its original pinned interpreter; send descriptors, not weights.
    original_python = REPOSITORY / ".venv/bin/python"
    if Path(sys.prefix).resolve() != (REPOSITORY / ".venv").resolve():
        payload = canonical(dict(manifest=model_manifest, report=report))
        result = subprocess.run([str(original_python), "-B", "-m", "hw8910.runtime", "verify",
                                 "--workspace", str(Path(workspace).resolve())],
                                input=payload, capture_output=True, cwd=REPOSITORY, timeout=180)
        if result.returncode != 0:
            raise ValueError("Pinned HW67 evidence verification failed in its model environment")
        return
    if report.get("synthetic") is True or model_manifest.get("synthetic") is not False:
        raise ValueError("Synthetic evidence is not a real model evaluation")
    bindings = ("model_id", "model_sha256", "protocol_sha256", "golden_sha256", "runtime_sha256")
    if (any(report.get(key) != model_manifest.get(key) for key in bindings)
            or sha(canonical(model_manifest.get("runtime"))) != model_manifest.get("runtime_sha256")):
        raise ValueError("Release wrapper and manifest/runtime bindings differ")
    reports, gate, evidence = frozen_evaluation()
    candidate = reports["candidate"]
    if (report.get("evidence_files") != evidence or report.get("quality_gate") != gate
            or report.get("passed") is not gate["passed"] or report.get("status") != "complete"
            or report.get("evaluation_run_id") != candidate["evaluation_run_id"]
            or model_manifest.get("model_spec") != candidate["model"]
            or model_manifest.get("training_run_id") != candidate["model"]["run_id"]
            or model_manifest.get("model_id") != candidate["model"]["run_id"]
            or report.get("training_run_id") != candidate["model"]["run_id"]
            or report.get("metrics") != {key: candidate["overall"][key] for key in ("accuracy", "format_rate")}):
        raise ValueError("Release report is not bound to the complete raw-verified HW67 decision")
    from evaluation.cli import manifest_for
    actual_manifest, adapter = manifest_for(candidate["model"])
    expected_adapter = {name: file_sha(adapter / name) for name in
                        ("adapter_model.safetensors", "adapter_config.json")}
    runtime, repository, base_hashes = evaluation_runtime(reports["baseline"]["model"], actual_manifest)
    if (model_manifest.get("runtime") != runtime
            or model_manifest.get("protocol_sha256") != candidate["protocol_sha256"]
            or model_manifest.get("golden_sha256") != candidate["golden_sha256"]
            or model_manifest.get("file_hashes") != expected_adapter
            or model_manifest.get("model_sha256") != sha(canonical(expected_adapter))
            or model_manifest.get("base_artifact_uri") != str(repository)
            or model_manifest.get("base_file_hashes") != base_hashes):
        raise ValueError("Serving runtime or base assets differ from the evaluated model")
    from release import verify_artifact
    verify_artifact(model_manifest, workspace)
    # The separate base is also checked before publication of an adapter.
    verify_artifact(dict(artifact_uri=str(repository), file_hashes=base_hashes,
                         model_sha256=sha(canonical(base_hashes))), workspace)


class ModelBackend:
    """One pinned Transformers baseline; generation only, never an optimizer."""

    def __init__(self, model_manifest, workspace):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.runtime import resolve_device, resolve_dtype, set_seed
        from release import verify_artifact
        self.manifest = copy.deepcopy(model_manifest)
        if model_manifest.get("synthetic") is not False or model_manifest.get("backend_kind") != "base":
            raise ValueError("The real demonstration serves only the explicitly pinned baseline")
        baseline = read(REPOSITORY / "evaluation/baseline.json")
        config = read(REPOSITORY / "evaluation/params.json")
        reports, gate, evidence = frozen_evaluation()
        from evaluation.cli import manifest_for
        evaluated_package, _ = manifest_for(reports["candidate"]["model"])
        expected_runtime, repository, base_hashes = evaluation_runtime(baseline, evaluated_package)
        generation = model_manifest["runtime"]["generation"]
        expected_generation = dict(system=config["system"], do_sample=False, seed=config["seed"],
                                   max_new_tokens=config["max_new_tokens"], enable_thinking=config["enable_thinking"])
        actual_model_id = f"{baseline['name']}@{baseline['revision']}"
        if (model_manifest.get("model_spec") != baseline or generation != expected_generation
                or type(model_manifest.get("schema_version")) is not int or model_manifest["schema_version"] != 1
                or model_manifest.get("model_id") != actual_model_id
                or model_manifest.get("protocol_sha256") != reports["baseline"]["protocol_sha256"]
                or model_manifest.get("golden_sha256") != reports["baseline"]["golden_sha256"]
                or model_manifest.get("model_sha256") != sha(canonical(base_hashes))
                or model_manifest.get("acceptance_basis") != "initial_pinned_baseline_when_candidate_rejected"
                or model_manifest["runtime"]["base_revision"] != baseline["revision"]
                or model_manifest["runtime"] != expected_runtime
                or model_manifest.get("artifact_uri") != str(repository)
                or model_manifest.get("file_hashes") != base_hashes
                or model_manifest.get("evidence_files") != evidence
                or model_manifest.get("source_evaluation_run_id") != reports["baseline"]["evaluation_run_id"]
                or gate["passed"] is not False
                or sha(canonical(model_manifest["runtime"])) != model_manifest.get("runtime_sha256")):
            raise ValueError("Manifest does not identify the evaluated initial baseline/runtime")
        verify_artifact(model_manifest, workspace)
        snapshot = Path(model_manifest["artifact_uri"]) / "snapshots" / baseline["revision"]
        hashes = {name: file_sha(snapshot / name) for name in TOKENIZER_FILES}
        if sha(canonical(hashes)) != model_manifest["runtime"]["tokenizer_sha256"]:
            raise ValueError("Tokenizer files changed")
        self.runtime = model_manifest["runtime"]
        self.loaded_id = actual_model_id
        self.runtime_sha256 = model_manifest["runtime_sha256"]
        self.max_new_tokens = generation["max_new_tokens"]
        self.calls = 0
        self.warmup_calls = 0
        self.device = resolve_device(self.runtime["device"])
        self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        if sha(self.tokenizer.chat_template.encode()) != self.runtime["template_sha256"]:
            raise ValueError("Loaded chat template differs from the evaluated template")
        set_seed(generation["seed"])
        self.model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True,
            dtype=resolve_dtype(self.runtime["dtype"])).to(self.device).eval()

    def _generate(self, prompt, limit):
        import torch
        generation = self.runtime["generation"]
        if type(limit) is not int or limit != self.max_new_tokens:
            raise ValueError("Generation limit differs from the evaluated protocol")
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": generation["system"]}, {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=generation["enable_thinking"])
        batch = self.tokenizer(text, return_tensors="pt")
        if batch["input_ids"].shape[1] > 512:
            raise ValueError("Prompt exceeds the supported 512 input-token bound")
        batch = batch.to(self.device)
        with torch.inference_mode():
            result = self.model.generate(**batch, max_new_tokens=limit, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)
        answer = result[0, batch["input_ids"].shape[1]:]
        return dict(answer=self.tokenizer.decode(answer, skip_special_tokens=True), output_tokens=len(answer))

    def warmup(self):
        result = self._generate("A spaceship crew discovers an alien civilization on an unknown planet.", self.max_new_tokens)
        self.warmup_calls += 1
        return result

    def generate(self, prompt, limit):
        result = self._generate(prompt, limit)
        self.calls += 1
        return result

    def close(self):
        import gc
        import torch
        del self.model
        del self.tokenizer
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--workspace", type=Path, default=REPOSITORY / "data/hw8910")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_workspace(args.workspace), ensure_ascii=False))
    else:
        payload = json.loads(sys.stdin.buffer.read())
        verify_release_evidence(payload["manifest"], payload["report"], args.workspace)
        print(json.dumps(dict(verified=True, scope="complete_raw_checked_HW67_evidence")))


if __name__ == "__main__":
    main()
