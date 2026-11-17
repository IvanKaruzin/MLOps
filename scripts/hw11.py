"""Manage only the isolated HW11 stand; never prune shared Docker resources."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
import tarfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(os.environ.get("HW11_TOOLS", "/private/tmp/mlops-hw11-tools" if sys.platform == "darwin"
                            else "/tmp/mlops-hw11-tools"))
KUBECONFIG = Path(os.environ.get("HW11_KUBECONFIG", str(TOOLS / "kubeconfig")))
CONTEXT = os.environ.get("HW11_DOCKER_CONTEXT", "colima-mlops-hw11")
CLUSTER = "mlops-hw11"
NAMESPACE = "hw11"
IMAGE = "mlops-hw11:v1"


def binary(name):
    return str(TOOLS / name) if (TOOLS / name).exists() else shutil.which(name) or name


def run(args, *, capture=False, timeout=300, check=True):
    environment = os.environ.copy()
    environment.update(DOCKER_CONTEXT=CONTEXT, KUBECONFIG=str(KUBECONFIG))
    result = subprocess.run([str(arg) for arg in args], cwd=ROOT, env=environment,
                            text=True, capture_output=capture, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(map(str, args))}")
    return result


def kube(*args, **kwargs):
    return run([binary("kubectl"), "--kubeconfig", KUBECONFIG, "-n", NAMESPACE, *args], **kwargs)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download(url, expected_url, destination, archive_member=None):
    """Verify against the official published SHA before extracting an executable."""
    with urllib.request.urlopen(expected_url, timeout=60) as response:
        expected = response.read().decode().split()[0]
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"Checksum mismatch for {destination.name}")
    if archive_member:
        import io
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            member = archive.extractfile(archive_member)
            if member is None:
                raise ValueError("Helm archive has no expected executable")
            payload = member.read()
    destination.write_bytes(payload)
    destination.chmod(0o755)


def bootstrap():
    system = platform.system().lower()
    architecture = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "amd64"}.get(platform.machine())
    if system not in ("darwin", "linux") or architecture is None:
        raise ValueError("Bootstrap supports Linux/macOS arm64/amd64")
    TOOLS.mkdir(parents=True, exist_ok=True)
    base = f"https://dl.k8s.io/release/v1.34.1/bin/{system}/{architecture}/kubectl"
    download(base, base + ".sha256", TOOLS / "kubectl")
    base = f"https://kind.sigs.k8s.io/dl/v0.30.0/kind-{system}-{architecture}"
    download(base, base + ".sha256sum", TOOLS / "kind")
    base = f"https://get.helm.sh/helm-v3.19.0-{system}-{architecture}.tar.gz"
    download(base, base + ".sha256sum", TOOLS / "helm", f"{system}-{architecture}/helm")
    print(f"Verified tools installed in {TOOLS}")


def environment():
    disk = shutil.disk_usage(ROOT)
    value = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                 host_disk_bytes=dict(total=disk.total, used=disk.used, free=disk.free),
                 docker_context=CONTEXT, kubeconfig="temporary, excluded from repository")
    commands = {
        "docker": ["docker", "--context", CONTEXT, "info", "--format",
                   '{"version":"{{.ServerVersion}}","cpus":{{.NCPU}},"memory_bytes":{{.MemTotal}}}'],
        "docker_disk": ["docker", "--context", CONTEXT, "system", "df"],
        "kubectl": [binary("kubectl"), "version", "--client", "-o", "json"],
        "kind": [binary("kind"), "version"],
        "helm": [binary("helm"), "version", "--short"],
    }
    for key, args in commands.items():
        try:
            result = run(args, capture=True, timeout=30, check=False)
            value[key] = dict(returncode=result.returncode, stdout=result.stdout.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired):
            value[key] = {"unavailable": True}
    return value


def preflight(output):
    value = environment()
    save(output, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def ensure_cluster():
    clusters = run([binary("kind"), "get", "clusters"], capture=True).stdout.splitlines()
    if CLUSTER not in clusters:
        run([binary("kind"), "create", "cluster", "--name", CLUSTER, "--image", "kindest/node:v1.34.0",
             "--kubeconfig", KUBECONFIG, "--wait", "120s"], timeout=600)
    else:
        # kind lists stopped nodes too. Resume them without deleting workloads.
        nodes = run([binary("kind"), "get", "nodes", "--name", CLUSTER], capture=True).stdout.splitlines()
        if not nodes:
            raise RuntimeError(f"Cluster {CLUSTER} has no nodes")
        for node in nodes:
            state = json.loads(run(["docker", "--context", CONTEXT, "inspect", "--format",
                                    "{{json .State}}", node], capture=True).stdout)
            if not state["Running"]:
                started = run(["docker", "--context", CONTEXT, "start", node], capture=True,
                              check=False, timeout=120)
                if started.returncode:
                    raise RuntimeError(
                        f"Cannot restart HW11 node {node}: {started.stderr.strip()}. "
                        "Its data was not deleted. To recreate this disposable stand, run "
                        f"DOCKER_CONTEXT={CONTEXT} {binary('kind')} delete cluster --name {CLUSTER}, "
                        "then make up-hw11.")
        # Temporary kubeconfig may have been cleaned up while the VM survived.
        run([binary("kind"), "export", "kubeconfig", "--name", CLUSTER, "--kubeconfig", KUBECONFIG])
    deadline = time.monotonic() + 120
    while True:
        try:
            if kube("get", "--raw", "/readyz", "--request-timeout=5s", capture=True,
                    check=False, timeout=10).returncode == 0:
                break
        except subprocess.TimeoutExpired:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Kubernetes API for {CLUSTER} did not become ready within 120s")
        time.sleep(2)
    kube("wait", "--for=condition=Ready", "nodes", "--all", "--timeout=120s", timeout=135)


def up(build=True):
    TOOLS.mkdir(parents=True, exist_ok=True)
    if platform.system() == "Darwin":
        healthy = run(["docker", "--context", CONTEXT, "info"], capture=True,
                      timeout=20, check=False).returncode == 0
        if not healthy:
            previous = subprocess.run(["docker", "context", "show"], text=True,
                                      capture_output=True, check=True).stdout.strip()
            try:
                run(["colima", "start", "--profile", "mlops-hw11", "--cpu", "4", "--memory", "6",
                     "--disk", "12", "--runtime", "docker"], timeout=600)
            finally:
                if previous:
                    subprocess.run(["docker", "context", "use", previous], check=True)
    ensure_cluster()
    if build:
        run(["docker", "--context", CONTEXT, "build", "-f", "deploy/hw11/Dockerfile", "-t", IMAGE, "."],
            timeout=900)
        run([binary("kind"), "load", "docker-image", IMAGE, "--name", CLUSTER], timeout=300)
    run([binary("helm"), "upgrade", "--install", "hw11", "deploy/hw11/chart", "--namespace", NAMESPACE,
         "--create-namespace", "--kubeconfig", KUBECONFIG, "--wait", "--timeout", "300s"], timeout=330)
    if build:
        # The local tag is deliberately fixed; a rebuilt tag alone does not
        # change the Deployment template or replace existing client containers.
        kube("delete", "pod", "-l", "app=hw11-client", "--ignore-not-found=true")
        kube("rollout", "restart", "deployment/hw11-api")
    check_infrastructure()


def check_infrastructure():
    kube("rollout", "status", "deployment/hw11-api", "--timeout=120s", timeout=135)
    kube("get", "pods", "-o", "wide")
    kube("get", "endpointslices", "-l", "kubernetes.io/service-name=hw11-api", "-o", "json")
    deadline = time.monotonic() + 90
    while True:
        result = kube("get", "--raw", "/apis/metrics.k8s.io/v1beta1/nodes", capture=True,
                      check=False, timeout=15)
        if result.returncode == 0 and json.loads(result.stdout).get("items"):
            print("Resource metrics API: PASS")
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("Resource metrics API has no samples")
        time.sleep(5)
    kube("get", "deployment/hw11-api", "-o",
         "jsonpath={.spec.template.spec.containers[0].resources}")
    # Validate provisioned monitoring through its actual APIs as well as the
    # Kubernetes Ready state. A healthy Grafana process alone does not prove
    # that its dashboard was provisioned successfully.
    from hw11_experiments import forward, http

    with forward("service/hw11-prometheus", 9090) as address:
        status, targets = http(address + "/api/v1/targets")
        active = targets.get("data", {}).get("activeTargets", [])
        current = json.loads(kube("get", "pods", "-l", "app=hw11-api", "-o", "json", capture=True).stdout)
        ready_names = {pod["metadata"]["name"] for pod in current["items"]
                       if not pod["metadata"].get("deletionTimestamp") and any(
                           condition["type"] == "Ready" and condition["status"] == "True"
                           for condition in pod.get("status", {}).get("conditions", []))}
        api_targets = [target for target in active if target.get("labels", {}).get("job") == "hw11-api"
                       and target["labels"].get("pod") in ready_names]
        monitors = [target for target in active if target.get("labels", {}).get("job") != "hw11-api"]
        if (status != 200 or len(ready_names) < 2 or len(api_targets) != len(ready_names)
                or len(monitors) < 2 or any(target["health"] != "up" for target in api_targets + monitors)):
            raise RuntimeError("Prometheus discovery/scraping is incomplete")
    with forward("service/hw11-grafana", 3000) as address:
        status, dashboard = http(address + "/api/dashboards/uid/hw11")
        if status != 200 or len(dashboard.get("dashboard", {}).get("panels", [])) < 8:
            raise RuntimeError("Grafana dashboard is missing required panels")
    save(ROOT / "results/hw11/grafana-export.json", dashboard)
    save(ROOT / "results/hw11/monitoring-validation.json",
         dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), targets=targets,
              grafana_dashboard_uid=dashboard["dashboard"]["uid"],
              grafana_panels=len(dashboard["dashboard"]["panels"]), passed=True))
    print("\nInfrastructure: PASS")


def size(output):
    value = environment()
    for key, args in {
        "nodes": ("get", "nodes", "-o", "json"),
        "pods": ("get", "pods", "-o", "json"),
        "hpa": ("get", "hpa", "-o", "json"),
    }.items():
        result = kube(*args, capture=True, check=False)
        value[key] = json.loads(result.stdout) if result.returncode == 0 else {"unavailable": True}
    image_data = run(["docker", "--context", CONTEXT, "image", "ls", "--digests", "--format", "{{json .}}"],
                     capture=True).stdout
    value["images"] = [json.loads(row) for row in image_data.splitlines() if row.strip()]
    if platform.system() == "Darwin":
        result = run(["colima", "ssh", "--profile", "mlops-hw11", "--", "df", "-h", "/", "/var/lib/docker"],
                     capture=True, check=False)
        value["vm_disk"] = result.stdout.strip()
    save(output, value)
    print(f"Stand size saved: {output}")


def down(delete_vm=False):
    # Both names are hard-coded to the stand created by this homework.
    previous = subprocess.run(["docker", "context", "show"], text=True,
                              capture_output=True, check=True).stdout.strip()
    try:
        run([binary("kind"), "delete", "cluster", "--name", CLUSTER], timeout=180)
        if platform.system() == "Darwin":
            action = ["colima", "delete", "--profile", "mlops-hw11", "--force"] if delete_vm else [
                "colima", "stop", "--profile", "mlops-hw11"]
            run(action, timeout=180)
    finally:
        # Colima may select `default` even when another profile was selected.
        # Restore the caller's context if it still exists after deletion.
        if previous and subprocess.run(["docker", "context", "inspect", previous],
                                       capture_output=True).returncode == 0:
            subprocess.run(["docker", "context", "use", previous], check=True,
                           capture_output=True, text=True)
    print("Removed HW11 cluster; other profiles and their volumes were not modified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bootstrap", "preflight", "up", "check", "run", "report", "size", "down"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--delete-vm", action="store_true")
    parser.add_argument("--infrastructure", action="store_true", help="Explicit infrastructure check")
    args = parser.parse_args()
    if args.command == "bootstrap":
        bootstrap()
    elif args.command == "preflight":
        preflight(args.output or ROOT / "results/hw11/environment-before.json")
    elif args.command == "up":
        up(not args.no_build)
    elif args.command == "check":
        check_infrastructure()
    elif args.command == "size":
        size(args.output or ROOT / "results/hw11/environment-after.json")
    elif args.command == "down":
        down(args.delete_vm)
    elif args.command == "run":
        for experiment in ("a", "b", "c"):
            run([sys.executable, "-B", "scripts/hw11_experiments.py", experiment], timeout=7200)
    else:
        run([sys.executable, "-B", "scripts/hw11_report.py"], timeout=300)


if __name__ == "__main__":
    main()
