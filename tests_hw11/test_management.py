"""Recover a stopped kind node without silently destroying its workloads."""
import json
import subprocess
import unittest
from unittest.mock import patch

from scripts import hw11


class ClusterStartupTests(unittest.TestCase):
    def setUp(self):
        self.commands = []
        self.clusters = [hw11.CLUSTER]
        self.running = True
        self.start_error = False
        self.run_patch = patch.object(hw11, "run", side_effect=self.run_command)
        self.kube_patch = patch.object(hw11, "kube")
        self.run_patch.start()
        self.kube = self.kube_patch.start()
        self.kube.return_value = subprocess.CompletedProcess([], 0, "ok", "")
        self.addCleanup(self.run_patch.stop)
        self.addCleanup(self.kube_patch.stop)

    def run_command(self, args, **kwargs):
        self.commands.append(list(map(str, args)))
        if args[1:3] == ["get", "clusters"]:
            output = "\n".join(self.clusters)
        elif args[1:3] == ["get", "nodes"]:
            output = "mlops-hw11-control-plane\n"
        elif "inspect" in args:
            output = json.dumps({"Running": self.running})
        elif "start" in args and self.start_error:
            return subprocess.CompletedProcess(args, 1, "", "RWLayer unexpectedly nil")
        else:
            output = ""
        return subprocess.CompletedProcess(args, 0, output, "")

    def test_running_cluster_exports_kubeconfig_and_waits_before_reuse(self):
        hw11.ensure_cluster()
        self.assertTrue(any(command[1:3] == ["export", "kubeconfig"] for command in self.commands))
        self.assertFalse(any("start" in command or "create" in command for command in self.commands))
        self.kube.assert_any_call("get", "--raw", "/readyz", "--request-timeout=5s",
                                  capture=True, check=False, timeout=10)
        self.kube.assert_any_call(
            "wait", "--for=condition=Ready", "nodes", "--all", "--timeout=120s", timeout=135)

    def test_stopped_node_is_resumed_before_export_and_wait(self):
        self.running = False
        hw11.ensure_cluster()
        start_index = next(i for i, command in enumerate(self.commands) if "start" in command)
        export_index = next(i for i, command in enumerate(self.commands) if "export" in command)
        self.assertLess(start_index, export_index)
        self.assertEqual(self.kube.call_count, 2)

    def test_corrupted_node_fails_with_recovery_hint_without_deleting(self):
        self.running = False
        self.start_error = True
        with self.assertRaisesRegex(RuntimeError, "RWLayer.*make up-hw11"):
            hw11.ensure_cluster()
        self.assertFalse(any("delete" in command for command in self.commands))
        self.kube.assert_not_called()

    def test_missing_cluster_is_created_and_waited_for(self):
        self.clusters = []
        hw11.ensure_cluster()
        self.assertTrue(any(command[1:3] == ["create", "cluster"] for command in self.commands))
        self.assertEqual(self.kube.call_count, 2)

    def test_api_timeout_is_bounded_and_does_not_delete_cluster(self):
        self.kube.return_value = subprocess.CompletedProcess([], 1, "", "connection refused")
        with patch.object(hw11.time, "monotonic", side_effect=[0, 121]):
            with self.assertRaisesRegex(RuntimeError, "API.*120s"):
                hw11.ensure_cluster()
        self.assertFalse(any("delete" in command for command in self.commands))

    def test_unready_node_aborts_startup_before_build(self):
        self.kube.side_effect = [subprocess.CompletedProcess([], 0, "ok", ""),
                                 RuntimeError("node not ready")]
        with patch.object(hw11.platform, "system", return_value="Linux"):
            with self.assertRaisesRegex(RuntimeError, "node not ready"):
                hw11.up()
        self.assertFalse(any("build" in command or "load" in command for command in self.commands))

    def test_api_process_timeout_is_retried_before_node_wait(self):
        self.kube.side_effect = [subprocess.TimeoutExpired("kubectl", 10),
                                 subprocess.CompletedProcess([], 0, "ok", ""),
                                 subprocess.CompletedProcess([], 0, "", "")]
        with patch.object(hw11.time, "sleep") as sleep:
            hw11.ensure_cluster()
        sleep.assert_called_once_with(2)
        self.assertEqual(self.kube.call_count, 3)


if __name__ == "__main__":
    unittest.main()
