"""Регрессии учёта tied-параметров и очистки hooks без загрузки весов."""

import unittest
from unittest.mock import patch
import threading

import torch

from src.inspect_model import forward_hooks, group_table, parameter_rows
from src.memory import PeakMemory


class InspectionTests(unittest.TestCase):
    def test_peak_retains_transient_allocation_and_stops_on_exception(self):
        sampled = threading.Event()
        allocation = [10]

        def read_memory(device):
            value = allocation[0]
            if value == 100:
                sampled.set()
            return value

        with patch("src.memory.synchronize"), patch("torch.mps.empty_cache"), \
                patch("src.memory.device_allocated_bytes", side_effect=read_memory):
            peak = PeakMemory(torch.device("mps"), interval=0.001)
            with self.assertRaisesRegex(RuntimeError, "step failed"):
                with peak:
                    allocation[0] = 100
                    self.assertTrue(sampled.wait(2), "sampler did not observe peak")
                    allocation[0] = 10
                    raise RuntimeError("step failed")
            self.assertEqual(peak.used, 100)
            self.assertFalse(peak.thread.is_alive())

    def test_tied_and_independent_equal_weights(self):
        model = torch.nn.Module()
        model.embed_tokens = torch.nn.Embedding(5, 3)
        model.lm_head = torch.nn.Linear(3, 5, bias=False)
        model.lm_head.weight = model.embed_tokens.weight
        model.q_proj = torch.nn.Linear(3, 5, bias=False)
        with torch.no_grad():
            model.q_proj.weight.copy_(model.embed_tokens.weight)
        table = {row["group"]: row for row in group_table(parameter_rows(model))}
        self.assertEqual(sum(row["params"] for row in table.values()), 30)
        self.assertEqual(table["lm_head"]["params"], 0)
        self.assertEqual(table["lm_head"]["tied_params"], 15)
        self.assertEqual(table["q_proj"]["params"], 15)

    def test_hooks_clean_up_after_exception_and_preserve_existing_hook(self):
        module = torch.nn.Identity()
        calls = []
        existing = module.register_forward_hook(lambda *args: calls.append(1))
        try:
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                with forward_hooks({"block": module}) as norms:
                    module(torch.ones(1, 2, 3))
                    self.assertEqual(len(norms["block"]), 2)
                    raise RuntimeError("forward failed")
            self.assertEqual(len(module._forward_hooks), 1)
            module(torch.ones(1, 2, 3))
            self.assertEqual(calls, [1, 1])
        finally:
            existing.remove()


if __name__ == "__main__":
    unittest.main()
