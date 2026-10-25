"""Pairing guards for the before/after experiment report."""

import copy
import unittest

from scripts.compare_hw67_runs import METRICS, paired_changes


def report(correct):
    return {
        "per_item": [{"id": f"q{i}", "correct": value} for i, value in enumerate(correct)],
        "overall": {key: 0.5 for key in METRICS},
        "protocol_sha256": "same-protocol", "golden_sha256": "same-golden",
        "baseline_spec_sha256": "same-base",
    }


class PairedExperimentComparison(unittest.TestCase):
    def test_equal_aggregate_scores_can_hide_changed_answers(self):
        before, after = report([True, False]), report([False, True])
        result = paired_changes(before, after)
        self.assertEqual(result["improved"], ["q1"])
        self.assertEqual(result["regressed"], ["q0"])
        self.assertEqual(result["metric_deltas"]["accuracy"], 0)

    def test_unpaired_or_duplicate_questions_are_rejected(self):
        before = report([True, False])
        for ids in (["q0", "different"], ["q0", "q0"]):
            after = report([True, False])
            for item, identity in zip(after["per_item"], ids):
                item["id"] = identity
            with self.assertRaises(ValueError):
                paired_changes(before, after)

    def test_different_protocols_are_not_subtracted(self):
        before = report([True, False])
        for key in ("protocol_sha256", "golden_sha256", "baseline_spec_sha256"):
            after = copy.deepcopy(before)
            after[key] = "other-task-or-model"
            with self.assertRaises(ValueError):
                paired_changes(before, after)


if __name__ == "__main__":
    unittest.main()
