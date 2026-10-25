"""The constant diagnostic must never derive its rule from held-out labels."""

import unittest

from scripts.naive_hw67_baseline import frequent_genres


def row(answer):
    return {"messages": [{"role": "assistant", "content": answer}]}


class ConstantTrainingPrior(unittest.TestCase):
    def test_unselected_source_labels_do_not_change_the_rule(self):
        source = {"a": row("драма, комедия"), "b": row("драма, комедия")}
        before = frequent_genres(source, ["a", "b"])
        source.update({f"unused{i}": row("боевик") for i in range(100)})
        self.assertEqual(frequent_genres(source, ["a", "b"]), before)
        self.assertEqual(before[0], ["драма", "комедия"])

    def test_duplicate_training_ids_cannot_reweight_frequencies(self):
        with self.assertRaises(ValueError):
            frequent_genres({"a": row("драма")}, ["a", "a"])


if __name__ == "__main__":
    unittest.main()
