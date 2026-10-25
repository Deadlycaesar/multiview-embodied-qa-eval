"""Offline tests for probability and visual-token measurement primitives."""

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "evaluation", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)],
)
package = importlib.util.module_from_spec(spec)
sys.modules["evaluation"] = package
spec.loader.exec_module(package)

from evaluation.models.common import NativeVLM, option_probabilities


class Tokenizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, text, add_special_tokens=False):
        return self.mapping[text]


class InferenceProtocolTests(unittest.TestCase):
    def test_option_probs_use_first_position_and_normalize_over_labels(self):
        import torch
        logits = torch.tensor([[0.0, 1.0, 2.0, 100.0]])
        result = option_probabilities(logits, ["A", "B", "C"], Tokenizer({"A": [0], "B": [1], "C": [2]}))
        expected = torch.tensor([0.0, 1.0, 2.0]).softmax(dim=0).tolist()
        self.assertEqual(list(result), ["A", "B", "C"])
        self.assertTrue(all(math.isclose(result[label], value) for label, value in zip("ABC", expected)))
        self.assertAlmostEqual(sum(result.values()), 1.0)

    def test_option_probs_reject_multitoken_labels(self):
        import torch
        with self.assertRaisesRegex(ValueError, "single token"):
            option_probabilities(torch.zeros(1, 4), ["A", "AA"], Tokenizer({"A": [0], "AA": [1, 2]}))

    def test_visual_tokens_count_expanded_placeholders_only(self):
        import torch
        model = object.__new__(NativeVLM)
        model.config = {}
        model.model = SimpleNamespace(config=SimpleNamespace(image_token_id=9))
        inputs = {"input_ids": torch.tensor([[1, 9, 9, 2, 9, 3]])}
        self.assertEqual(model.visual_token_count(inputs, 1), 3)
        self.assertEqual(model.visual_token_count({"input_ids": torch.tensor([[1, 2]])}, 0), 0)


if __name__ == "__main__":
    unittest.main()
