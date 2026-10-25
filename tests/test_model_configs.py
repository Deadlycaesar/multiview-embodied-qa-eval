"""Validate the complete model registry without importing model runtimes."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "evaluation", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)],
)
package = importlib.util.module_from_spec(spec)
sys.modules["evaluation"] = package
spec.loader.exec_module(package)

from evaluation.core.config import read_config
from evaluation.core.distribution import inference_protocol
from evaluation.core.io import digest
from evaluation.models import ADAPTERS


class ModelConfigTests(unittest.TestCase):
    def test_all_ten_models_are_registered_and_pinned(self):
        registry = json.loads((ROOT / "configs/experiments.mcq.json").read_text())["models"]
        self.assertEqual(len(registry), 10)
        self.assertEqual(set(registry), set(ADAPTERS))
        for key in registry:
            with self.subTest(model=key):
                config = read_config("configs/experiments.mcq.json", ["a1_no_image"], model_name=key)
                self.assertEqual(config["model_key"], key)
                self.assertEqual(config["model"]["adapter"], key)
                self.assertRegex(config["model"]["revision"], r"^[0-9a-f]{40}$")

    def test_model_tiers_and_shared_protocol_identity(self):
        baseline = read_config("configs/experiments.mcq.json", model_name="phi4")
        analysis = read_config("configs/experiments.mcq.json", model_name="qwen3vl")
        self.assertEqual(set(baseline["experiments"]), {"a1_no_image", "b0_all_views"})
        self.assertEqual(len(analysis["experiments"]), 8)
        with self.assertRaisesRegex(ValueError, "not assigned"):
            read_config("configs/experiments.mcq.json", ["b1_random_views"], model_name="phi4")
        self.assertEqual(digest(inference_protocol(baseline)), digest(inference_protocol(analysis)))

    def test_smoke_offset_requires_limit(self):
        config = read_config("configs/experiments.mcq.json", ["a2_remove_evidence"],
                             limit=1, model_name="qwen3vl", smoke_offset=2)
        self.assertEqual(config["dataset"]["offset"], 2)
        self.assertEqual(config["dataset"]["count"], 1)
        with self.assertRaisesRegex(ValueError, "requires --limit"):
            read_config("configs/experiments.mcq.json", model_name="qwen3vl", smoke_offset=2)


if __name__ == "__main__":
    unittest.main()
