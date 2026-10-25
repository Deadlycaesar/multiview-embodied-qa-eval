"""Pure aggregation tests for paired evidence removal and seed statistics."""

import importlib.util
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

from evaluation.analysis.comparisons import a2_summary, aggregate_seeds, budget_points


def scored(qid, correct, seed=None):
    return {"sample_id": qid, "correct": correct, "view_metadata": {"seed": seed}}


class AnalysisTests(unittest.TestCase):
    def test_a2_uses_evidence_not_random_as_baseline(self):
        evidence = [scored("q1", False), scored("q2", True)]
        random_rows = [scored("q1", True, 0), scored("q2", True, 0),
                       scored("q1", False, 1), scored("q2", False, 1)]
        result = a2_summary("m", evidence, random_rows)
        self.assertEqual(result["per_seed"][0]["evidence_accuracy"], 0.5)
        self.assertEqual(result["per_seed"][0]["random_accuracy"], 1.0)
        self.assertEqual(result["per_seed"][0]["evidence_minus_random"], -0.5)
        self.assertEqual(result["per_seed"][1]["evidence_minus_random"], 0.5)
        self.assertEqual(result["mean_delta"], 0.0)

    def test_seed_aggregation_preserves_per_seed_values(self):
        result = aggregate_seeds({0: [1], 1: [0]}, lambda rows: {"value": float(rows[0]), "visual_tokens": 10})
        self.assertEqual(result["value"], 0.5)
        self.assertEqual(result["std"], 0.7071067811865476)
        self.assertEqual([row["seed"] for row in result["seeds"]], [0, 1])

    def test_budget_points_support_deterministic_evidence_curve(self):
        rows = [{"correct": True, "visual_tokens": 100, "view_metadata": {"k": 1, "seed": None}},
                {"correct": False, "visual_tokens": 200, "view_metadata": {"k": 2, "seed": None}}]
        points = budget_points({"rows": rows})
        self.assertEqual([point["k"] for point in points], [1, 2])
        self.assertEqual(points[0]["accuracy"], 1.0)
        self.assertEqual(points[0]["std"], 0.0)
        self.assertEqual(points[1]["visual_tokens"], 200)


if __name__ == "__main__":
    unittest.main()
