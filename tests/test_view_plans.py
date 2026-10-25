"""Offline checks for canonical sample sets and deterministic camera plans."""

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

from evaluation.core.schemas import Sample
from evaluation.protocols.plans import build_view_plan, select_sample_set, validate_view_plan
from evaluation.protocols.prompts import build_request
from evaluation.core.config import read_config


def sample(qid, dataset="dexycb", kind=1, per_view=None):
    per_view = per_view or ["Y", "N", "P", "N", "N", "N", "N", "N"]
    return Sample(qid, "Question?", [f"{qid}-{i}.png" for i in range(len(per_view))],
                  {"A": "left", "B": "right"}, "A", ["left"],
                  {"dataset": dataset, "type": kind, "per_view": per_view,
                   "necessary_view_indices": [i for i, value in enumerate(per_view) if value == "Y"]})


class ViewPlanTests(unittest.TestCase):
    def setUp(self):
        self.samples = [sample("s1"), sample("s2", kind=4),
                        sample("s3", dataset="h2o", per_view=["Y", "N", "N", "N", "N"])]

    def test_sample_set_definitions(self):
        self.assertEqual([s.id for s in select_sample_set(self.samples, "S0")], ["s1", "s2", "s3"])
        self.assertEqual([s.id for s in select_sample_set(self.samples, "S1")], ["s1"])
        self.assertEqual([s.id for s in select_sample_set(self.samples, "S2")], ["s1"])
        self.assertEqual([s.id for s in select_sample_set(self.samples, "H2O")], ["s3"])

    def test_plan_is_deterministic_and_paired(self):
        rows = build_view_plan(self.samples)
        self.assertEqual(rows, build_view_plan(self.samples))
        validate_view_plan(rows, self.samples)
        conditions = ("a1_no_image", "a2_remove_evidence", "a2_remove_random", "b0_all_views",
                      "b1_random_views", "b2_evidence_views", "g0_all_views", "g1_random_views")
        counts = {name: sum(row["condition"] == name for row in rows) for name in conditions}
        self.assertEqual(counts, {"a1_no_image": 3, "a2_remove_evidence": 1, "a2_remove_random": 3,
                                  "b0_all_views": 1, "b1_random_views": 9, "b2_evidence_views": 3,
                                  "g0_all_views": 1, "g1_random_views": 9})
        evidence = next(row for row in rows if row["condition"] == "a2_remove_evidence")
        random_rows = [row for row in rows if row["condition"] == "a2_remove_random"]
        self.assertEqual(evidence["cameras"], [1, 2, 3, 4, 5, 6, 7])
        self.assertTrue(all(row["k"] == evidence["k"] for row in random_rows))
        self.assertTrue(all(row["cameras"] == sorted(row["cameras"]) for row in rows))
        with self.assertRaises(ValueError):
            validate_view_plan(rows[:-1], self.samples)

    def test_evidence_priority_is_y_then_p_then_n(self):
        rows = [row for row in build_view_plan(self.samples) if row["condition"] == "b2_evidence_views"]
        self.assertEqual([row["cameras"] for row in rows], [[0], [0, 2], [0, 1, 2, 3]])
        self.assertTrue(all(row["seed"] is None for row in rows))

    def test_no_image_and_random_budgets_reach_requests(self):
        config = read_config("configs/experiments.mcq.json")
        rows = build_view_plan(self.samples)
        by_condition = {}
        for row in rows:
            by_condition.setdefault(row["condition"], []).append(row)
        a1 = build_request(self.samples[0], config["experiments"]["a1_no_image"], config,
                           plan_row=by_condition["a1_no_image"][0])
        self.assertEqual(a1.images, [])
        self.assertEqual(a1.metadata["k"], 0)
        b1 = [build_request(self.samples[0], config["experiments"]["b1_random_views"], config, plan_row=row)
              for row in by_condition["b1_random_views"]]
        self.assertEqual({request.metadata["k"] for request in b1}, {1, 2, 4})
        self.assertEqual({request.metadata["seed"] for request in b1}, {0, 1, 2})
        for request in b1:
            self.assertEqual(request.images, [self.samples[0].images[i] for i in request.metadata["cameras"]])


if __name__ == "__main__":
    unittest.main()
