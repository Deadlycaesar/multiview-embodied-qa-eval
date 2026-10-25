"""Offline regressions for parsing, request construction and persisted scoring."""

import json
import importlib.util
from dataclasses import asdict
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "evaluation", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)],
)
package = importlib.util.module_from_spec(spec)
sys.modules["evaluation"] = package
spec.loader.exec_module(package)

from evaluation.core.config import read_config
from evaluation.core.io import digest, read_json, write_json, write_jsonl
from evaluation.core.distribution import export_run, inference_protocol, validate_bundle, validate_run
from evaluation.core.schemas import Sample
from evaluation.core.scoring import score_run
from evaluation.metrics.choice import extract_choice, format_compliant
from evaluation.protocols.prompts import build_request


def all_view_row(sample):
    return {"qid": sample.id, "experiment": "B0", "condition": "b0_all_views",
            "k": len(sample.images), "seed": None, "cameras": list(range(len(sample.images)))}


class VariableOptionTests(unittest.TestCase):
    def sample(self, labels):
        choices = {label: f"position-{i}" for i, label in enumerate(labels)}
        return Sample("fixture", "Where is the object?", ["view-0.png", "view-1.png"], choices,
                      labels[-1], [choices[labels[-1]]])

    def test_variable_counts_reach_mcq_and_open_requests(self):
        for count in (1, 2, 3, 4, 5, 8, 28):
            labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:count]
            if count > 26:
                labels += ["AA", "AB"]
            with self.subTest(count=count):
                sample = self.sample(labels)
                self.assertEqual(list(sample.choices), labels)
                for model in ("qwen3vl", "qwen35"):
                    mcq_config = read_config("configs/experiments.mcq.json", ["b0_all_views"], model_name=model)
                    open_config = read_config("configs/experiments.open.json", ["open_all_views"], model_name=model)
                    mcq = build_request(sample, mcq_config["experiments"]["b0_all_views"], mcq_config,
                                        plan_row=all_view_row(sample))
                    expected = sample.question + "\n" + "\n".join(
                        f"{label}. {sample.choices[label]}" for label in labels
                    ) + "\n\n" + mcq_config["prompts"]["mcq"]
                    self.assertEqual(mcq.text, expected)
                    self.assertEqual(mcq.option_labels, labels)
                    opened = build_request(sample, open_config["experiments"]["open_all_views"], open_config)
                    self.assertEqual(opened.text, sample.question)
                    self.assertEqual(opened.option_labels, [])
                    self.assertNotIn("references", asdict(mcq))
                    self.assertNotIn("answer", asdict(mcq))

    def test_scoring_uses_actual_labels_and_preserves_format_distinction(self):
        choices = {"A": "left", "B": "right", "E": "above", "AA": "behind"}
        for text, label, compliant in (("E", "E", True), ("e", "E", False),
                                       ("E.", "E", False), ("Answer: E.", "E", False),
                                       ("AA", "AA", True), ("aa", "AA", False),
                                       ("behind", "AA", False), ("left", "A", False),
                                       ("D", None, False), ("Answer: D.", None, False)):
            with self.subTest(text=text):
                self.assertEqual(extract_choice(text, choices)["letter"], label)
                self.assertEqual(format_compliant(text, choices), compliant)
        self.assertEqual(extract_choice("E or AA", choices)["parse_status"], "ambiguous")
        self.assertEqual(extract_choice("Answer: E. Option: D.", choices)["parse_status"], "ambiguous")
        self.assertIsNone(extract_choice("D", {"A": "left", "B": "right", "C": "above"})["letter"])

    def test_saved_scores_and_summary_include_extra_options(self):
        config = read_config("configs/experiments.mcq.json", ["b0_all_views"])
        samples, requests, predictions = [], [], []
        for index, (labels, response) in enumerate(((list("AB"), "B"), (list("ABC"), "D"),
                                                   (list("ABCDE"), "E"), (["A", "AA"], "AA"))):
            sample = self.sample(labels)
            sample.id = f"fixture-{index}"
            samples.append(asdict(sample))
            request = asdict(build_request(sample, config["experiments"]["b0_all_views"], config,
                                           plan_row=all_view_row(sample)))
            requests.append(request)
            predictions.append({"sample_id": sample.id, "request": request, "status": "ok", "text": response,
                                "option_probs": {label: 1 / len(labels) for label in labels},
                                "option_probs_status": "ok", "option_probs_version": "first-token-label-softmax-v1",
                                "visual_tokens": 2})
        request_map = {"b0_all_views": requests}
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            protocol = inference_protocol(config)
            write_json(directory / "manifest.json", {
                "schema_version": 4, "fingerprint": "offline-fixture", "config": config,
                "protocol": protocol, "protocol_hash": digest(protocol), "environment": {},
                "samples_sha256": digest(samples), "requests_sha256": digest(request_map),
            })
            write_jsonl(directory / "samples.jsonl", samples)
            write_json(directory / "requests.json", request_map)
            for index, prediction in enumerate(predictions):
                write_json(directory / "b0_all_views" / f"{index:04d}.json", prediction)
            result = score_run(directory, json.loads((ROOT / "configs/metrics.json").read_text()))
            scored = Path(result["score_dir"])
            self.assertTrue(result["complete"])
            self.assertEqual([row["correct"] for row in read_json(scored / "b0_all_views.json")],
                             [True, False, True, True])
            summary = read_json(scored / "summary.json")["b0_all_views"]
            self.assertEqual(summary["option_accuracy"], 0.75)
            self.assertEqual(summary["format_compliance"], 0.75)
            self.assertEqual(summary["option_probs_coverage"], 1.0)
            self.assertEqual(summary["mean_visual_tokens"], 2)
            for index, prediction in enumerate(predictions):
                self.assertEqual(read_json(directory / "b0_all_views" / f"{index:04d}.json"), prediction)
            validation = validate_run(directory, require_complete=True)
            self.assertTrue(validation["complete"])
            bundle = directory / "bundle"
            exported = export_run(directory, bundle, require_complete=True)
            self.assertEqual(exported["predictions"], 4)
            self.assertTrue(validate_bundle(bundle, require_complete=True)["complete"])
            write_jsonl(bundle / "predictions.jsonl", [])
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
