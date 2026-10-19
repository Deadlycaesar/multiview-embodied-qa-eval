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
from evaluation.core.schemas import Sample
from evaluation.core.scoring import score_run
from evaluation.metrics.choice import extract_choice, format_compliant
from evaluation.protocols.prompts import build_request


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
                    config = read_config("configs/experiments.mcq.json", ["multi_mcq", "multi_open"], model_name=model)
                    mcq = build_request(sample, config["experiments"]["multi_mcq"], config, ROOT)
                    expected = sample.question + "\n" + "\n".join(
                        f"{label}. {sample.choices[label]}" for label in labels
                    ) + "\n\n" + config["prompts"]["mcq"]
                    self.assertEqual(mcq.text, expected)
                    opened = build_request(sample, config["experiments"]["multi_open"], config, ROOT)
                    self.assertEqual(opened.text, sample.question)
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
        config = read_config("configs/experiments.mcq.json", ["multi_mcq"])
        rows, requests, predictions = [], [], []
        for index, (labels, response) in enumerate(((list("AB"), "B"), (list("ABC"), "D"),
                                                   (list("ABCDE"), "E"), (["A", "AA"], "AA"))):
            sample = self.sample(labels)
            sample.id = f"fixture-{index}"
            rows.append(asdict(sample))
            request = asdict(build_request(sample, config["experiments"]["multi_mcq"], config, ROOT))
            requests.append(request)
            predictions.append({"sample_id": sample.id, "request": request, "status": "ok", "text": response})
        request_map = {"multi_mcq": requests}
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            write_json(directory / "manifest.json", {
                "schema_version": 2, "fingerprint": "offline-fixture", "config": config,
                "samples_sha256": digest(rows), "requests_sha256": digest(request_map),
            })
            write_jsonl(directory / "samples.jsonl", rows)
            write_json(directory / "requests.json", request_map)
            for index, prediction in enumerate(predictions):
                write_json(directory / "multi_mcq" / f"{index:04d}.json", prediction)
            result = score_run(directory, json.loads((ROOT / "configs/metrics.json").read_text()))
            scored = Path(result["score_dir"])
            self.assertTrue(result["complete"])
            self.assertEqual([row["correct"] for row in read_json(scored / "multi_mcq.json")],
                             [True, False, True, True])
            summary = read_json(scored / "summary.json")["multi_mcq"]
            self.assertEqual(summary["option_accuracy"], 0.75)
            self.assertEqual(summary["format_compliance"], 0.75)
            for index, prediction in enumerate(predictions):
                self.assertEqual(read_json(directory / "multi_mcq" / f"{index:04d}.json"), prediction)


if __name__ == "__main__":
    unittest.main()
