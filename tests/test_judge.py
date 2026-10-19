"""Check Judge request options and strict binary responses without API calls."""

import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "evaluation", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)],
)
package = importlib.util.module_from_spec(spec)
sys.modules["evaluation"] = package
spec.loader.exec_module(package)

from evaluation.metrics.judge import judge_answer, validate_config


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/judge.json").read_text())
        self.sample = {"question": "Where?", "references": ["left"],
                       "choices": {"A": "left", "B": "right"}, "answer": "A"}

    def response(self, verdict):
        return io.BytesIO(json.dumps({"choices": [{"message": {
            "content": json.dumps(verdict)}}]}).encode())

    @patch.dict(os.environ, {"JUDGE_API_KEY": "test-only-secret"})
    def test_deepseek_options_and_credentials(self):
        with patch("evaluation.metrics.judge.urlopen", return_value=self.response({"correct": True})) as call:
            result = judge_answer(self.sample, "A", "mcq", self.config)
        request = call.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(request.get_header("Authorization"), "Bearer test-only-secret")
        self.assertNotIn("test-only-secret", json.dumps(result))
        self.assertIs(result["correct"], True)

    @patch.dict(os.environ, {"JUDGE_API_KEY": "test-only-secret"})
    def test_optional_fields_and_false_verdict(self):
        config = {k: v for k, v in self.config.items() if k not in ("thinking", "response_format")}
        with patch("evaluation.metrics.judge.urlopen", return_value=self.response({"correct": False})) as call:
            result = judge_answer(self.sample, "right", "open", config)
        payload = json.loads(call.call_args.args[0].data)
        self.assertNotIn("thinking", payload)
        self.assertNotIn("response_format", payload)
        self.assertIs(result["correct"], False)

    @patch.dict(os.environ, {"JUDGE_API_KEY": "test-only-secret"})
    def test_nonbinary_or_extra_fields_are_rejected(self):
        for verdict in ({"correct": "false"}, {"correct": 1}, {"correct": True, "reason": "extra"}):
            with self.subTest(verdict=verdict), patch("evaluation.metrics.judge.urlopen", return_value=self.response(verdict)):
                with self.assertRaises(ValueError):
                    judge_answer(self.sample, "A", "mcq", self.config)

    def test_invalid_service_options_are_rejected(self):
        for key, value in (("thinking", "disabled"), ("thinking", {"type": "auto"}),
                           ("response_format", {"type": "text"})):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_config({**self.config, key: value}, require_key=False)


if __name__ == "__main__":
    unittest.main()
