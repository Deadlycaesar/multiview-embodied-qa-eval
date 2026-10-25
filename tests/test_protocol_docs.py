"""Guard the human-readable protocol against drift from executable configs."""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ProtocolDocumentTests(unittest.TestCase):
    def test_models_prompt_and_experiment_counts_match_configs(self):
        document = (ROOT / "configs/PROTOCOL.md").read_text(encoding="utf-8")
        config = json.loads((ROOT / "configs/experiments.mcq.json").read_text())
        for key, path in config["models"].items():
            model = json.loads((ROOT / path).read_text())
            with self.subTest(model=key):
                self.assertIn(f"`{key}`", document)
                self.assertIn(model["repo_id"], document)
                self.assertIn(model["revision"], document)
                self.assertIn(f"requirements/models/{key}.txt", document)
        self.assertIn(config["prompts"]["mcq"], document)
        self.assertIn(config["prompts"]["version"], document)
        for condition in config["experiments"]:
            self.assertIn(f"`{condition}`", document)
        for expected in ("3,814", "4,002", "19,956", "seed=0/1/2", "k=1/2/4"):
            self.assertIn(expected, document)
        self.assertIn(Path(config["view_plan"]).name, document)
        for model in config["model_groups"]["view_analysis"]:
            self.assertIn(f"`{model}`", document)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("正式提交以完整 run 目录为主", readme)
        self.assertIn("19,956", readme)
        judge = json.loads((ROOT / "configs/judge.json").read_text())
        self.assertEqual(judge["model"], "jev")
        self.assertIn("`model=jev`", document)


if __name__ == "__main__":
    unittest.main()
