"""Offline checks for the MVQA adapter against a synthetic local snapshot."""

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "evaluation", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)],
)
package = importlib.util.module_from_spec(spec)
sys.modules["evaluation"] = package
spec.loader.exec_module(package)

from evaluation.datasets import load_samples


def local_record(qid, images, options, answer, per_view):
    return {"qid": qid, "type": 3, "subtype": None, "dataset": "dexycb",
            "frame_id": f"frame-{qid}", "images": images, "question": "Which best describes the interaction?",
            "options": options, "answer": answer,
            "answer_text": options[ord(answer) - ord("A")], "per_view": per_view,
            "evidence_level": None, "combo": [], "misleading": [], "steps": 1,
            "pre_type": None, "rationale": "checked", "note": None}


class MVQAAdapterTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        for qid, colors in (("000001", ("red", "green")), ("000002", ("blue", "yellow"))):
            for index, color in enumerate(colors, 1):
                path = self.root / "mvqa-local-v1/images" / qid / f"cam{index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (4, 4), color).save(path)
        self.local_records = [
            local_record("000001", ["images/000001/cam1.png", "images/000001/cam2.png"],
                         ["No touch", "Touching", "Holding"], "B", ["Y", "N"]),
            local_record("000002", ["images/000002/cam1.png", "images/000002/cam2.png"],
                         ["Left", "Right"], "A", ["Y", "Y"]),
        ]
        self.open_records = [
            {**local_record("000001", ["../mvqa-local-v1/images/000001/cam1.png",
                                       "../mvqa-local-v1/images/000001/cam2.png"],
                            ["No touch", "Touching", "Holding"], "B", ["Y", "N"]),
             "open_reference": "Touching", "distractors": ["No touch", "Holding"]},
        ]
        self.write_qa("mvqa-local-v1", self.local_records)
        self.write_qa("mvqa-open-v1", self.open_records)

    def tearDown(self):
        self.temp.cleanup()

    def write_qa(self, subset, records):
        qa = self.root / subset / "questions.jsonl"
        qa.parent.mkdir(parents=True, exist_ok=True)
        qa.write_text("".join(json.dumps(record) + "\n" for record in records))
        setattr(self, f"{subset}_sha256", hashlib.sha256(qa.read_bytes()).hexdigest())

    def config(self, subset="mvqa-local-v1", count=2, **overrides):
        return {"adapter": "mvqa", "repo_id": "unused/local", "revision": "0" * 40,
                "subset": subset, "questions_sha256": getattr(self, f"{subset}_sha256", None),
                "count": count, "offset": 0, "snapshot_dir": str(self.root), **overrides}

    def test_loads_mcq_records_with_views_and_digests(self):
        samples = load_samples(self.config(), self.root / "data")
        self.assertEqual([s.id for s in samples], ["000001", "000002"])
        first = samples[0]
        self.assertEqual(list(first.choices), ["A", "B", "C"])
        self.assertEqual(first.answer, "B")
        self.assertEqual(first.references, ["Touching"])
        self.assertEqual(len(first.images), 2)
        self.assertEqual(len(first.metadata["image_sha256"]), 2)
        self.assertEqual(first.metadata["necessary_view_indices"], [0])
        self.assertEqual(samples[1].metadata["necessary_view_indices"], [0, 1])
        self.assertEqual(first.metadata["dataset"], "dexycb")
        self.assertEqual(first.metadata["reference_source"], "answer_text")

    def test_loads_open_records_without_choices(self):
        samples = load_samples(self.config(subset="mvqa-open-v1", count=1), self.root / "data")
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample.choices, {})
        self.assertEqual(sample.answer, "B")
        self.assertEqual(sample.references, ["Touching"])
        self.assertEqual(sample.metadata["distractors"], ["No touch", "Holding"])
        self.assertEqual(sample.metadata["reference_source"], "open_reference")
        for path in sample.images:
            self.assertTrue(Path(path).is_file())

    def test_questions_sha256_mismatch_fails(self):
        with self.assertRaises(ValueError):
            load_samples(self.config(questions_sha256="1" * 64), self.root / "data")

    def test_unknown_subset_fails(self):
        with self.assertRaises(ValueError):
            load_samples(self.config(subset="mvqa-other", questions_sha256=None), self.root / "data")

    def test_rejects_unsafe_image_paths(self):
        bad = dict(self.local_records[0], images=["/abs/path.png"])
        self.write_qa("mvqa-local-v1", [bad, self.local_records[1]])
        with self.assertRaises(ValueError):
            load_samples(self.config(), self.root / "data")
        escaping = dict(self.local_records[0], images=["../../outside.png", "images/000001/cam2.png"])
        self.write_qa("mvqa-local-v1", [escaping, self.local_records[1]])
        with self.assertRaises(ValueError):
            load_samples(self.config(), self.root / "data")

    def test_rejects_missing_image(self):
        missing = dict(self.local_records[0], images=["images/000001/cam9.png", "images/000001/cam2.png"])
        self.write_qa("mvqa-local-v1", [missing, self.local_records[1]])
        with self.assertRaises(ValueError):
            load_samples(self.config(), self.root / "data")

    def test_rejects_answer_text_mismatch_and_bad_per_view(self):
        mismatch = dict(self.local_records[0], answer_text="Holding")
        self.write_qa("mvqa-local-v1", [mismatch, self.local_records[1]])
        with self.assertRaises(ValueError):
            load_samples(self.config(), self.root / "data")
        bad_view = dict(self.local_records[0], per_view=["Y"])
        self.write_qa("mvqa-local-v1", [bad_view, self.local_records[1]])
        with self.assertRaises(ValueError):
            load_samples(self.config(), self.root / "data")
        bad_value = dict(self.local_records[0], per_view=["Y", "X"])
        self.write_qa("mvqa-local-v1", [bad_value, self.local_records[1]])
        with self.assertRaises(ValueError):
            load_samples(self.config(), self.root / "data")

    def test_open_record_requires_reference(self):
        bad = {**self.open_records[0], "open_reference": ""}
        self.write_qa("mvqa-open-v1", [bad])
        with self.assertRaises(ValueError):
            load_samples(self.config(subset="mvqa-open-v1", count=1), self.root / "data")


if __name__ == "__main__":
    unittest.main()
