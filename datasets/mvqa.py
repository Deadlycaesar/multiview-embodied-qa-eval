"""ee6008 multiview embodied QA: MCQ (mvqa-local-v1) and open (mvqa-open-v1) tiers.

The dataset is a private HF dataset; authenticate with HF_TOKEN. Each record
references an ordered list of per-camera PNGs (DexYCB 8 views, H2O 5 views).
Open-tier records point into the local-tier image directory with relative
paths, so `..` segments are allowed but must stay inside the snapshot.

`revision` pins the dataset commit (checked by core/config.py) and
`questions_sha256` pins the QA file content. Set `snapshot_dir` to a local
copy of the dataset repository to read files instead of downloading them.
"""

import hashlib
import json
import posixpath
from pathlib import Path, PurePosixPath

from ..core.schemas import Sample
from ..core.paths import project_path

SUBSETS = ("mvqa-local-v1", "mvqa-open-v1")
PER_VIEW_VALUES = ("Y", "P", "N")


def load_samples(config: dict, data_dir: Path) -> list[Sample]:
    from PIL import Image

    count, offset = config["count"], config.get("offset", 0)
    if type(count) is not int or count < 1 or type(offset) is not int or offset < 0:
        raise ValueError("count must be positive and offset nonnegative")
    subset = config.get("subset")
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}")
    snapshot = config.get("snapshot_dir")
    if snapshot:
        snapshot_root = project_path(snapshot)

        def fetch(name: str) -> Path:
            path = snapshot_root / name
            if not path.is_file():
                raise ValueError(f"Missing file in local snapshot: {path}")
            return path
    else:
        from huggingface_hub import hf_hub_download

        def fetch(name: str) -> Path:
            return Path(hf_hub_download(
                config["repo_id"], name, repo_type="dataset",
                revision=config["revision"], cache_dir=str(data_dir / "hub"),
            ))

    qa_bytes = fetch(f"{subset}/questions.jsonl").read_bytes()
    expected = config.get("questions_sha256")
    if expected:
        actual = hashlib.sha256(qa_bytes).hexdigest()
        if actual != expected:
            raise ValueError(f"questions.jsonl changed: expected {expected}, found {actual}")
    lines = qa_bytes.decode("utf-8").splitlines()[offset:offset + count]
    if len(lines) != count:
        raise ValueError(f"Requested {count} records, found {len(lines)}")

    samples = []
    for index, line in enumerate(lines, offset):
        row = json.loads(line)
        qid = row["qid"]
        question = row["question"].strip()
        images = row["images"]
        if not qid or not question or not images:
            raise ValueError(f"Invalid sample: {qid}")
        per_view = row.get("per_view")
        if per_view is not None and (
            len(per_view) != len(images) or any(v not in PER_VIEW_VALUES for v in per_view)
        ):
            raise ValueError(f"Invalid per_view annotation: {qid}")

        paths, digests = [], []
        for name in images:
            relative = PurePosixPath(name)
            if relative.is_absolute():
                raise ValueError(f"Unsafe image path: {name}")
            resolved = posixpath.normpath(posixpath.join(subset, name))
            if resolved == ".." or resolved.startswith("../"):
                raise ValueError(f"Image path escapes the dataset snapshot: {name}")
            path = fetch(resolved)
            data = path.read_bytes()
            with Image.open(path) as image:
                image.verify()
            paths.append(str(path))
            digests.append(hashlib.sha256(data).hexdigest())

        if subset == "mvqa-local-v1":
            options = row["options"]
            if not isinstance(options, list) or not options or any(not str(o).strip() for o in options):
                raise ValueError(f"Invalid options: {qid}")
            choices = {chr(ord("A") + i): str(o).strip() for i, o in enumerate(options)}
            answer = str(row["answer"]).strip().upper()
            answer_text = str(row["answer_text"]).strip()
            if answer not in choices or answer_text != choices[answer]:
                raise ValueError(f"Invalid answer or answer_text: {qid}")
            references = [answer_text]
        else:
            reference = str(row.get("open_reference", "")).strip()
            if not reference:
                raise ValueError(f"Missing open_reference: {qid}")
            # Open-tier inference never sees the options; keep them out of Sample.
            choices = {}
            answer = str(row.get("answer", "")).strip().upper()
            references = [reference]

        required = [i for i, v in enumerate(per_view) if v == "Y"] if per_view else None
        metadata = {
            "row_index": index, "subset": subset, "type": row.get("type"),
            "subtype": row.get("subtype"), "dataset": row.get("dataset"),
            "frame_id": row.get("frame_id"), "per_view": per_view,
            "evidence_level": row.get("evidence_level"), "steps": row.get("steps"),
            "source_images": images, "image_sha256": digests,
            "necessary_view_indices": required,
            "annotation_evidence": "dataset per_view annotation: Y = answerable from this view" if per_view else None,
            "reference_source": "answer_text" if subset == "mvqa-local-v1" else "open_reference",
        }
        if subset == "mvqa-open-v1":
            metadata["distractors"] = row.get("distractors")
        samples.append(Sample(
            id=qid, question=question, images=paths, choices=choices,
            answer=answer, references=references, metadata=metadata,
        ))
    if len({s.id for s in samples}) != count:
        raise ValueError("Duplicate sample IDs")
    return samples
