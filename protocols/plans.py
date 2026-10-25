"""Deterministic, model-independent sample sets and oracle/random view plans."""

import hashlib
import random
from pathlib import Path

from ..core.io import read_jsonl, write_jsonl


MCQ_CONDITIONS = (
    "a1_no_image", "a2_remove_evidence", "a2_remove_random",
    "b0_all_views", "b1_random_views", "b2_evidence_views",
    "g0_all_views", "g1_random_views",
)
SEEDS = (0, 1, 2)
BUDGETS = (1, 2, 4)
EXPECTED_COUNTS = {"S0": 280, "S1": 190, "S2": 166, "H2O": 40, "OPEN": 188}


def select_sample_set(samples, name):
    if name == "S0":
        selected = list(samples)
    elif name == "S1":
        selected = [s for s in samples if s.metadata.get("dataset") == "dexycb" and s.metadata.get("type") != 4]
    elif name == "S2":
        selected = [s for s in select_sample_set(samples, "S1")
                    if len(s.metadata.get("necessary_view_indices") or ()) < len(s.images)]
    elif name == "H2O":
        selected = [s for s in samples if s.metadata.get("dataset") == "h2o"]
    elif name == "OPEN":
        selected = list(samples)
    else:
        raise ValueError(f"Unknown sample set: {name}")
    official_size = 188 if name == "OPEN" else 280
    if len(samples) == official_size and len(selected) != EXPECTED_COUNTS[name]:
        raise ValueError(f"{name}: expected {EXPECTED_COUNTS[name]} samples, found {len(selected)}")
    return selected


def experiment_for_condition(condition):
    if condition not in MCQ_CONDITIONS:
        raise ValueError(f"Unknown MCQ condition: {condition}")
    return condition.split("_", 1)[0].upper()


def _rng(seed, qid, condition, k):
    value = f"mvqa-view-v1\0{seed}\0{qid}\0{condition}\0{k}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(value).digest()[:16], "big"))


def _row(sample, condition, cameras, k, seed=None):
    return {"qid": sample.id, "experiment": experiment_for_condition(condition), "condition": condition,
            "k": k, "seed": seed, "cameras": cameras}


def _random_budget_rows(sample, condition):
    rows = []
    for k in BUDGETS:
        if len(sample.images) < k:
            raise ValueError(f"{sample.id}: fewer than {k} views")
        for seed in SEEDS:
            cameras = sorted(_rng(seed, sample.id, condition, k).sample(range(len(sample.images)), k))
            rows.append(_row(sample, condition, cameras, k, seed))
    return rows


def _evidence_budget_rows(sample):
    per_view = sample.metadata.get("per_view")
    if not per_view or len(per_view) != len(sample.images):
        raise ValueError(f"{sample.id}: evidence-priority views require complete Y/P/N annotations")
    rank = {"Y": 0, "P": 1, "N": 2}
    priority = sorted(range(len(sample.images)), key=lambda index: (rank[per_view[index]], index))
    return [_row(sample, "b2_evidence_views", sorted(priority[:k]), k) for k in BUDGETS]


def build_view_plan(samples):
    """Build the canonical MCQ plan for all supplied S0 samples."""
    rows = [_row(sample, "a1_no_image", [], 0) for sample in samples]
    s2_ids = {s.id for s in select_sample_set(samples, "S2")}
    for sample in samples:
        if sample.id not in s2_ids:
            continue
        required = set(sample.metadata.get("necessary_view_indices") or ())
        kept = [i for i in range(len(sample.images)) if i not in required]
        rows.append(_row(sample, "a2_remove_evidence", kept, len(kept)))
        for seed in SEEDS:
            removed = set(_rng(seed, sample.id, "a2_remove_random", len(required)).sample(
                range(len(sample.images)), len(required)))
            cameras = [i for i in range(len(sample.images)) if i not in removed]
            rows.append(_row(sample, "a2_remove_random", cameras, len(cameras), seed))
    for condition, sample_set in (("b", "S1"), ("g", "H2O")):
        selected_ids = {s.id for s in select_sample_set(samples, sample_set)}
        for sample in samples:
            if sample.id not in selected_ids:
                continue
            rows.append(_row(sample, f"{condition}0_all_views", list(range(len(sample.images))), len(sample.images)))
            rows.extend(_random_budget_rows(sample, f"{condition}1_random_views"))
            if condition == "b":
                rows.extend(_evidence_budget_rows(sample))
    return rows


def validate_view_plan(rows, samples):
    expected = build_view_plan(samples)
    if rows != expected:
        if len(rows) != len(expected):
            raise ValueError(f"View plan length mismatch: expected {len(expected)}, found {len(rows)}")
        for index, (actual, wanted) in enumerate(zip(rows, expected)):
            if actual != wanted:
                raise ValueError(f"View plan mismatch at row {index}: {actual} != {wanted}")
        raise ValueError("View plan does not match the canonical protocol")
    by_condition = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)
    return by_condition


def load_view_plan(path, samples):
    sample_ids = {sample.id for sample in samples}
    rows = [row for row in read_jsonl(Path(path)) if row["qid"] in sample_ids]
    return validate_view_plan(rows, samples)


def write_view_plan(path, samples):
    rows = build_view_plan(samples)
    write_jsonl(Path(path), rows)
    return rows
