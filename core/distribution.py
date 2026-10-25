"""Validate runs and export portable, content-hashed result bundles."""

import math
from collections import Counter

from .io import digest, file_hash, read_json, read_jsonl, write_json, write_jsonl
from .paths import project_path
from .reporting import write_csv


def inference_protocol(config):
    dataset = {key: value for key, value in config["dataset"].items() if key != "snapshot_dir"}
    return {
        "schema_version": config["schema_version"],
        "dataset": dataset,
        "view_plan_sha256": file_hash(project_path(config["view_plan"])) if config.get("view_plan") else None,
        "generation": config["generation"],
        "runtime": config["runtime"],
        "prompts": config["prompts"],
        "model_groups": config["model_groups"],
        "experiments": config.get("canonical_experiments", config["experiments"]),
    }


def _probabilities_valid(prediction, request, mode):
    if mode != "mcq":
        return prediction.get("option_probs_status") in (None, "not_applicable")
    probabilities = prediction.get("option_probs")
    labels = request.get("option_labels", [])
    return (prediction.get("option_probs_status") == "ok"
            and prediction.get("option_probs_version") == "first-token-label-softmax-v1"
            and isinstance(probabilities, dict) and list(probabilities) == labels
            and all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
                    for value in probabilities.values())
            and math.isclose(sum(probabilities.values()), 1.0, rel_tol=0, abs_tol=1e-4))


def validate_run(run_dir, require_complete=False):
    run_dir = project_path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    if manifest.get("schema_version") != 4:
        raise ValueError("Only v4 runs can be validated")
    samples = read_jsonl(run_dir / "samples.jsonl")
    requests = read_json(run_dir / "requests.json")
    if digest(samples) != manifest["samples_sha256"] or digest(requests) != manifest["requests_sha256"]:
        raise ValueError("Saved samples or requests fail their content hash")
    if manifest.get("protocol_hash") != digest(inference_protocol(manifest["config"])):
        raise ValueError("Inference protocol hash mismatch")
    counts = {}
    for name, experiment in manifest["config"]["experiments"].items():
        counter = Counter()
        for index, request in enumerate(requests[name]):
            path = run_dir / name / f"{index:04d}.json"
            prediction = read_json(path) if path.exists() else {"status": "missing"}
            status = prediction.get("status")
            if status not in ("ok", "error", "missing"):
                raise ValueError(f"Unknown prediction status: {name}/{index}")
            if status != "missing" and (prediction.get("sample_id") != request["sample_id"]
                                        or prediction.get("request") != request):
                raise ValueError(f"Prediction/request mismatch: {name}/{index}")
            if status == "ok":
                visual_tokens = prediction.get("visual_tokens")
                if type(visual_tokens) is not int or visual_tokens < 0:
                    raise ValueError(f"Invalid visual token count: {name}/{index}")
                if not _probabilities_valid(prediction, request, experiment["mode"]):
                    raise ValueError(f"Invalid option probabilities: {name}/{index}")
                if name == "a1_no_image" and (request["images"] or visual_tokens != 0):
                    raise ValueError("A1 must have zero images and zero visual tokens")
            counter[status] += 1
        counts[name] = {"requested": len(requests[name]), **dict(counter)}
    latest_path = run_dir / "scores" / "latest.json"
    latest = read_json(latest_path) if latest_path.exists() else None
    complete = all(row["requested"] == row.get("ok", 0) for row in counts.values())
    scored_complete = bool(latest and latest.get("complete"))
    if require_complete and not (complete and scored_complete):
        raise ValueError("Run or its latest scores are incomplete")
    return {"complete": complete, "scored_complete": scored_complete, "counts": counts,
            "protocol_hash": manifest["protocol_hash"], "score_id": latest.get("score_id") if latest else None}


def export_run(run_dir, output_dir, require_complete=False):
    run_dir, output_dir = project_path(run_dir), project_path(output_dir)
    if output_dir.exists():
        raise ValueError("Export output already exists")
    validation = validate_run(run_dir, require_complete)
    manifest = read_json(run_dir / "manifest.json")
    score_id = validation["score_id"]
    if score_id is None:
        raise ValueError("Score the run before exporting")
    score_dir = run_dir / "scores" / score_id
    requests = read_json(run_dir / "requests.json")
    prediction_rows, score_rows = [], []
    for name, condition_requests in requests.items():
        scored = read_json(score_dir / f"{name}.json")
        for index, (request, score) in enumerate(zip(condition_requests, scored)):
            path = run_dir / name / f"{index:04d}.json"
            prediction = read_json(path) if path.exists() else {"status": "missing", "text": ""}
            metadata = request["metadata"]
            row = {"sample_id": request["sample_id"], "model_key": manifest["config"]["model_key"],
                   "condition": name, **{key: metadata.get(key) for key in ("experiment", "k", "seed", "cameras")},
                   "status": prediction["status"], "raw": prediction.get("text", "")}
            for field in ("visual_tokens", "input_tokens", "output_tokens", "elapsed_seconds", "generation_seconds",
                          "peak_allocated_bytes", "peak_reserved_bytes", "reached_token_limit", "option_probs",
                          "option_probs_status", "option_probs_version"):
                row[field] = prediction.get(field)
            prediction_rows.append(row)
            score_rows.append(score)
    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    write_jsonl(output_dir / "scores.jsonl", score_rows)
    summary = read_json(score_dir / "summary.json")
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "summary.csv", [{"condition": name, **row} for name, row in summary.items()])
    write_json(output_dir / "model.json", read_json(run_dir / "model.json") if (run_dir / "model.json").exists() else {})
    write_json(output_dir / "environment.json", manifest["environment"])
    files = ["predictions.jsonl", "scores.jsonl", "summary.json", "summary.csv", "model.json", "environment.json"]
    export_manifest = {
        "schema_version": 4, "artifact_type": "mvqa-result-bundle",
        "complete": validation["complete"] and validation["scored_complete"],
        "protocol_hash": manifest["protocol_hash"], "run_fingerprint": manifest["fingerprint"],
        "score_id": score_id, "model_key": manifest["config"]["model_key"],
        "prediction_count": len(prediction_rows), "score_count": len(score_rows),
        "file_sha256": {name: file_hash(output_dir / name) for name in files},
    }
    write_json(output_dir / "manifest.json", export_manifest)
    return {"output_dir": str(output_dir), **validation, "predictions": len(prediction_rows)}


def validate_bundle(bundle_dir, require_complete=False):
    bundle_dir = project_path(bundle_dir)
    manifest = read_json(bundle_dir / "manifest.json")
    if manifest.get("artifact_type") != "mvqa-result-bundle" or manifest.get("schema_version") != 4:
        raise ValueError("Unknown result bundle")
    for name, expected in manifest["file_sha256"].items():
        if file_hash(bundle_dir / name) != expected:
            raise ValueError(f"Bundle file hash mismatch: {name}")
    predictions = read_jsonl(bundle_dir / "predictions.jsonl")
    scores = read_jsonl(bundle_dir / "scores.jsonl")
    if len(predictions) != manifest["prediction_count"] or len(scores) != manifest["score_count"]:
        raise ValueError("Bundle row counts do not match its manifest")
    keys = [(row["sample_id"], row["condition"], row["seed"], tuple(row["cameras"])) for row in predictions]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate prediction rows")
    if require_complete and not manifest["complete"]:
        raise ValueError("Bundle is not complete")
    return {"complete": manifest["complete"], "protocol_hash": manifest["protocol_hash"],
            "model_key": manifest["model_key"], "predictions": len(predictions)}
