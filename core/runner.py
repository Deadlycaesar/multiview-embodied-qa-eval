"""CLI and inference persistence. Scoring code is excluded from inference identity."""

import argparse
import platform
from dataclasses import asdict, replace
from datetime import datetime, timezone

from ..datasets import load_samples
from ..models import load_model
from ..protocols import build_request
from ..protocols.plans import load_view_plan, select_sample_set
from .config import read_config
from .distribution import inference_protocol
from .io import digest, file_hash, lock, now, read_json, read_jsonl, sources, versions, write_json, write_jsonl
from .paths import project_path, storage_paths


def prepare(config):
    data_dir = storage_paths(config)["data_dir"]
    samples = load_samples(config["dataset"], data_dir)
    sample_map = {sample.id: sample for sample in samples}
    planned = load_view_plan(project_path(config["view_plan"]), samples) if config.get("view_plan") else {}
    requests = {}
    for name, experiment in config["experiments"].items():
        selected = select_sample_set(samples, experiment["sample_set"])
        if experiment["views"] == "plan":
            selected_ids = {sample.id for sample in selected}
            rows = planned.get(name, [])
            if {row["qid"] for row in rows} != selected_ids:
                raise ValueError(f"{name}: view-plan sample set does not match {experiment['sample_set']}")
            requests[name] = [build_request(sample_map[row["qid"]], experiment, config,
                                            data_dir / "assets", row) for row in rows]
        else:
            requests[name] = [build_request(sample, experiment, config, data_dir / "assets")
                              for sample in selected]
    return samples, requests


def preflight(config):
    _, requests = prepare(config)
    model = load_model(config["model"], storage_paths(config)["model_cache_dir"])
    labels = sorted({label for rows in requests.values() for request in rows for label in request.option_labels})
    processor = getattr(model, "processor", None)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer is None:
        raise ValueError("Model adapter does not expose a tokenizer for preflight")
    invalid = [label for label in labels if len(tokenizer.encode(label, add_special_tokens=False)) != 1]
    if invalid:
        raise ValueError(f"Option labels are not single tokens for this model: {invalid}")
    return {"model_key": config["model_key"], "model_class": type(model).__name__,
            "conditions": {name: len(rows) for name, rows in requests.items()},
            "option_labels": labels, "metadata": model.metadata()}


def run(config, run_dir, resume=False):
    if resume:
        if not (run_dir / "manifest.json").exists():
            raise ValueError("No run manifest to resume")
    elif run_dir.exists():
        raise ValueError("Output exists; use --resume or a new directory")
    samples, requests = prepare(config)
    sample_rows = [asdict(sample) for sample in samples]
    request_rows = {name: [asdict(request) for request in rows] for name, rows in requests.items()}
    image_hashes = {path: file_hash(project_path(path))
                    for path in {p for rows in requests.values() for r in rows for p in r.images}}
    source_hashes = sources(
        ("datasets", "models", "protocols"),
        ("__init__.py", "run.py", "core/__init__.py", "core/config.py", "core/io.py", "core/paths.py", "core/schemas.py", "core/runner.py"),
    )
    environment = {**versions(("torch", "torchvision", "transformers", "huggingface-hub", "Pillow", "tokenizers", "numpy", "safetensors")),
                   "python": platform.python_version(), "platform": platform.platform()}
    identity = digest([config, sample_rows, request_rows, image_hashes, source_hashes, environment])
    protocol = inference_protocol(config)
    manifest = {"schema_version": 4, "config": config, "fingerprint": identity,
                "protocol": protocol, "protocol_hash": digest(protocol),
                "source_sha256": source_hashes, "environment": environment, "image_sha256": image_hashes,
                "samples_sha256": digest(sample_rows), "requests_sha256": digest(request_rows), "created_at": now()}
    run_dir.mkdir(parents=True, exist_ok=resume)
    with lock(run_dir):
        if resume:
            if read_json(run_dir / "manifest.json")["fingerprint"] != identity:
                raise ValueError("Inference inputs, code or environment changed; use a new run directory")
            if read_jsonl(run_dir / "samples.jsonl") != sample_rows or read_json(run_dir / "requests.json") != request_rows:
                raise ValueError("Saved sample or request manifest was modified")
        else:
            write_json(run_dir / "manifest.json", manifest)
            write_jsonl(run_dir / "samples.jsonl", sample_rows)
            write_json(run_dir / "requests.json", request_rows)
        model = None
        counts = {}
        for name, rows in requests.items():
            directory = run_dir / name
            directory.mkdir(exist_ok=True)
            warmed = False
            counts[name] = {"requested": len(rows), "successful": 0, "errors": 0}
            for index, request in enumerate(rows):
                path = directory / f"{index:04d}.json"
                old = read_json(path) if path.exists() else None
                if old:
                    if old.get("request") != asdict(request) or old.get("sample_id") != request.sample_id:
                        raise ValueError("Checkpoint does not match request")
                    if old["status"] == "ok":
                        counts[name]["successful"] += 1
                        continue
                if model is None:
                    print(f"Loading {config['model']['repo_id']}...", flush=True)
                    model = load_model(config["model"], storage_paths(config)["model_cache_dir"])
                    metadata = model.metadata()
                    model_path = run_dir / "model.json"
                    if model_path.exists() and read_json(model_path) != metadata:
                        raise ValueError("Resolved model or hardware changed; use a new run")
                    write_json(model_path, metadata)
                if not warmed:
                    # Warm up with a separate generic request, never re-answer a benchmark question.
                    warmup = replace(rows[0], sample_id="__warmup__", system="", text="Describe the image briefly.", metadata={"warmup": True})
                    for _ in range(config["runtime"]["warmup_per_condition"]):
                        model.generate(warmup)
                    warmed = True
                record = {"sample_id": request.sample_id, "model_key": config["model_key"],
                          "condition": name, "request": asdict(request), "created_at": now(),
                          "attempt": old.get("attempt", 1) + 1 if old else 1}
                if old:
                    write_json(directory / "attempts" / f"{index:04d}-{old.get('attempt', 1)}.json", old)
                try:
                    result = model.generate(request)
                    record.update(status="ok", **asdict(result))
                except Exception as exc:
                    record.update(status="error", error_type=type(exc).__name__, text="")
                    # Tracebacks can retain CUDA tensors. Exit the exception scope before retrying.
                write_json(path, record)
                counts[name]["successful" if record["status"] == "ok" else "errors"] += 1
                print(f"{name} [{index + 1}/{len(rows)}] {record['status']}", flush=True)
        write_json(run_dir / "inference-summary.json", counts)
    return counts


def main():
    parser = argparse.ArgumentParser(description="Multiview embodied QA evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "preflight", "run"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", default="configs/experiments.mcq.json")
        sub.add_argument("--model", help="Model key from the shared config")
        sub.add_argument("--experiments", nargs="+")
        sub.add_argument("--limit", type=int)
        sub.add_argument("--offset", type=int, help="Smoke-test dataset offset; requires --limit")
        if command == "run":
            sub.add_argument("--run-dir")
            sub.add_argument("--resume", action="store_true")
    score = commands.add_parser("score")
    score.add_argument("--run-dir", required=True)
    score.add_argument("--metrics-config", default="configs/metrics.json")
    score.add_argument("--judge-config", help="Explicitly enable external Judge calls")
    score.add_argument("--bertscore", action="store_true", help="Explicitly load the pinned BERTScore model")
    validate = commands.add_parser("validate")
    validate.add_argument("--run-dir", required=True, help="Run directory or exported result bundle")
    validate.add_argument("--bundle", action="store_true")
    validate.add_argument("--complete", action="store_true", help="Require complete predictions and scores")
    export = commands.add_parser("export")
    export.add_argument("--run-dir", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--complete", action="store_true", help="Refuse to export an incomplete result")
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--run-dirs", nargs="+", required=True)
    analyze.add_argument("--score-ids", nargs="+", help="One score ID per run directory; defaults to latest for each run")
    analyze.add_argument("--plots", action="store_true", help="Also export accuracy-visual-token PNG/SVG figures")
    analyze.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "score":
        from .scoring import score_run
        result = score_run(project_path(args.run_dir), read_json(project_path(args.metrics_config)),
                           read_json(project_path(args.judge_config)) if args.judge_config else None, args.bertscore)
        print(f"Scores: {result['score_dir']}")
        return int(not result["complete"])
    if args.command in ("validate", "export"):
        from .distribution import export_run, validate_bundle, validate_run
        if args.command == "export":
            result = export_run(args.run_dir, args.output_dir, args.complete)
        elif args.bundle:
            result = validate_bundle(args.run_dir, args.complete)
        else:
            result = validate_run(args.run_dir, args.complete)
        print(result)
        return 0
    if args.command == "analyze":
        from ..analysis.comparisons import analyze_runs
        result = analyze_runs([project_path(path) for path in args.run_dirs], project_path(args.output_dir), args.score_ids, args.plots)
        print(f"Analysis: {result}")
        return 0
    config = read_config(args.config, args.experiments, args.limit, args.model, args.offset)
    if args.command == "preflight":
        result = preflight(config)
        metadata = result.pop("metadata")
        result.update(device=metadata["device"], dtype=metadata["dtype"], resolved_model_class=metadata["model_class"])
        print(result)
        return 0
    if args.command == "prepare":
        samples, requests = prepare(config)
        rows = {name: [asdict(request) for request in values] for name, values in requests.items()}
        output = storage_paths(config)["data_dir"] / "prepared" / (digest([config, rows]) + ".json")
        write_json(output, {"config": config, "samples": [asdict(sample) for sample in samples], "requests": rows})
        print(f"Prepared {len(samples)} samples × {len(requests)} conditions: {output}")
        return 0
    if args.resume and not args.run_dir:
        parser.error("--resume requires --run-dir")
    directory = project_path(args.run_dir) if args.run_dir else storage_paths(config)["runs_dir"] / (
        config["model_key"] + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    result = run(config, directory, args.resume)
    print(f"Predictions: {directory}. Use score separately.")
    return int(any(row["successful"] != row["requested"] for row in result.values()))
