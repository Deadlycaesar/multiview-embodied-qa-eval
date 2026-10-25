"""Versioned independent scoring of immutable raw prediction snapshots."""

import math

from ..metrics import extract_choice, format_compliant, token_f1, tokens
from .io import digest, lock, now, read_json, read_jsonl, sources, versions, write_json
from .paths import storage_paths
from .reporting import summarize, write_report


PREDICTION_FIELDS = (
    "elapsed_seconds", "generation_seconds", "input_tokens", "visual_tokens", "output_tokens",
    "reached_token_limit", "peak_allocated_bytes", "peak_reserved_bytes", "option_probs",
    "option_probs_status", "option_probs_version",
)


def validate_metrics(config):
    if not config.get("protocol_version"):
        raise ValueError("Metric protocol version is required")
    if config["choice"]["format"] != "valid_option_label" or type(config["choice"]["allow_exact_choice_text"]) is not bool:
        raise ValueError("Unsupported choice protocol")
    if config["token_f1"] != {"normalization": "casefold_unicode_words", "references": "max"}:
        raise ValueError("Unsupported Token F1 protocol")
    review = config["review"]
    for key in ("low_token_f1", "high_token_f1", "low_bertscore", "high_bertscore"):
        if not isinstance(review[key], (int, float)) or not 0 <= review[key] <= 1:
            raise ValueError("Review thresholds must be within [0, 1]")
    if review["low_token_f1"] >= review["high_token_f1"] or review["low_bertscore"] >= review["high_bertscore"]:
        raise ValueError("Review low thresholds must be smaller than high thresholds")


def score_run(run_dir, metrics_config, judge_config=None, use_bertscore=False):
    validate_metrics(metrics_config)
    if judge_config is not None:
        from ..metrics.judge import JUDGE_PROMPT, judge_answer, validate_config
        validate_config(judge_config)
    if use_bertscore:
        from ..metrics.bertscore import BertScore, validate_config
        validate_config(metrics_config["bertscore"])
    with lock(run_dir):
        manifest = read_json(run_dir / "manifest.json")
        if manifest.get("schema_version") != 4:
            raise ValueError("v4 scoring requires a v4 run")
        samples = read_jsonl(run_dir / "samples.jsonl")
        sample_map = {sample["id"]: sample for sample in samples}
        requests = read_json(run_dir / "requests.json")
        if digest(samples) != manifest["samples_sha256"] or digest(requests) != manifest["requests_sha256"]:
            raise ValueError("Saved sample/request manifest was modified")
        predictions, prediction_hashes = {}, {}
        for name, condition_requests in requests.items():
            predictions[name] = []
            for index, request in enumerate(condition_requests):
                path = run_dir / name / f"{index:04d}.json"
                prediction = read_json(path) if path.exists() else {"status": "missing", "text": ""}
                if path.exists() and (prediction.get("sample_id") != request["sample_id"]
                                      or prediction.get("request") != request):
                    raise ValueError("Prediction does not match saved request")
                if prediction["status"] not in ("ok", "error", "missing"):
                    raise ValueError("Unknown prediction status")
                predictions[name].append(prediction)
                prediction_hashes[f"{name}/{index:04d}"] = digest(prediction)
        code = sources(("metrics",), ("core/scoring.py", "core/reporting.py", "core/io.py", "core/config.py", "core/paths.py"))
        environment = versions(("bert-score", "transformers", "torch", "tokenizers", "huggingface-hub", "numpy"))
        protocol = {"metrics": metrics_config, "judge": judge_config, "bertscore_enabled": use_bertscore,
                    "source_sha256": code, "environment": environment}
        protocol_hash = digest(protocol)
        identity = digest([manifest["fingerprint"], prediction_hashes, protocol_hash])
        directory = run_dir / "scores" / identity
        directory.mkdir(parents=True, exist_ok=True)
        completion_path = directory / "completion.json"
        if completion_path.exists():
            completion_path.unlink()
        write_json(directory / "manifest.json", {
            "schema_version": 4, "run_dir": str(run_dir), "run_fingerprint": manifest["fingerprint"],
            "score_id": identity, "protocol_hash": protocol_hash, "protocol": protocol,
            "prediction_sha256": prediction_hashes, "created_at": now(),
        })
        cache_root = run_dir / "metric-cache" / protocol_hash
        bert = None
        summaries = {}
        score_hashes = {}
        for name, experiment in manifest["config"]["experiments"].items():
            mode = experiment["mode"]
            rows = []
            for index, (request, prediction) in enumerate(zip(requests[name], predictions[name])):
                sample = sample_map[request["sample_id"]]
                ok, text = prediction["status"] == "ok", prediction.get("text", "")
                row = {"sample_id": sample["id"], "condition": name, "model_key": manifest["config"]["model_key"],
                       "status": prediction["status"], "prediction": text, "reference": sample["references"],
                       "output_words": len(tokens(text)) if ok else 0,
                       "view_metadata": request["metadata"],
                       "judge_status": "not_configured" if judge_config is None else "prediction_unavailable",
                       "bertscore_status": "not_applicable" if mode == "mcq" else ("not_enabled" if not use_bertscore else "prediction_unavailable")}
                for field in PREDICTION_FIELDS:
                    row[field] = prediction.get(field)
                if mode == "mcq":
                    parsed = extract_choice(text, sample["choices"], metrics_config["choice"]["allow_exact_choice_text"]) if ok else {"letter": None, "parse_status": "prediction_unavailable"}
                    row.update(predicted_letter=parsed["letter"], parse_status=parsed["parse_status"],
                               correct=ok and parsed["letter"] == sample["answer"], format_compliant=ok and format_compliant(text, sample["choices"]))
                else:
                    row["token_f1"] = token_f1(text, sample["references"]) if ok else 0.0
                if judge_config is not None and ok:
                    key = digest([JUDGE_PROMPT, mode, sample["question"], sample["choices"] if mode == "mcq" else None,
                                  sample["references"], sample["answer"], text])
                    path = cache_root / "judge" / (key + ".json")
                    try:
                        verdict = read_json(path) if path.exists() else judge_answer(sample, text, mode, judge_config)
                        if not isinstance(verdict, dict) or type(verdict.get("correct")) is not bool:
                            raise ValueError("Invalid cached Judge response")
                        if not path.exists():
                            write_json(path, verdict)
                        row.update(judge_status="ok", judge_correct=verdict["correct"])
                    except Exception as exc:
                        row.update(judge_status="error", judge_error=type(exc).__name__)
                if use_bertscore and mode == "open" and ok:
                    key = digest([sample["references"], text])
                    path = cache_root / "bertscore" / (key + ".json")
                    if bert is None and not path.exists():
                        bert = BertScore(metrics_config["bertscore"], storage_paths(manifest["config"])["model_cache_dir"])
                        write_json(directory / "bertscore-model.json", bert.metadata)
                    try:
                        value = read_json(path)["f1"] if path.exists() else bert.score(text, sample["references"])
                        if not isinstance(value, (float, int)) or not math.isfinite(value):
                            raise ValueError("Invalid BERTScore")
                        if not path.exists():
                            write_json(path, {"f1": value})
                        row.update(bertscore_status="ok", bertscore_f1=value)
                    except Exception as exc:
                        row.update(bertscore_status="error", bertscore_error=type(exc).__name__)
                rows.append(row)
            write_json(directory / (name + ".json"), rows)
            score_hashes[name] = digest(rows)
            summaries[name] = summarize(rows, mode)
        write_report(directory, summaries)
        complete = all(row["complete"] and (judge_config is None or row["judge_scored"] == row["requested"])
                       and (not use_bertscore or row["mode"] == "mcq" or row["bertscore_scored"] == row["requested"])
                       for row in summaries.values())
        write_json(directory / "completion.json", {"complete": complete, "finished_at": now(), "score_sha256": score_hashes})
        write_json(run_dir / "scores" / "latest.json", {"score_id": identity, "complete": complete})
        return {"score_dir": str(directory), "complete": complete}
