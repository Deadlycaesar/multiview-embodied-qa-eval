"""Cross-run, paired analysis with input/metric compatibility checks."""

from collections import defaultdict
from itertools import combinations
from statistics import mean

from ..core.io import digest, read_json, read_jsonl, sources, write_json
from ..core.reporting import number, summarize, write_csv
from .disagreements import candidates
from .efficiency import summarize_pair


PAIRS = (
    ("white_mcq", "single_mcq", "visual_content"),
    ("single_mcq", "dual_mcq", "views"),
    ("dual_mcq", "multi_mcq", "views"),
    ("multi_mcq_no_instruction", "multi_mcq", "prompt"),
    ("multi_open", "multi_open_short", "prompt"),
    ("multi_mcq", "multi_open", "answer_form"),
)


def sample_identity(samples):
    # Local extraction paths can differ between machines; source paths and bytes must agree.
    return digest([{key: value for key, value in sample.items() if key != "images"} for sample in samples])


def load_run(run_dir, score_id):
    manifest = read_json(run_dir / "manifest.json")
    if manifest.get("schema_version") != 2:
        raise ValueError("Analysis requires v2 runs")
    score_id = score_id or read_json(run_dir / "scores" / "latest.json")["score_id"]
    if len(score_id) != 64 or any(char not in "0123456789abcdef" for char in score_id):
        raise ValueError("Invalid score ID")
    score_dir = run_dir / "scores" / score_id
    completion = read_json(score_dir / "completion.json")
    scoring = read_json(score_dir / "manifest.json")
    if scoring["run_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Scores refer to a different run")
    samples = read_jsonl(run_dir / "samples.jsonl")
    requests = read_json(run_dir / "requests.json")
    if digest(samples) != manifest["samples_sha256"] or digest(requests) != manifest["requests_sha256"]:
        raise ValueError("Saved inputs were modified")
    model_path = run_dir / "model.json"
    model_metadata = read_json(model_path) if model_path.exists() else {}
    nodes = []
    for name, experiment in manifest["config"]["experiments"].items():
        rows = read_json(score_dir / (name + ".json"))
        if digest(rows) != completion["score_sha256"][name]:
            raise ValueError("Score rows were modified after scoring")
        if [row["sample_id"] for row in rows] != [sample["id"] for sample in samples]:
            raise ValueError("Score sample order differs from manifest")
        for index, row in enumerate(rows):
            path = run_dir / name / f"{index:04d}.json"
            prediction = read_json(path) if path.exists() else {"status": "missing", "text": ""}
            if digest(prediction) != scoring["prediction_sha256"][f"{name}/{index:04d}"]:
                raise ValueError("Predictions changed after scoring; score again before analysis")
            if row["prediction"] != prediction.get("text", "") or row["status"] != prediction["status"]:
                raise ValueError("Scores do not match original predictions")
        nodes.append({
            "run_dir": str(run_dir), "score_id": score_id, "manifest": manifest,
            "score_manifest": scoring, "config": manifest["config"], "model_metadata": model_metadata,
            "model_key": manifest["config"]["model_key"], "condition": name, "experiment": experiment,
            "rows": rows, "requests": requests[name], "samples": samples,
            "sample_map": {sample["id"]: sample for sample in samples},
        })
    return nodes


def image_identity(node, request):
    return [node["manifest"]["image_sha256"][path] for path in request["images"]]


def validate_pair(left, right, kind):
    if left["score_manifest"]["protocol_hash"] != right["score_manifest"]["protocol_hash"]:
        raise ValueError("Scoring protocol/environment differs; rescore both runs with matching settings")
    if left["config"]["generation"] != right["config"]["generation"]:
        raise ValueError("Generation settings differ")
    if kind != "model" and left["config"]["model"] != right["config"]["model"]:
        raise ValueError("Model settings differ within a paired experiment")
    for lreq, rreq in zip(left["requests"], right["requests"]):
        if lreq["generation"] != rreq["generation"] or lreq["max_new_tokens"] != rreq["max_new_tokens"]:
            raise ValueError("Paired generation budgets differ")
        if kind in ("model", "views", "visual_content"):
            if (lreq["text"], lreq["system"]) != (rreq["text"], rreq["system"]):
                raise ValueError("Paired prompts differ")
        if kind in ("model", "prompt", "answer_form"):
            if image_identity(left, lreq) != image_identity(right, rreq):
                raise ValueError("Paired image content/order differs")
        if kind == "prompt":
            def base_text(request):
                instruction = request["metadata"]["instruction"]
                text = request["text"]
                if instruction and request["metadata"]["prompt_placement"] == "user_suffix":
                    suffix = "\n\n" + instruction
                    if not text.endswith(suffix):
                        raise ValueError("Recorded prompt suffix does not match request")
                    text = text[:-len(suffix)]
                return text
            if base_text(lreq) != base_text(rreq):
                raise ValueError("Prompt comparison changes question/options")


def transition(before, after):
    return ("correct" if before else "wrong") + "_to_" + ("correct" if after else "wrong")


def pair_summary(rows, kind):
    successful = [row for row in rows if row["left_status"] == row["right_status"] == "ok"]
    eligible = [row for row in successful if kind != "views" or row["added_views"]]
    result = {"requested_pairs": len(rows), "successful_pairs": len(successful),
              "excluded_prediction_pairs": len(rows) - len(successful),
              "identical_input_pairs": sum(row["identical_visual_input"] for row in rows),
              "eligible_pairs": len(eligible)}
    for metric in ("option", "judge"):
        scored = [row for row in eligible if row.get(metric + "_transition") is not None]
        result[metric + "_paired"] = len(scored)
        for state in ("wrong_to_correct", "correct_to_wrong", "correct_to_correct", "wrong_to_wrong"):
            result[metric + "_" + state] = sum(row[metric + "_transition"] == state for row in scored)
        result[metric + "_net_gain"] = (result[metric + "_wrong_to_correct"] - result[metric + "_correct_to_wrong"]) / len(scored) if scored else None
    for field in ("token_f1", "bertscore_f1", "output_words", "format_compliant"):
        comparable = [row for row in eligible if row.get("left_" + field) is not None and row.get("right_" + field) is not None]
        result[field + "_paired"] = len(comparable)
        result[field + "_mean_delta"] = mean(row["right_" + field] - row["left_" + field] for row in comparable) if comparable else None
    return result


def compare(left, right, kind):
    validate_pair(left, right, kind)
    rows = []
    for before, after, lreq, rreq in zip(left["rows"], right["rows"], left["requests"], right["requests"]):
        sample = left["sample_map"][before["sample_id"]]
        lviews = set(lreq["metadata"]["selected_view_indices"])
        rviews = set(rreq["metadata"]["selected_view_indices"])
        row = {"sample_id": sample["id"], "left_status": before["status"], "right_status": after["status"],
               "left_prediction": before["prediction"], "right_prediction": after["prediction"],
               "category": sample["metadata"].get("category"), "type": sample["metadata"].get("type"),
               "original_image_count": len(sample["images"]),
               "identical_visual_input": image_identity(left, lreq) == image_identity(right, rreq),
               "added_views": lviews < rviews,
               "necessary_view_missing_before": lreq["metadata"]["necessary_view_missing"],
               "necessary_view_missing_after": rreq["metadata"]["necessary_view_missing"]}
        both_ok = before["status"] == after["status"] == "ok"
        row["option_transition"] = transition(before["correct"], after["correct"]) if both_ok and "correct" in before and "correct" in after else None
        row["judge_transition"] = transition(before["judge_correct"], after["judge_correct"]) if both_ok and before["judge_status"] == after["judge_status"] == "ok" else None
        for field in ("token_f1", "bertscore_f1", "output_words", "format_compliant", "elapsed_seconds", "generation_seconds", "peak_allocated_bytes", "peak_reserved_bytes"):
            row["left_" + field] = before.get(field)
            row["right_" + field] = after.get(field)
        for metric in ("correct", "judge_correct"):
            row["left_" + metric] = before.get(metric) if both_ok else None
            row["right_" + metric] = after.get(metric) if both_ok else None
        rows.append(row)
    groups = []
    for field in ("category", "type", "original_image_count"):
        grouped = defaultdict(list)
        for row in rows:
            grouped[str(row.get(field))].append(row)
        for value, subset in grouped.items():
            groups.append({"group_by": field, "group": value, **pair_summary(subset, kind)})
    return {"kind": kind, "left": left["model_key"] + "/" + left["condition"],
            "right": right["model_key"] + "/" + right["condition"],
            "summary": pair_summary(rows, kind), "groups": groups,
            "efficiency": summarize_pair(left, right, rows), "rows": rows}


def analyze_runs(run_dirs, output_dir, score_ids=None, plots=False):
    if output_dir.exists():
        raise ValueError("Analysis output already exists; choose a new directory")
    if score_ids is not None and len(score_ids) != len(run_dirs):
        raise ValueError("Provide one score ID per run directory, in the same order")
    nodes = [node for directory, score_id in zip(run_dirs, score_ids or [None] * len(run_dirs))
             for node in load_run(directory, score_id)]
    if not nodes:
        raise ValueError("No conditions to analyze")
    identities = {sample_identity(node["samples"]) for node in nodes}
    dataset_versions = {digest({key: value for key, value in node["config"]["dataset"].items() if key != "annotations_file"}) for node in nodes}
    if len(identities) != 1 or len(dataset_versions) != 1:
        raise ValueError("Runs must use exactly the same ordered samples, references, image bytes and dataset version")
    if len({node["score_manifest"]["protocol_hash"] for node in nodes}) != 1:
        raise ValueError("All runs need matching scoring protocol and scoring environment")
    indexed = {}
    for node in nodes:
        key = (node["model_key"], node["condition"])
        if key in indexed:
            raise ValueError(f"Duplicate model/condition across runs: {key}; select one result")
        indexed[key] = node
    comparisons, skipped = {}, []
    models = sorted({node["model_key"] for node in nodes})
    for model in models:
        for before, after, kind in PAIRS:
            if (model, before) in indexed and (model, after) in indexed:
                comparisons[f"{model}__{before}__{after}"] = compare(indexed[model, before], indexed[model, after], kind)
            else:
                skipped.append({"model": model, "left": before, "right": after, "reason": "condition_not_selected"})
    for left_model, right_model in combinations(models, 2):
        for condition in sorted({node["condition"] for node in nodes}):
            if (left_model, condition) in indexed and (right_model, condition) in indexed:
                comparisons[f"{left_model}__{right_model}__{condition}"] = compare(indexed[left_model, condition], indexed[right_model, condition], "model")
    overview, grouped, review = [], [], []
    for node in nodes:
        overview.append({"model": node["model_key"], "condition": node["condition"], **summarize(node["rows"], node["experiment"]["mode"])})
        review.extend(candidates(node))
        for field in ("category", "type", "original_image_count"):
            buckets = defaultdict(list)
            for row in node["rows"]:
                sample = node["sample_map"][row["sample_id"]]
                value = len(sample["images"]) if field == "original_image_count" else sample["metadata"].get(field)
                buckets[str(value)].append(row)
            for value, subset in buckets.items():
                grouped.append({"model": node["model_key"], "condition": node["condition"], "group_by": field,
                                "group": value, **summarize(subset, node["experiment"]["mode"])})
    output_dir.mkdir(parents=True)
    write_json(output_dir / "manifest.json", {"inputs": [{"run_dir": node["run_dir"], "score_id": node["score_id"], "condition": node["condition"]} for node in nodes],
                                              "sample_identity": next(iter(identities)),
                                              "source_sha256": sources(("analysis",), ("core/reporting.py", "core/io.py"))})
    write_json(output_dir / "overview.json", overview)
    write_csv(output_dir / "overview.csv", overview)
    write_csv(output_dir / "groups.csv", grouped)
    efficiencies = []
    for key, comparison in comparisons.items():
        write_json(output_dir / "pairs" / (key + ".json"), comparison)
        write_csv(output_dir / "pairs" / (key + ".csv"), comparison["rows"])
        efficiencies.append({"comparison": key, **comparison["efficiency"]})
        if comparison["kind"] in ("views", "visual_content"):
            for row in comparison["rows"]:
                if row["option_transition"] == "correct_to_wrong" or row["judge_transition"] == "correct_to_wrong":
                    review.append({"comparison": key, "sample_id": row["sample_id"], "reasons": "visual_condition_regression",
                                   "left_prediction": row["left_prediction"], "right_prediction": row["right_prediction"],
                                   "human_verdict": "", "human_notes": ""})
    write_json(output_dir / "review-cases.json", review)
    write_csv(output_dir / "review-cases.csv", review)
    write_csv(output_dir / "efficiency.csv", efficiencies)
    write_json(output_dir / "skipped-pairs.json", skipped)
    figures = []
    if plots:
        from .efficiency import write_plots
        figures = write_plots(nodes, output_dir)
    lines = ["# 实验对照", "", "只比较相同样本及评分协议。逐题转移仅计双方生成成功的题；Judge 转移另要求双方评分成功。",
             "不完整运行仅供排查，不用于正式研究结论；指标覆盖率同时见总表与 overview.csv。",
             "新增视角收益排除图像未增加的题。开放问答分数不与选择题准确率直接相减。",
             "选择题/开放问答主实验同时改变选项与格式要求，只作回答形式对照，不能解释为纯选项效应。",
             "效率对照的条件检查见 efficiency.csv；仅在相同硬件、精度、环境和测量规则下输出配对均值。",
             "开放问答的相似度阈值只用于筛选核验案例，不代表正确性阈值。", "",
             "| 模型 | 条件 | 完成 | 选项准确率 | 格式合规 | Judge | Judge覆盖 | Token F1 | BERTScore | BERT覆盖 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in overview:
        lines.append("| " + " | ".join([row["model"], row["condition"], f"{row['successful']}/{row['requested']}",
                     number(row.get("option_accuracy")), number(row.get("format_compliance")), number(row["judge_accuracy"]),
                     f"{row['judge_scored']}/{row['requested']}", number(row.get("token_f1")), number(row["bertscore_f1"]),
                     f"{row['bertscore_scored']}/{row['requested']}"]) + " |")
    lines += ["", "| 对照 | 有效题数 | 错→对 | 对→错 | Judge配对数 | Judge净变化 |", "|---|---:|---:|---:|---:|---:|"]
    for key, comparison in comparisons.items():
        row = comparison["summary"]
        gained = row['option_wrong_to_correct'] if row['option_paired'] else '—'
        lost = row['option_correct_to_wrong'] if row['option_paired'] else '—'
        lines.append(f"| {key} | {row['eligible_pairs']} | {gained} | {lost} | {row['judge_paired']} | {number(row['judge_net_gain'])} |")
    lines += ["", f"人工核验候选：{len(review)} 条；尚未填写人工结论。缺失条件对照：{len(skipped)} 组。",
              "详细数据见 overview.csv、groups.csv、pairs/、review-cases.csv 和 efficiency.csv。", ""]
    for figure in figures:
        lines += [f"![Efficiency]({figure})", ""]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return str(output_dir)
