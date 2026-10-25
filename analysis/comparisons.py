"""Cross-machine aggregation for the v4 main, ablation and appendix protocols."""

from collections import defaultdict
from statistics import mean, stdev

from ..core.distribution import validate_run
from ..core.io import digest, read_json, read_jsonl, sources, write_json
from ..core.reporting import number, summarize, write_csv


def sample_identity(samples):
    return digest([{key: value for key, value in sample.items() if key != "images"} for sample in samples])


def metric_identity(score_manifest):
    protocol = score_manifest["protocol"]
    return digest({"metrics": protocol["metrics"], "judge": protocol["judge"],
                   "bertscore_enabled": protocol["bertscore_enabled"], "source_sha256": protocol["source_sha256"]})


def load_run(run_dir, score_id):
    validation = validate_run(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    score_id = score_id or validation["score_id"]
    if score_id is None:
        raise ValueError(f"Run has not been scored: {run_dir}")
    score_dir = run_dir / "scores" / score_id
    completion = read_json(score_dir / "completion.json")
    scoring = read_json(score_dir / "manifest.json")
    if scoring["run_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Scores refer to a different run")
    samples = read_jsonl(run_dir / "samples.jsonl")
    requests = read_json(run_dir / "requests.json")
    model_path = run_dir / "model.json"
    nodes = []
    for name, experiment in manifest["config"]["experiments"].items():
        rows = read_json(score_dir / f"{name}.json")
        if digest(rows) != completion["score_sha256"][name] or len(rows) != len(requests[name]):
            raise ValueError("Score rows were modified or do not match requests")
        for index, row in enumerate(rows):
            path = run_dir / name / f"{index:04d}.json"
            prediction = read_json(path) if path.exists() else {"status": "missing", "text": ""}
            if digest(prediction) != scoring["prediction_sha256"][f"{name}/{index:04d}"]:
                raise ValueError("Predictions changed after scoring; score again before analysis")
            if row["prediction"] != prediction.get("text", "") or row["status"] != prediction["status"]:
                raise ValueError("Scores do not match original predictions")
        nodes.append({
            "run_dir": str(run_dir), "score_id": score_id, "manifest": manifest,
            "score_manifest": scoring, "metric_identity": metric_identity(scoring),
            "config": manifest["config"], "model_metadata": read_json(model_path) if model_path.exists() else {},
            "model_key": manifest["config"]["model_key"], "condition": name,
            "experiment": experiment, "rows": rows, "requests": requests[name], "samples": samples,
            "run_complete": validation["complete"] and completion["complete"],
        })
    return nodes


def accuracy(rows):
    return sum(bool(row.get("correct")) for row in rows) / len(rows) if rows else None


def aggregate_seeds(groups, value_function):
    per_seed = []
    for seed in sorted(groups, key=lambda value: (-1 if value is None else value)):
        rows = groups[seed]
        per_seed.append({"seed": seed, "count": len(rows), **value_function(rows)})
    values = [row["value"] for row in per_seed if row.get("value") is not None]
    return {"seeds": per_seed, "value": mean(values) if values else None,
            "std": stdev(values) if len(values) > 1 else 0.0 if values else None}


def budget_points(node):
    """Return one accuracy/token point per k for random or deterministic budget conditions."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in node["rows"]:
        metadata = row["view_metadata"]
        groups[metadata["k"]][metadata["seed"]].append(row)
    points = []
    for k, seed_groups in sorted(groups.items()):
        aggregated = aggregate_seeds(seed_groups, lambda rows: {
            "value": accuracy(rows), "visual_tokens": mean(row["visual_tokens"] for row in rows)})
        points.append({"k": k, "accuracy": aggregated["value"], "std": aggregated["std"],
                       "visual_tokens": mean(row["visual_tokens"] for row in aggregated["seeds"]),
                       "seeds": aggregated["seeds"]})
    return points


def a2_summary(model_key, evidence_rows, random_rows):
    evidence = {row["sample_id"]: row for row in evidence_rows}
    random_groups = defaultdict(list)
    for row in random_rows:
        random_groups[row["view_metadata"]["seed"]].append(row)
    result = []
    for seed in sorted(random_groups):
        rows = random_groups[seed]
        deltas = [evidence[row["sample_id"]]["correct"] - row["correct"] for row in rows]
        result.append({"model": model_key, "seed": seed, "count": len(rows),
                       "evidence_accuracy": mean(evidence[row["sample_id"]]["correct"] for row in rows),
                       "random_accuracy": accuracy(rows), "evidence_minus_random": mean(deltas)})
    deltas = [row["evidence_minus_random"] for row in result]
    return {"per_seed": result, "mean_delta": mean(deltas) if deltas else None,
            "std_delta": stdev(deltas) if len(deltas) > 1 else 0.0 if deltas else None}


def write_main_plot(models, nodes, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indexed = {(node["model_key"], node["condition"]): node for node in nodes}
    models = [model for model in models if (model, "b1_random_views") in indexed]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, model in zip(axes.flat, models):
        for condition, label, marker in (("b1_random_views", "B1 random", "o"),
                                         ("b2_evidence_views", "B2 evidence priority", "d")):
            node = indexed.get((model, condition))
            if node is None:
                continue
            points = budget_points(node)
            axis.errorbar([point["visual_tokens"] for point in points],
                          [point["accuracy"] for point in points],
                          yerr=[point["std"] for point in points], marker=marker, capsize=3, label=label)
            for point in points:
                axis.annotate(f"k={point['k']}", (point["visual_tokens"], point["accuracy"]))
        b0 = indexed.get((model, "b0_all_views"))
        if b0:
            axis.scatter(mean(row["visual_tokens"] for row in b0["rows"]), accuracy(b0["rows"]),
                         marker="s", label="B0 all")
        a1 = indexed.get((model, "a1_no_image"))
        if a1:
            s1_ids = {sample["id"] for sample in a1["samples"]
                      if sample["metadata"].get("dataset") == "dexycb" and sample["metadata"].get("type") != 4}
            rows = [row for row in a1["rows"] if row["sample_id"] in s1_ids]
            axis.scatter(0, accuracy(rows), marker="^", label="A1 no image (S1)")
        axis.set_title(model)
        axis.set_xlabel("Visual tokens")
        axis.set_ylabel("Accuracy")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.savefig(output_dir / "figures" / "accuracy-visual-tokens.png", dpi=180)
    fig.savefig(output_dir / "figures" / "accuracy-visual-tokens.svg")
    plt.close(fig)
    return ["figures/accuracy-visual-tokens.png", "figures/accuracy-visual-tokens.svg"]


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
    if len(identities) != 1:
        raise ValueError("Runs must use exactly the same ordered samples and references")
    if len({node["manifest"]["protocol_hash"] for node in nodes}) != 1:
        raise ValueError("Runs use different inference protocols")
    if len({node["metric_identity"] for node in nodes}) != 1:
        raise ValueError("Runs use different scoring protocols")
    indexed = {}
    for node in nodes:
        key = (node["model_key"], node["condition"])
        if key in indexed:
            raise ValueError(f"Duplicate model/condition across runs: {key}")
        indexed[key] = node
    models = sorted({node["model_key"] for node in nodes})
    overview, a2_rows = [], []
    budget_outputs = {"b1_random_views": [], "b2_evidence_views": [], "g1_random_views": []}
    for node in nodes:
        summary = summarize(node["rows"], node["experiment"]["mode"])
        official = node["run_complete"] and (node["experiment"]["mode"] == "mcq" or summary["judge_scored"] == summary["requested"])
        sample_map = {sample["id"]: sample for sample in node["samples"]}
        random_chance = (mean(1 / len(sample_map[row["sample_id"]]["choices"]) for row in node["rows"])
                         if node["condition"] == "a1_no_image" else None)
        overview.append({"model": node["model_key"], "condition": node["condition"], "official": official,
                         "random_chance": random_chance,
                         "hardware": node["model_metadata"].get("hardware", {}).get("name"), **summary})
        if node["condition"] == "a2_remove_random":
            evidence = indexed.get((node["model_key"], "a2_remove_evidence"))
            if evidence is None:
                raise ValueError(f"A2 evidence condition missing for {node['model_key']}")
            paired = a2_summary(node["model_key"], evidence["rows"], node["rows"])
            a2_rows.extend(paired["per_seed"])
            a2_rows.append({"model": node["model_key"], "seed": "mean", "count": len(node["rows"]) // 3,
                            "evidence_minus_random": paired["mean_delta"], "std_delta": paired["std_delta"]})
        if node["condition"] in budget_outputs:
            for point in budget_points(node):
                for row in point["seeds"]:
                    budget_outputs[node["condition"]].append({"model": node["model_key"], "k": point["k"], **row})
                budget_outputs[node["condition"]].append({
                    "model": node["model_key"], "k": point["k"], "seed": "mean",
                    "count": sum(row["count"] for row in point["seeds"]),
                    "value": point["accuracy"], "std": point["std"], "visual_tokens": point["visual_tokens"]})
    output_dir.mkdir(parents=True)
    (output_dir / "figures").mkdir()
    write_json(output_dir / "manifest.json", {
        "inputs": [{"run_dir": node["run_dir"], "score_id": node["score_id"], "condition": node["condition"]} for node in nodes],
        "sample_identity": next(iter(identities)),
        "source_sha256": sources(("analysis",), ("core/reporting.py", "core/io.py")),
    })
    write_json(output_dir / "overview.json", overview)
    write_csv(output_dir / "overview.csv", overview)
    write_csv(output_dir / "a2.csv", a2_rows)
    for condition, rows in budget_outputs.items():
        write_csv(output_dir / f"{condition.split('_')[0]}.csv", rows)
    figures = write_main_plot(models, nodes, output_dir) if plots and any(node["condition"] == "b1_random_views" for node in nodes) else []
    lines = ["# 统一实验汇总", "", "只有 official=True 的结果可进入正式结论；开放题还需 Jev 全覆盖。",
             "A2 差值为去掉 Y 的准确率减去同种子随机删除准确率。B1/G1 报告三个种子的均值与标准差；B2 是确定性证据优先上限。",
             "G0/G1 是 H2O 附录泛化，不混入 DexYCB 主曲线。不同机器的耗时和显存不做跨模型排名。", "",
             "| 模型 | 条件 | 正式 | 完成 | 准确率 | Judge | Token F1 | 视觉Token |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in overview:
        metric = row.get("option_accuracy") if row["mode"] == "mcq" else None
        lines.append("| " + " | ".join([row["model"], row["condition"], str(row["official"]),
                     f"{row['successful']}/{row['requested']}", number(metric), number(row.get("judge_accuracy")),
                     number(row.get("token_f1")), number(row.get("mean_visual_tokens"))]) + " |")
    lines += ["", "详细结果见 overview.csv、a2.csv、b1.csv、b2.csv、g1.csv。"]
    lines += [f"![Accuracy vs visual tokens]({figures[0]})"] if figures else []
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(output_dir)
