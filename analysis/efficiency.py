"""Only compare measured efficiency under matching hardware and runtime policies."""

from statistics import mean


def compatibility(left, right):
    reasons = []
    for key in ("device", "dtype", "hardware", "attn_implementation", "measurement_version", "timing_scope", "memory_scope"):
        if not left["model_metadata"].get(key) or left["model_metadata"].get(key) != right["model_metadata"].get(key):
            reasons.append(key)
    if left["config"]["runtime"] != right["config"]["runtime"]:
        reasons.append("warmup_policy")
    if left["config"]["generation"] != right["config"]["generation"]:
        reasons.append("generation_policy")
    if left["experiment"]["mode"] != right["experiment"]["mode"]:
        reasons.append("answer_form")
    if left["experiment"]["max_new_tokens"] != right["experiment"]["max_new_tokens"]:
        reasons.append("token_limit")
    if left["manifest"]["environment"] != right["manifest"]["environment"]:
        reasons.append("runtime_environment")
    return reasons


def summarize_pair(left, right, rows):
    reasons = compatibility(left, right)
    successful = [row for row in rows if row["left_status"] == row["right_status"] == "ok"]
    result = {"comparable": not reasons, "mismatches": reasons, "paired_successful": len(successful)}
    if not reasons:
        for metric in ("correct", "judge_correct"):
            measured = [row for row in successful if row.get("left_" + metric) is not None and row.get("right_" + metric) is not None]
            result[metric + "_paired"] = len(measured)
            result[metric + "_left_mean"] = mean(row["left_" + metric] for row in measured) if measured else None
            result[metric + "_right_mean"] = mean(row["right_" + metric] for row in measured) if measured else None
        for field in ("elapsed_seconds", "generation_seconds", "peak_allocated_bytes", "peak_reserved_bytes"):
            measured = [row for row in successful if row.get("left_" + field) is not None and row.get("right_" + field) is not None]
            result[field + "_paired"] = len(measured)
            result[field + "_left_mean"] = mean(row["left_" + field] for row in measured) if measured else None
            result[field + "_right_mean"] = mean(row["right_" + field] for row in measured) if measured else None
    return result


def write_plots(nodes, output_dir):
    """Export standalone figures; only complete results with compatible policies share axes."""
    from collections import defaultdict
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..core.io import digest, write_json
    from ..core.reporting import summarize

    groups = defaultdict(list)
    for node in nodes:
        mode = node["experiment"]["mode"]
        summary = summarize(node["rows"], mode)
        if not summary["complete"] or not node["model_metadata"]:
            continue
        if mode == "open" and summary["judge_scored"] != summary["requested"]:
            continue
        metadata = node["model_metadata"]
        policy = {key: metadata.get(key) for key in ("hardware", "device", "dtype", "attn_implementation", "measurement_version", "timing_scope", "memory_scope")}
        policy.update(runtime=node["config"]["runtime"], generation=node["config"]["generation"],
                      max_new_tokens=node["experiment"]["max_new_tokens"], mode=mode,
                      environment=node["manifest"]["environment"])
        groups[digest(policy)[:12]].append((node, summary, policy))
    directory = output_dir / "figures"
    directory.mkdir(exist_ok=True)
    figures = []
    policies = {}
    for key, entries in groups.items():
        policies[key] = entries[0][2]
        mode = entries[0][0]["experiment"]["mode"]
        metric = "option_accuracy" if mode == "mcq" else "judge_accuracy"
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        for node, summary, _ in entries:
            label = node["model_key"] + "/" + node["condition"]
            for axis, field, scale in ((axes[0], "mean_elapsed_seconds", 1), (axes[1], "max_peak_allocated_bytes", 1024 ** 3)):
                if summary[field] is not None:
                    x, y = summary[field] / scale, summary[metric]
                    axis.scatter(x, y, s=36)
                    axis.annotate(label, (x, y), xytext=(4, 5), textcoords="offset points", fontsize=6)
        axes[0].set_xlabel("Mean end-to-end seconds / question")
        axes[1].set_xlabel("Maximum CUDA allocated memory (GiB)")
        for axis in axes:
            axis.set_ylabel(metric.replace("_", " "))
            axis.set_ylim(-0.03, 1.08)
            axis.grid(alpha=0.2)
        fig.suptitle(f"{mode}: complete runs, matched hardware and runtime policy")
        name = f"efficiency-{mode}-{key}"
        fig.savefig(directory / (name + ".png"), dpi=180, bbox_inches="tight")
        fig.savefig(directory / (name + ".svg"), bbox_inches="tight")
        plt.close(fig)
        figures.append("figures/" + name + ".png")
    write_json(directory / "policies.json", policies)
    return figures
