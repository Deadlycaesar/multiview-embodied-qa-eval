"""Pure aggregation and Markdown/CSV output; never calls a model or service."""

import csv
from collections import Counter
from statistics import mean, median

from .io import write_json


def summarize(rows, mode):
    count = len(rows)
    successful = [row for row in rows if row["status"] == "ok"]
    result = {"requested": count, "successful": len(successful),
              "errors": sum(row["status"] == "error" for row in rows),
              "missing": sum(row["status"] == "missing" for row in rows),
              "complete": len(successful) == count, "mode": mode,
              "truncated": sum(bool(row.get("reached_token_limit")) for row in successful),
              "mean_output_words": mean(row["output_words"] for row in successful) if successful else None}
    if mode == "mcq":
        result.update(
            option_accuracy=sum(row["correct"] for row in rows) / count if count else None,
            format_compliance=sum(row["format_compliant"] for row in rows) / count if count else None,
            unparsed=sum(row.get("parse_status") == "unparsed" for row in rows),
            ambiguous=sum(row.get("parse_status") == "ambiguous" for row in rows),
            option_probs_recorded=sum(row.get("option_probs_status") == "ok" for row in successful),
        )
        result["option_probs_coverage"] = result["option_probs_recorded"] / count if count else 0
        result["complete"] = result["complete"] and result["option_probs_recorded"] == count
    else:
        result["token_f1"] = mean(row["token_f1"] for row in rows) if rows else None
    for prefix, value in (("judge", "judge_correct"), ("bertscore", "bertscore_f1")):
        scored = [row for row in rows if row.get(prefix + "_status") == "ok"]
        result[prefix + ("_accuracy" if prefix == "judge" else "_f1")] = mean(row[value] for row in scored) if scored else None
        result[prefix + "_scored"] = len(scored)
        result[prefix + "_coverage"] = len(scored) / count if count else 0
        result[prefix + "_statuses"] = dict(Counter(row[prefix + "_status"] for row in rows))
    for field in ("elapsed_seconds", "generation_seconds", "output_tokens", "input_tokens", "visual_tokens", "peak_allocated_bytes", "peak_reserved_bytes"):
        values = [row[field] for row in successful if row.get(field) is not None]
        result["mean_" + field] = mean(values) if values else None
        result["median_" + field] = median(values) if values else None
        if field.startswith("peak_"):
            result["max_" + field] = max(values) if values else None
    return result


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    return "—" if value is None else f"{value:.4f}"


def write_report(directory, summaries):
    write_json(directory / "summary.json", summaries)
    write_csv(directory / "summary.csv", [{"condition": name, **row} for name, row in summaries.items()])
    lines = ["# 评分结果", "", "规则准确率、格式合规率、Token F1 以计划题数为分母；生成失败/缺失记 0。",
             "Judge/BERTScore 只汇总成功评分，必须同时查看覆盖数。未完成的运行不用于正式结论。", "",
             "| 条件 | 完成 | 选项准确率 | 格式合规 | Judge | Judge覆盖 | Token F1 | BERTScore F1 | BERT覆盖 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in summaries.items():
        values = [name, f"{row['successful']}/{row['requested']}", number(row.get("option_accuracy")),
                  number(row.get("format_compliance")), number(row["judge_accuracy"]),
                  f"{row['judge_scored']}/{row['requested']}", number(row.get("token_f1")),
                  number(row["bertscore_f1"]), f"{row['bertscore_scored']}/{row['requested']}"]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "详细资源统计见 summary.csv；逐题评分保存在各条件 JSON 文件中。", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
