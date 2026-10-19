"""Review candidates are flags, never automatic human judgments."""


def candidates(node):
    result = []
    thresholds = node["score_manifest"]["protocol"]["metrics"]["review"]
    for row in node["rows"]:
        reasons = []
        if row["status"] != "ok":
            reasons.append("prediction_" + row["status"])
        if row.get("reached_token_limit"):
            reasons.append("truncated")
        if row.get("judge_status") == "error" or row.get("bertscore_status") == "error":
            reasons.append("scoring_error")
        if row.get("parse_status") in ("ambiguous", "unparsed"):
            reasons.append("choice_" + row["parse_status"])
        if node["experiment"]["mode"] == "open" and row.get("bertscore_status") == "ok":
            lexical, semantic = row["token_f1"], row["bertscore_f1"]
            if ((lexical < thresholds["low_token_f1"] and semantic > thresholds["high_bertscore"])
                    or (lexical > thresholds["high_token_f1"] and semantic < thresholds["low_bertscore"])):
                reasons.append("token_f1_bertscore_disagreement")
        if row["judge_status"] == "ok":
            judged = row["judge_correct"]
            if node["experiment"]["mode"] == "mcq":
                if judged != row["correct"]:
                    reasons.append("rule_judge_disagreement")
            else:
                for metric, field in (("token_f1", "token_f1"), ("bertscore", "bertscore_f1")):
                    value = row.get(field)
                    if value is not None and ((judged and value < thresholds["low_" + metric]) or (not judged and value > thresholds["high_" + metric])):
                        reasons.append(metric + "_judge_disagreement")
        if reasons:
            sample = node["sample_map"][row["sample_id"]]
            result.append({
                "model": node["model_key"], "condition": node["condition"], "sample_id": row["sample_id"],
                "reasons": ";".join(reasons), "question": sample["question"], "choices": sample["choices"],
                "images": sample["metadata"]["source_images"], "view_metadata": row["view_metadata"],
                "prediction": row["prediction"], "references": row["reference"],
                "token_f1": row.get("token_f1"),
                "bertscore_f1": row.get("bertscore_f1"), "human_verdict": "", "human_notes": "",
            })
    return result
