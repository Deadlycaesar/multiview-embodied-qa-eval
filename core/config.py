"""Resolve and strictly validate the shared v4 experiment configuration."""

import json
import re
from copy import deepcopy

from .paths import project_path


EXPERIMENT_FIELDS = {"sample_set", "views", "mode", "prompt", "max_new_tokens", "model_group"}
ADAPTERS = ("gemma4", "phi4", "molmo2", "lfm25", "smolvlm2", "qwen35", "qwen3vl",
            "internvl35", "minicpmv4", "sailvl2")


def pinned(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("Use an immutable 40-character revision")


def validate_model(model):
    pinned(model["revision"])
    if model["adapter"] not in ADAPTERS:
        raise ValueError("Unknown model adapter")
    if model.get("device") not in ("cuda", "cpu", "mps", "auto"):
        raise ValueError("Unknown model device")
    if model.get("dtype") not in ("float32", "float16", "bfloat16"):
        raise ValueError("Unknown model dtype")
    if model.get("attn_implementation") not in ("eager", "sdpa"):
        raise ValueError("Supported attention implementations: eager, sdpa")
    if model["adapter"] == "minicpmv4" and model["attn_implementation"] != "sdpa":
        raise ValueError("MiniCPM-V-4 requires sdpa attention")
    if type(model.get("trust_remote_code", False)) is not bool:
        raise ValueError("trust_remote_code must be boolean")
    visual_ids = model.get("visual_token_ids")
    if visual_ids is not None and (not isinstance(visual_ids, list) or not visual_ids
                                   or any(type(value) is not int or value < 0 for value in visual_ids)):
        raise ValueError("visual_token_ids must be a nonempty list of nonnegative integers")
    processor = model.get("processor", {})
    if not isinstance(processor, dict) or set(processor) - {"min_pixels", "max_pixels"}:
        raise ValueError("Supported processor overrides: min_pixels, max_pixels")
    if any(type(value) is not int or value < 1 for value in processor.values()):
        raise ValueError("Pixel limits must be positive integers")
    if "min_pixels" in processor and "max_pixels" in processor and processor["min_pixels"] > processor["max_pixels"]:
        raise ValueError("min_pixels exceeds max_pixels")
    if not isinstance(model.get("chat_template_kwargs", {}), dict):
        raise ValueError("chat_template_kwargs must be an object")
    if type(model.get("seed")) is not int:
        raise ValueError("Model seed must be an integer")


def select_experiments(config, model_key, requested):
    all_experiments = config["experiments"]
    groups = config["model_groups"]
    if set(groups["all"]) != set(config["models"]):
        raise ValueError("The all model group must exactly match the model registry")
    for name, experiment in all_experiments.items():
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or set(experiment) != EXPERIMENT_FIELDS:
            raise ValueError(f"Invalid condition name or fields: {name}")
        if experiment["sample_set"] not in ("S0", "S1", "S2", "H2O", "OPEN"):
            raise ValueError(f"Invalid sample set: {name}")
        if experiment["views"] not in ("plan", "all") or experiment["mode"] not in ("mcq", "open"):
            raise ValueError(f"Invalid condition: {name}")
        if (experiment["mode"] == "mcq") != (experiment["views"] == "plan"):
            raise ValueError(f"MCQ uses plan views and Open uses all views: {name}")
        if experiment["mode"] == "open" and experiment["sample_set"] != "OPEN":
            raise ValueError(f"Open conditions must use OPEN: {name}")
        if experiment["prompt"] not in ("mcq", "none"):
            raise ValueError("Unknown instruction")
        if (experiment["mode"] == "mcq") != (experiment["prompt"] == "mcq"):
            raise ValueError(f"Prompt does not match answer mode: {name}")
        if type(experiment["max_new_tokens"]) is not int or experiment["max_new_tokens"] < 1:
            raise ValueError("max_new_tokens must be positive")
        if experiment["model_group"] not in groups or model_key not in groups[experiment["model_group"]]:
            if requested and name in requested:
                raise ValueError(f"{model_key} is not assigned to condition {name}")
    allowed = {name for name, experiment in all_experiments.items()
               if model_key in groups[experiment["model_group"]]}
    if requested:
        unknown = set(requested) - set(all_experiments)
        if unknown:
            raise ValueError(f"Unknown conditions: {sorted(unknown)}")
        if not set(requested) <= allowed:
            raise ValueError("A requested condition is not assigned to this model")
        selected = list(requested)
    else:
        selected = [name for name in all_experiments if name in allowed]
    if not selected:
        raise ValueError("Select at least one condition")
    config["canonical_experiments"] = deepcopy(all_experiments)
    config["experiments"] = {name: all_experiments[name] for name in selected}


def read_config(path, experiments=None, limit=None, model_name=None, smoke_offset=None):
    config = json.loads(project_path(path).read_text())
    if "extends" in config:
        base = json.loads(project_path(config.pop("extends")).read_text())
        if "extends" in base:
            raise ValueError("Only one level of configuration inheritance is supported")
        config = {**base, **config}
    config = deepcopy(config)
    if config.get("schema_version") != 4:
        raise ValueError("Use the v4 shared experiment configuration")
    key = model_name or config["default_model"]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", key) or key not in config["models"]:
        raise ValueError("Unknown or invalid model key")
    select_experiments(config, key, experiments)
    config["model"] = json.loads(project_path(config["models"][key]).read_text())
    config["model_key"] = key
    del config["models"]
    del config["default_model"]
    pinned(config["dataset"]["revision"])
    validate_model(config["model"])
    if config["generation"] != {"do_sample": False, "num_beams": 1}:
        raise ValueError("This protocol uses greedy decoding")
    warmup = config["runtime"]["warmup_per_condition"]
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup_per_condition must be nonnegative")
    prompts = config["prompts"]
    if set(prompts) != {"version", "placement", "mcq", "none"}:
        raise ValueError("Unknown or missing prompt settings")
    if prompts["placement"] not in ("user_suffix", "system"):
        raise ValueError("Unknown prompt placement")
    if not prompts["version"] or not prompts["mcq"] or prompts["none"] != "":
        raise ValueError("Invalid prompt protocol")
    if any(e["views"] == "plan" for e in config["experiments"].values()) and not config.get("view_plan"):
        raise ValueError("MCQ conditions require a view plan")
    count, offset = config["dataset"]["count"], config["dataset"].get("offset", 0)
    if type(count) is not int or count < 1 or type(offset) is not int or offset < 0:
        raise ValueError("Invalid sample selection")
    if smoke_offset is not None:
        if limit is None or type(smoke_offset) is not int or not 0 <= smoke_offset < count:
            raise ValueError("A smoke offset requires --limit and must be inside the configured dataset")
        config["dataset"]["offset"] = smoke_offset
    if limit is not None:
        if not 1 <= limit <= count:
            raise ValueError("limit must be between 1 and configured count")
        config["dataset"]["count"] = limit
    return config
