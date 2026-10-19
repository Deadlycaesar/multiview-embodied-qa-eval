"""Resolve shared settings without model imports or network access."""

import json
import re
from copy import deepcopy

from .paths import project_path


def pinned(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("Use an immutable 40-character revision")


def read_config(path, experiments=None, limit=None, model_name=None):
    config = json.loads(project_path(path).read_text())
    if "extends" in config:
        base = json.loads(project_path(config.pop("extends")).read_text())
        if "extends" in base:
            raise ValueError("Only one level of configuration inheritance is supported")
        config = {**base, **config}
    config = deepcopy(config)
    if config.get("schema_version") != 2:
        raise ValueError("Use the v2 shared experiment configuration")
    key = model_name or config["default_model"]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", key):
        raise ValueError("Model keys must be simple directory names")
    config["model"] = json.loads(project_path(config["models"][key]).read_text())
    config["model_key"] = key
    del config["models"]
    del config["default_model"]
    pinned(config["dataset"]["revision"])
    pinned(config["model"]["revision"])
    if config["model"]["adapter"] not in ("qwen3vl", "qwen35"):
        raise ValueError("Unknown model adapter")
    if config["model"].get("device") not in ("cuda", "cpu", "mps", "auto"):
        raise ValueError("Unknown model device")
    if config["model"].get("dtype") not in ("float32", "float16", "bfloat16"):
        raise ValueError("Unknown model dtype")
    if config["model"].get("attn_implementation") not in ("eager", "sdpa"):
        raise ValueError("Supported attention implementations: eager, sdpa")
    processor = config["model"].get("processor", {})
    if not isinstance(processor, dict) or set(processor) - {"min_pixels", "max_pixels"}:
        raise ValueError("Supported processor overrides: min_pixels, max_pixels")
    if any(type(value) is not int or value < 1 for value in processor.values()):
        raise ValueError("Pixel limits must be positive integers")
    if "min_pixels" in processor and "max_pixels" in processor and processor["min_pixels"] > processor["max_pixels"]:
        raise ValueError("min_pixels exceeds max_pixels")
    if not isinstance(config["model"].get("chat_template_kwargs", {}), dict):
        raise ValueError("chat_template_kwargs must be an object")
    if type(config["model"].get("seed")) is not int:
        raise ValueError("Model seed must be an integer")
    if config["generation"] != {"do_sample": False, "num_beams": 1}:
        raise ValueError("This protocol uses greedy decoding")
    warmup = config["runtime"]["warmup_per_condition"]
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup_per_condition must be nonnegative")
    prompts = config["prompts"]
    if prompts["placement"] not in ("user_suffix", "system"):
        raise ValueError("Unknown prompt placement")
    if not prompts["version"] or prompts["none"] != "":
        raise ValueError("Prompts need a version and an empty 'none' instruction")
    all_experiments = config["experiments"]
    for left, right in (("multi_mcq", "multi_mcq_no_instruction"), ("multi_open", "multi_open_short")):
        if left in all_experiments and right in all_experiments:
            for field in ("images", "mode", "max_new_tokens"):
                if all_experiments[left][field] != all_experiments[right][field]:
                    raise ValueError(f"Prompt pair differs in {field}: {left}, {right}")
    if experiments:
        aliases = {e["legacy_id"]: name for name, e in all_experiments.items() if "legacy_id" in e}
        selected = [aliases.get(name, name) for name in experiments]
        config["experiments"] = {name: all_experiments[name] for name in selected}
    if not config["experiments"]:
        raise ValueError("Select at least one condition")
    for name, experiment in config["experiments"].items():
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError("Condition names must be simple directory names")
        if experiment["images"] not in ("white", "first1", "first2", "all") or experiment["mode"] not in ("mcq", "open"):
            raise ValueError(f"Invalid condition: {name}")
        if experiment["prompt"] not in ("mcq", "short", "none"):
            raise ValueError("Unknown instruction")
        if type(experiment["max_new_tokens"]) is not int or experiment["max_new_tokens"] < 1:
            raise ValueError("max_new_tokens must be positive")
    count, offset = config["dataset"]["count"], config["dataset"].get("offset", 0)
    if type(count) is not int or count < 1 or type(offset) is not int or offset < 0:
        raise ValueError("Invalid sample selection")
    if limit is not None:
        if not 1 <= limit <= count:
            raise ValueError("limit must be between 1 and configured count")
        config["dataset"]["count"] = limit
    if any(e["images"] == "white" for e in config["experiments"].values()):
        white = config["white_image"]
        if white.get("mode") not in ("file", "generated"):
            raise ValueError("White image is pending: set white_image.mode to file or generated, or exclude white_mcq")
        if white["mode"] == "file" and (not white.get("path") or not project_path(white["path"]).is_file()):
            raise ValueError("Set white_image.path to an existing image")
        if white["mode"] == "generated" and (type(white.get("size")) is not int or white["size"] < 1):
            raise ValueError("Generated white image needs a positive size")
    return config
