"""Lazy adapters: prepare, score and analyze do not load a tested VLM."""


def load_model(config, cache_dir):
    if config["adapter"] == "qwen3vl":
        from .qwen3vl import Qwen3VL
        return Qwen3VL(config, cache_dir)
    if config["adapter"] == "qwen35":
        from .qwen35 import Qwen35
        return Qwen35(config, cache_dir)
    raise ValueError(f"Unknown model adapter: {config['adapter']}")
