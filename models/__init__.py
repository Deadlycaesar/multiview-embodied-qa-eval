"""Lazy adapters: prepare, score and analyze do not import model runtimes."""

from importlib import import_module


ADAPTERS = {
    "gemma4": ("gemma4", "Gemma4"),
    "phi4": ("phi4", "Phi4"),
    "molmo2": ("molmo2", "Molmo2"),
    "lfm25": ("lfm25", "LFM25"),
    "smolvlm2": ("smolvlm2", "SmolVLM2"),
    "qwen35": ("qwen35", "Qwen35"),
    "qwen3vl": ("qwen3vl", "Qwen3VL"),
    "internvl35": ("internvl", "InternVL"),
    "minicpmv4": ("minicpm", "MiniCPM"),
    "sailvl2": ("sailvl", "SAILVL"),
}


def load_model(config, cache_dir):
    adapter = config["adapter"]
    if adapter not in ADAPTERS:
        raise ValueError(f"Unknown model adapter: {adapter}")
    module_name, class_name = ADAPTERS[adapter]
    model_class = getattr(import_module(f".{module_name}", __package__), class_name)
    return model_class(config, cache_dir)
