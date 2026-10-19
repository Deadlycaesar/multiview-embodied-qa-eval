"""Dataset entry point; add a new adapter here when needed."""


def load_samples(config, data_dir):
    adapter = config["adapter"]
    if adapter == "mvqa":
        from .mvqa import load_samples as load_adapter
    else:
        raise ValueError(f"Unknown dataset adapter: {adapter}")
    return load_adapter(config, data_dir)
