"""Resolve relative paths at the project root, one level above core/."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def storage_paths(config: dict) -> dict[str, Path]:
    defaults = {"data_dir": "data", "model_cache_dir": "weights", "runs_dir": "runs"}
    values = config.get("storage", {})
    return {name: project_path(values.get(name, default)) for name, default in defaults.items()}
