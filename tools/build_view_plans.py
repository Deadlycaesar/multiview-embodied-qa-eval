"""Regenerate the canonical MCQ view plan; members should not modify its rows."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "evaluation", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)],
)
package = importlib.util.module_from_spec(spec)
sys.modules["evaluation"] = package
spec.loader.exec_module(package)

from evaluation.core.config import read_config
from evaluation.core.paths import project_path, storage_paths
from evaluation.datasets import load_samples
from evaluation.protocols.plans import write_view_plan


def main():
    config = read_config("configs/experiments.mcq.json")
    samples = load_samples(config["dataset"], storage_paths(config)["data_dir"])
    rows = write_view_plan(project_path(config["view_plan"]), samples)
    print(f"Wrote {len(rows)} canonical view-plan rows to {config['view_plan']}")


if __name__ == "__main__":
    main()
