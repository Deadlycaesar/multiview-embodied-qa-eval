"""Run from the repository directory: python run.py <command> --help."""

import importlib.util
import sys
from pathlib import Path


if __name__ == "__main__":
    # Preserve package-relative imports even if the checkout directory is renamed.
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "evaluation", root / "__init__.py", submodule_search_locations=[str(root)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules["evaluation"] = package
    spec.loader.exec_module(package)
    from evaluation.core.runner import main
    raise SystemExit(main())
