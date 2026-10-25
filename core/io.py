"""Artifact I/O and content identities, independent of inference and scoring."""

import hashlib
import importlib.metadata
import json
from contextlib import contextmanager
from datetime import datetime, timezone

from .paths import ROOT


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sources(folders=(), files=()):
    paths = [ROOT / name for name in files]
    for folder in folders:
        paths.extend((ROOT / folder).rglob("*.py"))
    return {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(set(paths))}


def versions(names):
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


@contextmanager
def lock(directory):
    path = directory / ".active.lock"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(now())
    try:
        yield
    finally:
        path.unlink()
