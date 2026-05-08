"""Small file I/O helpers shared by release scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load records from CSV, JSONL, or JSON files."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        records.append(item)
                    else:
                        raise ValueError(f"JSONL item is not an object in {path}")
        return records

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            if not all(isinstance(item, dict) for item in data):
                raise ValueError(f"JSON list contains non-object items in {path}")
            return data
        if isinstance(data, dict):
            return [
                {"id": key, **value} if isinstance(value, dict) else {"id": key, "value": value}
                for key, value in data.items()
            ]
        raise ValueError(f"Unsupported JSON top-level type in {path}")

    raise ValueError(f"Unsupported file type: {path}")


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def iter_limited(records: Iterable[dict[str, Any]], limit: int | None) -> Iterable[dict[str, Any]]:
    for index, record in enumerate(records):
        if limit is not None and index >= limit:
            break
        yield record
