"""Golden datasets: versioned collections of Cases loaded from disk.

Two on-disk shapes, both plain text so they diff well in git:

  - JSONL: one JSON object per line, each a Case.
  - YAML:  a mapping with `name`, `version`, and a `cases:` list.

Version the dataset whenever you add, remove, or change cases. A run records
the version it evaluated, so a later comparison can refuse to diff two runs
that scored different data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import yaml

from assay.types import Case


class Dataset:
    def __init__(
        self,
        cases: Iterable[Case],
        name: str = "dataset",
        version: str = "unversioned",
    ) -> None:
        self.cases: list[Case] = list(cases)
        self.name = name
        self.version = version
        self._check_unique_ids()

    def _check_unique_ids(self) -> None:
        seen: set[str] = set()
        for c in self.cases:
            if c.id in seen:
                raise ValueError(f"duplicate case id in dataset {self.name!r}: {c.id!r}")
            seen.add(c.id)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[Case]:
        return iter(self.cases)

    def filter(self, tag: str | None = None) -> "Dataset":
        cases = [c for c in self.cases if tag is None or tag in c.tags]
        return Dataset(cases, name=self.name, version=self.version)

    # -- loaders -----------------------------------------------------------

    @classmethod
    def from_jsonl(cls, path: str | Path, version: str = "unversioned") -> "Dataset":
        p = Path(path)
        cases = [
            Case.model_validate_json(line)
            for line in p.read_text().splitlines()
            if line.strip()
        ]
        return cls(cases, name=p.stem, version=version)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Dataset":
        p = Path(path)
        doc = yaml.safe_load(p.read_text()) or {}
        cases = [Case.model_validate(c) for c in doc.get("cases", [])]
        return cls(
            cases,
            name=doc.get("name", p.stem),
            version=str(doc.get("version", "unversioned")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        """Dispatch on file extension."""
        p = Path(path)
        if p.suffix in (".yaml", ".yml"):
            return cls.from_yaml(p)
        if p.suffix in (".jsonl", ".ndjson"):
            return cls.from_jsonl(p)
        raise ValueError(f"unsupported dataset file: {p.suffix} ({p})")

    def to_jsonl(self, path: str | Path) -> None:
        Path(path).write_text(
            "\n".join(c.model_dump_json(exclude_defaults=True) for c in self.cases) + "\n"
        )
