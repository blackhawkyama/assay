"""Deterministic scorers: cheap, exact, no model calls.

Reach for these first. Anything you can check with code — schema validity,
numeric bounds, a required substring, latency — belongs here, not in the judge.
They cost nothing, never flake, and make the judge's job (and bill) smaller.

Each reads what it needs from `case.expected` and/or `prediction.output`. A
prediction that errored scores 0 everywhere except where noted.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from assay.scorers.base import get_path, make_score
from assay.types import Case, Prediction, Score


class NoError:
    """1.0 if the System produced output without error, else 0.0. The floor
    check every suite should include — an error-free run can still be wrong,
    but an errored one is never right."""

    def __init__(self, name: str = "no_error") -> None:
        self.name = name

    def score(self, case: Case, prediction: Prediction) -> Score:
        ok = prediction.ok
        return make_score(
            self.name,
            1.0 if ok else 0.0,
            passed=ok,
            detail="" if ok else (prediction.error or "errored"),
        )


class ExactMatch:
    """Output equals `case.expected[key]` exactly (after optional normalization)."""

    def __init__(
        self,
        key: str = "output",
        name: str = "exact_match",
        casefold: bool = False,
        strip: bool = True,
    ) -> None:
        self.key = key
        self.name = name
        self.casefold = casefold
        self.strip = strip

    def _norm(self, v: Any) -> Any:
        if isinstance(v, str):
            if self.strip:
                v = v.strip()
            if self.casefold:
                v = v.casefold()
        return v

    def score(self, case: Case, prediction: Prediction) -> Score:
        if not prediction.ok:
            return make_score(self.name, 0.0, passed=False, detail="prediction errored")
        got = self._norm(prediction.output)
        want = self._norm(case.expected.get(self.key))
        match = got == want
        return make_score(
            self.name,
            1.0 if match else 0.0,
            passed=match,
            detail="" if match else f"expected {want!r}, got {got!r}",
        )


class Contains:
    """Prediction output (stringified) contains every required substring from
    `case.expected[key]` (a string or list). Score is the fraction present."""

    def __init__(
        self, key: str = "contains", name: str = "contains", casefold: bool = True
    ) -> None:
        self.key = key
        self.name = name
        self.casefold = casefold

    def score(self, case: Case, prediction: Prediction) -> Score:
        if not prediction.ok:
            return make_score(self.name, 0.0, passed=False, detail="prediction errored")
        needles = case.expected.get(self.key, [])
        if isinstance(needles, str):
            needles = [needles]
        if not needles:
            return make_score(self.name, 1.0, passed=True, detail="no needles specified")
        hay = str(prediction.output)
        if self.casefold:
            hay = hay.casefold()
            needles = [n.casefold() for n in needles]
        hits = [n for n in needles if n in hay]
        frac = len(hits) / len(needles)
        missing = [n for n in needles if n not in hay]
        return make_score(
            self.name,
            frac,
            passed=frac == 1.0,
            detail="" if not missing else f"missing: {missing}",
        )


class Regex:
    """Output (stringified) matches the pattern in `case.expected[key]` or a
    fixed pattern passed at construction."""

    def __init__(
        self,
        pattern: Optional[str] = None,
        key: str = "pattern",
        name: str = "regex",
        flags: int = 0,
    ) -> None:
        self.pattern = pattern
        self.key = key
        self.name = name
        self.flags = flags

    def score(self, case: Case, prediction: Prediction) -> Score:
        if not prediction.ok:
            return make_score(self.name, 0.0, passed=False, detail="prediction errored")
        pat = self.pattern or case.expected.get(self.key)
        if not pat:
            return make_score(self.name, 0.0, passed=False, detail="no pattern")
        match = re.search(pat, str(prediction.output), self.flags) is not None
        return make_score(
            self.name,
            1.0 if match else 0.0,
            passed=match,
            detail="" if match else f"no match for /{pat}/",
        )


class JSONSchemaValid:
    """Output validates against a Pydantic model (structural gate for structured
    outputs). If pydantic can build the model from the output, it passes."""

    def __init__(self, model: type, name: str = "schema_valid") -> None:
        self.model = model
        self.name = name

    def score(self, case: Case, prediction: Prediction) -> Score:
        if not prediction.ok:
            return make_score(self.name, 0.0, passed=False, detail="prediction errored")
        try:
            self.model.model_validate(prediction.output)  # type: ignore[attr-defined]
            return make_score(self.name, 1.0, passed=True)
        except Exception as exc:  # noqa: BLE001
            return make_score(self.name, 0.0, passed=False, detail=str(exc)[:300])


class PathEquals:
    """A dotted path in the output equals the expected value.

    `path` addresses into `prediction.output`; the target comes from
    `case.expected[expect_key]` (defaults to the leaf name of `path`)."""

    def __init__(
        self, path: str, expect_key: Optional[str] = None, name: Optional[str] = None
    ) -> None:
        self.path = path
        self.expect_key = expect_key or path.split(".")[-1]
        self.name = name or f"path:{path}"

    def score(self, case: Case, prediction: Prediction) -> Score:
        if not prediction.ok:
            return make_score(self.name, 0.0, passed=False, detail="prediction errored")
        got = get_path(prediction.output, self.path)
        want = case.expected.get(self.expect_key)
        match = got == want
        return make_score(
            self.name,
            1.0 if match else 0.0,
            passed=match,
            detail="" if match else f"{self.path}: expected {want!r}, got {got!r}",
        )


class NumericBounds:
    """A numeric value at `path` sits within a tolerance of the expected target,
    or within an absolute [lo, hi] window. Score decays linearly to 0 at the
    tolerance edge, so 'close' beats 'wildly off' instead of both scoring 0."""

    def __init__(
        self,
        path: str,
        expect_key: Optional[str] = None,
        name: Optional[str] = None,
        rel_tol: float = 0.10,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
    ) -> None:
        self.path = path
        self.expect_key = expect_key or path.split(".")[-1]
        self.name = name or f"num:{path}"
        self.rel_tol = rel_tol
        self.lo = lo
        self.hi = hi

    def score(self, case: Case, prediction: Prediction) -> Score:
        if not prediction.ok:
            return make_score(self.name, 0.0, passed=False, detail="prediction errored")
        raw = get_path(prediction.output, self.path)
        try:
            got = float(raw)
        except (TypeError, ValueError):
            return make_score(self.name, 0.0, passed=False, detail=f"non-numeric: {raw!r}")

        if self.lo is not None or self.hi is not None:
            lo = self.lo if self.lo is not None else float("-inf")
            hi = self.hi if self.hi is not None else float("inf")
            inside = lo <= got <= hi
            return make_score(
                self.name,
                1.0 if inside else 0.0,
                passed=inside,
                detail="" if inside else f"{got} not in [{self.lo}, {self.hi}]",
            )

        target = case.expected.get(self.expect_key)
        if target is None:
            return make_score(self.name, 0.0, passed=False, detail="no target")
        target = float(target)
        band = abs(target) * self.rel_tol
        if band == 0:
            match = got == target
            return make_score(self.name, 1.0 if match else 0.0, passed=match)
        err = abs(got - target)
        value = max(0.0, 1.0 - err / band)
        passed = err <= band
        return make_score(
            self.name,
            value,
            passed=passed,
            detail="" if passed else f"{got} vs target {target} (±{self.rel_tol:.0%})",
        )


class LatencyUnder:
    """Wall-clock latency under a budget. A guardrail, not a quality metric —
    keep it in the suite so a quality win that tanks latency still shows up."""

    def __init__(self, budget_s: float, name: str = "latency") -> None:
        self.budget_s = budget_s
        self.name = name

    def score(self, case: Case, prediction: Prediction) -> Score:
        lat = prediction.latency_s
        if lat is None:
            return make_score(self.name, 0.0, passed=None, detail="no latency recorded")
        under = lat <= self.budget_s
        return make_score(
            self.name,
            1.0 if under else 0.0,
            passed=under,
            detail=f"{lat:.2f}s (budget {self.budget_s:.2f}s)",
            metadata={"latency_s": lat},
        )
