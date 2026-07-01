"""Tests for the assay spine. The judge path is covered with a fake client so
the suite runs with no API key."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from assay import report
from assay.dataset import Dataset
from assay.runner import run
from assay.scorers import (
    Contains,
    ExactMatch,
    LatencyUnder,
    LLMJudge,
    NoError,
    NumericBounds,
    PathEquals,
    Regex,
    Rubric,
)
from assay.scorers import judge as judge_mod
from assay.systems import CallableSystem
from assay.types import Case, Prediction, RunResult


# --- fixtures -------------------------------------------------------------


def make_case(**kw) -> Case:
    return Case(id=kw.pop("id", "c1"), **kw)


def ok(output) -> Prediction:
    return Prediction(output=output, latency_s=0.01)


# --- deterministic scorers ------------------------------------------------


def test_no_error():
    assert NoError().score(make_case(), ok("x")).value == 1.0
    bad = Prediction(error="boom")
    s = NoError().score(make_case(), bad)
    assert s.value == 0.0 and s.passed is False


def test_exact_match_normalization():
    c = make_case(expected={"output": "Positive"})
    s = ExactMatch(casefold=True).score(c, ok("positive "))
    assert s.value == 1.0 and s.passed


def test_contains_fraction():
    c = make_case(expected={"contains": ["alpha", "beta", "gamma"]})
    s = Contains().score(c, ok("alpha and beta only"))
    assert s.value == pytest.approx(2 / 3)
    assert s.passed is False
    assert "gamma" in s.detail


def test_regex():
    c = make_case(expected={"pattern": r"\d{3}-\d{4}"})
    assert Regex().score(c, ok("call 555-1234")).passed
    assert not Regex().score(c, ok("no number")).passed


def test_path_equals_nested():
    c = make_case(expected={"severity": "high"})
    pred = ok({"summary": {"risks": [{"severity": "high"}]}})
    s = PathEquals("summary.risks.0.severity").score(c, pred)
    assert s.value == 1.0 and s.passed


def test_numeric_bounds_tolerance_and_decay():
    c = make_case(expected={"total": 1000.0})
    nb = NumericBounds("total", rel_tol=0.10)
    assert nb.score(c, ok({"total": 1000})).value == 1.0
    assert nb.score(c, ok({"total": 1050})).passed  # within 10%
    mid = nb.score(c, ok({"total": 1050}))
    assert 0.0 < mid.value < 1.0  # linear decay, not binary
    assert not nb.score(c, ok({"total": 2000})).passed


def test_numeric_bounds_window():
    c = make_case()
    nb = NumericBounds("x", lo=0, hi=10)
    assert nb.score(c, ok({"x": 5})).passed
    assert not nb.score(c, ok({"x": 11})).passed


def test_latency_budget():
    c = make_case()
    assert LatencyUnder(0.5).score(c, ok("x")).passed
    assert not LatencyUnder(0.005).score(c, ok("x")).passed


def test_errored_prediction_scores_zero_everywhere():
    c = make_case(expected={"output": "y", "total": 1.0})
    bad = Prediction(error="boom")
    for scorer in (ExactMatch(), NumericBounds("total"), PathEquals("a.b")):
        assert scorer.score(c, bad).value == 0.0


# --- dataset --------------------------------------------------------------


def test_dataset_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        Dataset([make_case(id="dup"), make_case(id="dup")])


def test_dataset_filter_by_tag():
    ds = Dataset([make_case(id="a", tags=["x"]), make_case(id="b", tags=["y"])])
    assert [c.id for c in ds.filter("x")] == ["a"]


# --- runner + aggregation -------------------------------------------------


def _toy_run() -> RunResult:
    ds = Dataset(
        [
            Case(id="p1", input={"t": "good"}, expected={"output": "pos"}, tags=["s"]),
            Case(id="n1", input={"t": "bad"}, expected={"output": "neg"}, tags=["s"]),
            Case(id="x1", input={"t": "meh"}, expected={"output": "neg"}, tags=["hard"]),
        ],
        name="toy",
        version="v1",
    )
    sut = CallableSystem(lambda i: "pos" if i["t"] == "good" else "neg", name="sut")
    return run(
        sut,
        ds,
        [NoError(), ExactMatch(key="output", name="acc")],
        save=False,
        progress=False,
    )


def test_runner_end_to_end():
    r = _toy_run()
    assert len(r.results) == 3
    assert r.error_rate() == 0.0
    assert r.mean("acc") == pytest.approx(1.0)  # all three correct
    # order preserved
    assert [cr.case.id for cr in r.results] == ["p1", "n1", "x1"]


def test_runner_catches_scorer_exceptions():
    class Boom:
        name = "boom"

        def score(self, case, prediction):
            raise RuntimeError("kaboom")

    ds = Dataset([make_case(id="a")], name="d")
    r = run(CallableSystem(lambda i: 1), ds, [Boom()], save=False, progress=False)
    s = r.results[0].score_for("boom")
    assert s is not None and s.value == 0.0 and "kaboom" in s.detail


def test_summary_and_by_tag():
    r = _toy_run()
    summ = r.summary()
    assert summ["n_cases"] == 3
    assert summ["scorers"]["acc"]["mean"] == pytest.approx(1.0)
    tags = r.by_tag("acc")
    assert set(tags) == {"s", "hard"}


# --- compare + gate -------------------------------------------------------


def _run_with_scores(vals: dict[str, float], name="acc", run_id="r") -> RunResult:
    from assay.types import CaseResult, Score

    results = [
        CaseResult(
            case=Case(id=cid),
            prediction=ok("x"),
            scores=[Score(scorer=name, value=v, passed=v >= 0.5)],
        )
        for cid, v in vals.items()
    ]
    return RunResult(run_id=run_id, system="s", dataset="d", dataset_version="v1", results=results)


def test_compare_detects_regression():
    base = _run_with_scores({"a": 1.0, "b": 1.0, "c": 1.0}, run_id="base")
    cand = _run_with_scores({"a": 1.0, "b": 0.0, "c": 1.0}, run_id="cand")
    cmp = report.compare(base, cand, min_delta=0.02)
    assert cmp.has_regression
    assert cmp.dataset_matches
    ids = [r.case_id for r in cmp.case_regressions]
    assert ids == ["b"]  # only b dropped


def test_compare_no_regression_on_improvement():
    base = _run_with_scores({"a": 0.0, "b": 0.0}, run_id="base")
    cand = _run_with_scores({"a": 1.0, "b": 1.0}, run_id="cand")
    cmp = report.compare(base, cand)
    assert not cmp.has_regression


def test_gate_threshold_and_baseline():
    cand = _run_with_scores({"a": 1.0, "b": 0.0}, run_id="cand")  # mean 0.5
    assert report.gate(cand, thresholds={"acc": 0.4}).passed
    assert not report.gate(cand, thresholds={"acc": 0.9}).passed

    base = _run_with_scores({"a": 1.0, "b": 1.0}, run_id="base")  # mean 1.0
    res = report.gate(cand, baseline=base, min_delta=0.02)
    assert not res.passed  # 0.5 vs 1.0 is a regression


def test_gate_max_error_rate():
    from assay.types import CaseResult

    r = RunResult(
        run_id="e",
        system="s",
        dataset="d",
        results=[
            CaseResult(case=Case(id="a"), prediction=Prediction(error="x")),
            CaseResult(case=Case(id="b"), prediction=ok("y")),
        ],
    )
    assert not report.gate(r, max_error_rate=0.0).passed
    assert report.gate(r, max_error_rate=0.6).passed


# --- LLM judge (fake client) ----------------------------------------------


class _FakeParse:
    """Stand-in for client.messages.parse returning a canned structured verdict."""

    def __init__(self, verdict, usage=None):
        self._verdict = verdict
        self._usage = usage or SimpleNamespace(input_tokens=100, output_tokens=50)

    def parse(self, **kwargs):
        return SimpleNamespace(parsed_output=self._verdict, usage=self._usage)


class _FakeClient:
    def __init__(self, verdict):
        self.messages = _FakeParse(verdict)


@pytest.fixture(autouse=True)
def _reset_judge_client():
    judge_mod._shared_client = None
    yield
    judge_mod._shared_client = None


def test_llm_judge_normalizes_and_costs(monkeypatch):
    verdict = judge_mod._Verdict(reasoning="clear and faithful", score=4, meets_bar=True)
    judge_mod._shared_client = _FakeClient(verdict)

    rubric = Rubric(name="quality", criteria="Is it faithful?", scale_min=1, scale_max=5)
    j = LLMJudge(rubric)
    s = j.score(make_case(input={"q": "?"}), ok({"answer": "yes"}))
    # 4 on a 1..5 scale → (4-1)/(5-1) = 0.75
    assert s.value == pytest.approx(0.75)
    assert s.passed
    assert s.label == "4/5"
    assert s.metadata["judge_cost_usd"] == pytest.approx((100 * 5 + 50 * 25) / 1e6)


def test_llm_judge_below_bar():
    verdict = judge_mod._Verdict(reasoning="hallucinated a figure", score=2, meets_bar=False)
    judge_mod._shared_client = _FakeClient(verdict)
    rubric = Rubric(criteria="faithful?", pass_at=0.6)
    s = LLMJudge(rubric).score(make_case(), ok("x"))
    assert s.value == pytest.approx(0.25) and not s.passed


def test_llm_judge_errored_prediction_short_circuits():
    # No client set: an errored prediction must not attempt a model call.
    judge_mod._shared_client = None
    s = LLMJudge(Rubric(criteria="x")).score(make_case(), Prediction(error="boom"))
    assert s.value == 0.0 and s.passed is False
