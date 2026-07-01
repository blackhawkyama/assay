# assay

A small, honest LLM-evaluation framework. You shipped a change to an LLM system —
a new prompt, a new model, a reworked pipeline. **Did it get better or worse?**
`assay` answers that with numbers instead of vibes.

Built to be provider-agnostic at the core, with an [Anthropic](https://docs.claude.com)
LLM-as-judge. Wrap any system under test — a plain callable, or an external
script/binary via `SubprocessSystem`.

## The spine

```
Dataset  → versioned golden cases (inputs + optional expectations)
System   → the thing under test (a callable, a script, an HTTP API)
Scorer   → grades one prediction (deterministic checks or an LLM judge)
Runner   → runs a System over a Dataset with Scorers; captures cost + latency
Report   → aggregates, diffs against a baseline, flags regressions, gates CI
```

Everything is a Pydantic model, so runs serialize to JSON and diff in git.

## Quickstart (no API key needed)

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

assay run examples/toy_spec.py        # runs a toy classifier vs a golden set
assay show runs/<id>.json             # pretty-print any saved run
```

The toy classifier misses the negation case on purpose, so you get a realistic
sub-100% score and a tag slice that isolates the weakness.

## Grading with an LLM judge

Deterministic scorers (schema validity, numeric bounds, required substrings,
latency) are cheap and exact — reach for them first. For quality that regex
can't reach — faithfulness, tone, correctness — use the judge:

```python
from assay.scorers import LLMJudge, Rubric

judge = LLMJudge(Rubric(
    name="faithfulness",
    criteria="Every claim in the summary must be supported by the source. "
             "Penalize invented figures, dates, or entities.",
    scale_min=1, scale_max=5, pass_at=0.6,
))
```

Two guards against the usual LLM-judge failure modes: verdicts are **structured
output** (no prose to parse), and `PairwiseJudge` runs **both orderings** to
cancel position bias. Judge model defaults to Claude Opus 4.8; token cost lands
on each score's metadata so the eval's own bill is visible.

Set credentials the usual Anthropic way (`ANTHROPIC_API_KEY`, or `ant auth login`).

## Comparing runs and gating CI

```bash
assay compare runs/<baseline>.json runs/<candidate>.json
```

```
  scorer                        base    cand        Δ
  ----------------------------  ------  ------  ---------
  label_accuracy                0.833   0.667   -0.167  ↓
  Regressed cases (1):
    neg-3   label_accuracy   1.00 → 0.00   expected 'negative', got 'positive'
```

Gate a run in CI (exit 1 on failure):

```bash
assay gate runs/<candidate>.json \
  --threshold label_accuracy=0.8 \
  --max-error-rate 0.0 \
  --baseline runs/<known-good>.json      # also fail on any regression
```

## Writing an eval spec

A spec is a Python file exposing `system`, `dataset`, `scorers` (or a `build()`
returning them). See [`examples/toy_spec.py`](examples/toy_spec.py).

```python
from assay.dataset import Dataset
from assay.systems import CallableSystem
from assay.scorers import NoError, ExactMatch, LatencyUnder

system  = CallableSystem(my_fn, name="my-system-v1")
dataset = Dataset.from_jsonl("data/cases.jsonl", version="2026-07-01")
scorers = [NoError(), ExactMatch(key="output"), LatencyUnder(2.0)]
```

## Layout

```
assay/            core framework
  types.py          Case / Prediction / Score / CaseResult / RunResult
  dataset.py        versioned golden datasets (JSONL / YAML)
  systems.py        Callable / Subprocess systems under test
  scorers/          deterministic checks + LLM judge
  runner.py         concurrent run + artifact persistence
  report.py         format / compare / gate
  cli.py            assay run | show | compare | gate
examples/         offline toy spec + data
tests/            pytest suite (judge covered with a fake client)
```

## Status

v0.1 — core spine, deterministic + judge scorers, compare/gate, offline example,
tested. Next: a GitHub Actions gate, more built-in scorers, and additional
worked-example adapters.
