"""Offline eval spec — runs end to end with no API key.

A deliberately naive keyword sentiment classifier graded against a golden set
with deterministic scorers only. This is the smoke test for the framework and a
worked example of how to wire a spec:

    assay run examples/toy_spec.py

The classifier gets the "negation" case wrong on purpose ("Not good" reads as
positive to a bag-of-words), so the run shows a realistic sub-100% score and a
tag slice that isolates the weakness.
"""

from pathlib import Path

from assay.dataset import Dataset
from assay.scorers import ExactMatch, LatencyUnder, NoError
from assay.systems import CallableSystem

_POS = {"loved", "love", "great", "best", "recommend", "good", "fast", "highly"}
_NEG = {"terrible", "awful", "broke", "waste", "avoid", "bad", "not", "ignored"}

_HERE = Path(__file__).resolve().parent


def classify(inp: dict) -> str:
    words = set(inp["text"].lower().replace("!", " ").replace(".", " ").split())
    pos, neg = len(words & _POS), len(words & _NEG)
    if pos == neg:
        return "neutral"
    return "positive" if pos > neg else "negative"


system = CallableSystem(classify, name="keyword-sentiment-v1")
dataset = Dataset.from_jsonl(_HERE / "data" / "sentiment.jsonl", version="2026-07-01")
scorers = [
    NoError(),
    ExactMatch(key="output", name="label_accuracy"),
    LatencyUnder(budget_s=0.5),
]
