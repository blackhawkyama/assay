"""Improved classifier: handle simple negation ("not good" → negative).

Same dataset and scorers as toy_spec.py, so runs are directly comparable:

    assay run     examples/toy_spec_v2.py
    assay compare runs/<v1>.json runs/<v2>.json
"""

from pathlib import Path

from assay.dataset import Dataset
from assay.scorers import ExactMatch, LatencyUnder, NoError
from assay.systems import CallableSystem

_POS = {"loved", "love", "great", "best", "recommend", "good", "fast", "highly"}
_NEG = {"terrible", "awful", "broke", "waste", "avoid", "bad", "ignored"}
_NEGATORS = {"not", "no", "never", "nothing"}

_HERE = Path(__file__).resolve().parent


def classify(inp: dict) -> str:
    words = inp["text"].lower().replace("!", " ").replace(".", " ").split()
    wset = set(words)
    pos, neg = len(wset & _POS), len(wset & _NEG)
    # A negator flips the polarity of nearby sentiment.
    if wset & _NEGATORS:
        pos, neg = neg, pos + 1
    if pos == neg:
        return "neutral"
    return "positive" if pos > neg else "negative"


system = CallableSystem(classify, name="keyword-sentiment-v2")
dataset = Dataset.from_jsonl(_HERE / "data" / "sentiment.jsonl", version="2026-07-01")
scorers = [
    NoError(),
    ExactMatch(key="output", name="label_accuracy"),
    LatencyUnder(budget_s=0.5),
]
