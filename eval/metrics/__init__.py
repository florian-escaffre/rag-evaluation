"""Evaluation metrics, one module per group.

retrieval.py    Precision@K, reciprocal rank (MRR), Recall@K
generation.py   yes/no accuracy, exact match, token F1
operational.py  latency
text.py         text helpers (normalize, contains, parse_yes_no), not metrics

Everything is re-exported here, so `from eval.metrics import token_f1`
style imports work. LLM-as-judge metrics (phase 2) get their own module.
"""

from eval.metrics.generation import exact_match, token_f1, yes_no_correct
from eval.metrics.operational import latency_summary
from eval.metrics.retrieval import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_hits,
)
from eval.metrics.text import contains, normalize, parse_yes_no

__all__ = [
    "contains",
    "exact_match",
    "latency_summary",
    "normalize",
    "parse_yes_no",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "retrieval_hits",
    "token_f1",
    "yes_no_correct",
]
