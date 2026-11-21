"""Retrieval metrics: did the retriever bring back the right passages?

Proxy labels: a passage counts as "relevant" if it contains the ground truth
(see eval/build_relevance.py for the corpus-wide count used by Recall@K).
"""

from eval.metrics.text import contains


def retrieval_hits(passages: list[str], ground_truth: str) -> list[bool]:
    """Helper: for each retrieved passage, in rank order, is it relevant (proxy label)?"""
    return [contains(passage, ground_truth) for passage in passages]


# Precision@K: share of retrieved passages that are relevant | [0, 1] | 1 = no noise, 0 = nothing useful retrieved
def precision_at_k(hits: list[bool]) -> float:
    """Per question; the reported value is the mean over open questions."""
    return sum(hits) / len(hits) if hits else 0.0


# Reciprocal rank (its mean is the MRR): rank of the first relevant passage | [0, 1] | 1 = first, 0.5 = second, 0 = none in top-k
def reciprocal_rank(hits: list[bool]) -> float:
    """Per question; the reported value is the mean over open questions (MRR)."""
    for rank, hit in enumerate(hits, start=1):
        if hit:
            return 1.0 / rank
    return 0.0


# Recall@K: share of all relevant corpus passages found in the top-k | [0, 1] | 1 = all found, 0 = none found
def recall_at_k(hits: list[bool], n_relevant_total: int | None) -> float | None:
    """Per question; None if the corpus total is unknown or zero (question skipped).

    The best reachable value is min(k, n_relevant_total) / n_relevant_total,
    so a low recall is normal when many passages match and k is small.
    """
    if not n_relevant_total:
        return None
    return sum(hits) / n_relevant_total