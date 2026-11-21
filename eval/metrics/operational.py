"""Operational metrics: how long does an answer take?"""


def quantile(values: list[float], q: float) -> float | None:
    """Quantile with linear interpolation (q=0.5 is the median); None if empty."""
    if not values:
        return None
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


# Latency (seconds): time to answer one question | [0, +inf) | lower is better; median = typical case, p95 = slow cases
def latency_summary(values: list[float | None]) -> dict | None:
    """Median and p95 of the per-question latencies (the mean hides slow calls).

    None values (timing not available) are ignored; None if nothing is left.
    """
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"median": quantile(values, 0.5), "p95": quantile(values, 0.95)}