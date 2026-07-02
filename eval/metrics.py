"""
Pure ranking-quality metrics -- no I/O, no LLM calls, no network. Kept
separate from run_eval.py so the scoring math is easy to read, trust,
and unit-test on its own (e.g. `python3 -m eval.metrics` runs a small
self-check with hand-computed expected values -- see __main__ below).

Every function takes:
  predicted: list[str]   -- ranked item identifiers (we use catalog URLs,
                             since names can vary in punctuation/casing
                             but a URL is the one unambiguous key both
                             the catalog and the reference traces use)
  relevant:  set[str]    -- ground-truth relevant identifiers

Everything is binary relevance (an item either is or isn't in the
reference shortlist) -- SHL's reference traces don't grade items by
degree, so graded relevance (real-valued gain) isn't available. nDCG
below therefore reduces to its binary-relevance form (gain = 1 for a
hit, 0 otherwise), which is still meaningful for ranking quality: it
rewards a relevant item at rank 1 more than one at rank 8, which raw
precision/recall does not.
"""
import math

DEFAULT_K_VALUES = (5, 10)


def precision_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    top_k = predicted[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for p in top_k if p in relevant)
    return hits / len(top_k)


def recall_at_k(predicted: list[str], relevant: set[str], k: int) -> float | None:
    """None (not 0.0) when the reference shortlist is empty -- that's an
    undefined ratio, not a failure, and averaging it in as 0 would wrongly
    drag down every other conversation's score for a fixture problem."""
    if not relevant:
        return None
    top_k = predicted[:k]
    hits = sum(1 for p in top_k if p in relevant)
    return hits / len(relevant)


def f1_at_k(predicted: list[str], relevant: set[str], k: int) -> float | None:
    p = precision_at_k(predicted, relevant, k)
    r = recall_at_k(predicted, relevant, k)
    if r is None:
        return None
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def reciprocal_rank(predicted: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant item anywhere in `predicted`
    (unbounded by k -- MRR is conventionally computed over the full
    ranked list, not truncated), 0.0 if none of the relevant items
    appear at all."""
    for i, p in enumerate(predicted, start=1):
        if p in relevant:
            return 1.0 / i
    return 0.0


def average_precision_at_k(predicted: list[str], relevant: set[str], k: int) -> float | None:
    """Standard AP@k: mean of precision@i taken at each rank i<=k where
    the item at that rank is itself relevant, normalized by
    min(|relevant|, k) (the most hits achievable within the window)."""
    if not relevant:
        return None
    top_k = predicted[:k]
    hits = 0
    precision_sum = 0.0
    for i, p in enumerate(top_k, start=1):
        if p in relevant:
            hits += 1
            precision_sum += hits / i
    denom = min(len(relevant), k)
    return precision_sum / denom if denom else 0.0


def ndcg_at_k(predicted: list[str], relevant: set[str], k: int) -> float | None:
    """Binary-relevance nDCG@k (see module docstring for why gain is
    0/1). IDCG is the DCG of the best possible arrangement: min(|relevant|, k)
    hits placed at the top ranks."""
    if not relevant:
        return None
    top_k = predicted[:k]
    dcg = sum(1.0 / math.log2(i + 1) for i, p in enumerate(top_k, start=1) if p in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_ranking(predicted: list[str], relevant: set[str], k_values=DEFAULT_K_VALUES) -> dict:
    """One conversation's / one query's full metric card."""
    result = {"mrr": reciprocal_rank(predicted, relevant), "num_predicted": len(predicted),
              "num_relevant": len(relevant)}
    for k in k_values:
        result[f"precision@{k}"] = precision_at_k(predicted, relevant, k)
        result[f"recall@{k}"] = recall_at_k(predicted, relevant, k)
        result[f"f1@{k}"] = f1_at_k(predicted, relevant, k)
        result[f"ap@{k}"] = average_precision_at_k(predicted, relevant, k)
        result[f"ndcg@{k}"] = ndcg_at_k(predicted, relevant, k)
    return result


def aggregate(per_query_results: list[dict]) -> dict:
    """Mean of each metric across queries/conversations, skipping None
    entries (undefined-for-that-query) rather than treating them as 0.
    Returns {} for an empty input rather than raising."""
    if not per_query_results:
        return {}
    keys = per_query_results[0].keys()
    out = {}
    for key in keys:
        values = [r[key] for r in per_query_results if r.get(key) is not None]
        out[key] = sum(values) / len(values) if values else None
    out["num_queries"] = len(per_query_results)
    return out


if __name__ == "__main__":
    # Tiny hand-checkable self-test -- run `python3 -m eval.metrics` to
    # sanity-check the math itself before trusting it inside a bigger
    # pipeline. Not a replacement for the pytest suite a longer-lived
    # project would want; just a fast, dependency-free gut check.
    predicted = ["a", "x", "b", "y", "c"]
    relevant = {"a", "b", "c", "d"}  # 4 relevant, only 3 retrieved in top 5

    assert precision_at_k(predicted, relevant, 5) == 3 / 5
    assert recall_at_k(predicted, relevant, 5) == 3 / 4
    assert reciprocal_rank(predicted, relevant) == 1.0  # "a" hits at rank 1
    assert round(average_precision_at_k(predicted, relevant, 5), 4) == round(
        (1 / 1 + 2 / 3 + 3 / 5) / 4, 4
    )
    # nDCG: hits land at ranks 1, 3, 5 -> DCG = 1/log2(2) + 1/log2(4) + 1/log2(6).
    # Ideal case for k=5 places all min(|relevant|, 5) = 4 hits at ranks 1-4.
    expected_dcg = 1 / math.log2(2) + 1 / math.log2(4) + 1 / math.log2(6)
    ideal_hits = min(len(relevant), 5)
    expected_idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    assert abs(ndcg_at_k(predicted, relevant, 5) - expected_dcg / expected_idcg) < 1e-9

    assert recall_at_k(predicted, set(), 5) is None
    assert aggregate([]) == {}

    print("eval/metrics.py self-test: all assertions passed.")