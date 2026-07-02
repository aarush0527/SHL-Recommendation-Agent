"""
Retrieval = base lexical similarity (TF-IDF cosine) + soft boosts from
whatever structured requirements the router has extracted so far.

Boosts are additive and capped, never multiplicative zeroing filters --
a hard filter on a sparse field (job_levels has only 10 values, language
has 42 and 37 items list none at all) can wipe out an otherwise-correct
result just because the requirement extraction phrased something the
catalog didn't literally use. Recall@10 punishes a false zero far more
than it punishes a slightly-imperfect ranking, so "never exclude, just
reorder" is the deliberate design here.
"""
import pickle
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from app.catalog import get_catalog

DATA_DIR = Path(__file__).parent.parent / "data"

# Boost weights, additive on top of a 0..1 cosine score. Kept small and
# named so they're one place to retune after eval results come in,
# rather than magic numbers scattered through the ranking logic.
BOOST_TEST_TYPE = 0.20
BOOST_ADAPTIVE = 0.08
BOOST_REMOTE = 0.05
BOOST_JOB_LEVEL = 0.08
BOOST_LANGUAGE = 0.05
BOOST_DURATION_FITS = 0.05
PENALTY_DURATION_EXCEEDS = 0.05


@dataclass
class Requirements:
    """What the router has extracted/merged from the conversation so
    far. Every field is optional -- absence means "no stated
    preference," not "excluded," which is what makes soft boosting over
    hard filtering possible."""
    test_types_wanted: list[str] = field(default_factory=list)   # e.g. ["P", "K"]
    job_level: str | None = None
    remote_required: bool = False
    adaptive_required: bool = False
    max_duration_minutes: int | None = None
    language: str | None = None


@lru_cache(maxsize=1)
def _load_index():
    with open(DATA_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    matrix = sparse.load_npz(DATA_DIR / "tfidf_matrix.npz")
    return vectorizer, matrix


def warm_up():
    """Forces the (lru_cache'd) index load to happen now rather than on
    the first real request -- called from main.py's startup event so a
    cold start pays this cost before /health reports ready, not during
    the first /chat call's 30s budget."""
    _load_index()


def _apply_boosts(scores: np.ndarray, items: list[dict], req: Requirements) -> np.ndarray:
    boosted = scores.copy()
    for i, item in enumerate(items):
        if req.test_types_wanted:
            item_types = set(item["test_type_codes"])
            overlap = len(item_types & set(req.test_types_wanted))
            if overlap:
                boosted[i] += BOOST_TEST_TYPE * min(overlap, 2) / 2  # cap the stacking
        if req.adaptive_required and item["adaptive_irt"]:
            boosted[i] += BOOST_ADAPTIVE
        if req.remote_required and item["remote_testing"]:
            boosted[i] += BOOST_REMOTE
        if req.job_level and req.job_level in item.get("job_levels", []):
            boosted[i] += BOOST_JOB_LEVEL
        if req.language and req.language in item.get("languages", []):
            boosted[i] += BOOST_LANGUAGE
        if req.max_duration_minutes and item.get("duration_minutes"):
            if item["duration_minutes"] <= req.max_duration_minutes:
                boosted[i] += BOOST_DURATION_FITS
            elif item["duration_minutes"] > req.max_duration_minutes * 1.5:
                boosted[i] -= PENALTY_DURATION_EXCEEDS
    return boosted


def search(query_text: str, requirements: Requirements | None = None, top_k: int = 10) -> list[dict]:
    """Returns up to top_k catalog items, ranked. Each result item is
    the full normalized catalog record plus a `_score` field (useful for
    logging/eval, stripped before anything reaches the API response)."""
    requirements = requirements or Requirements()
    vectorizer, matrix = _load_index()
    catalog = get_catalog()

    query_vec = vectorizer.transform([query_text])
    base_scores = cosine_similarity(query_vec, matrix)[0]
    final_scores = _apply_boosts(base_scores, catalog.items, requirements)

    ranked_idx = np.argsort(-final_scores)[:top_k]
    results = []
    for idx in ranked_idx:
        item = dict(catalog.items[idx])
        item["_score"] = float(final_scores[idx])
        results.append(item)
    return results


def lookup_named_items(names: list[str]) -> dict[str, list[dict]]:
    """For `compare`: resolve each user-mentioned name against the
    catalog rather than running it through semantic search -- if the
    user says "OPQ vs GSA" we want the exact items, not the most
    similar ones. Returns {queried_name: [ranked candidates]} rather
    than picking one silently, since SHL product families genuinely
    share abbreviations (OPQ32r vs OPQ User Report vs OPQ Leadership
    Report) and forcing a single guess here risks grounding a
    comparison in the wrong product."""
    catalog = get_catalog()
    return {name: catalog.find_by_name(name) for name in names}
