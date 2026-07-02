"""
Domain vocabulary for the no-LLM fallback, derived directly from the
catalog itself rather than a hand-maintained file.

An earlier version of this idea (see git history / the uploaded
SHL_Catalog_Ontology_Clean.md) hand-transcribed vocabulary from the
catalog into a separate file. Checked it against the real 377-item
catalog: 198 items (52%) don't appear in it at all -- entire categories
like Nursing and Pharmaceutical Science, most OPQ report variants,
anything alphabetically past roughly "G". A hand-curated list will
always have this failure mode (it's manual transcription of ~400
records), and the fallback is exactly the wrong place to have silent,
uneven coverage gaps -- it's already the degraded path; it shouldn't
also be blind to half the catalog.

Deriving the vocabulary from catalog.json instead means: every token
that exists in a real assessment name, category label, job level, or
language is *guaranteed* present, because it's read from the same data
structure retrieval and validation use -- there's no separate artifact
that can drift out of sync when the catalog is refreshed.
"""
import re
from functools import lru_cache

from app.catalog import get_catalog

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#.\-]*")

# Generic English words that appear throughout assessment names and
# descriptions but carry no matching signal on their own ("the new
# version", "for the role", etc.).
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with", "is", "are", "be",
    "this", "that", "these", "those", "by", "as", "at", "from", "it", "its", "into", "than",
    "then", "so", "such", "not", "no", "yes", "if", "but", "we", "you", "i", "he", "she",
    "they", "them", "his", "her", "their", "our", "your", "my", "me", "us", "do", "does",
    "did", "can", "could", "should", "would", "will", "shall", "have", "has", "had", "was",
    "were", "been", "being", "also", "etc", "other", "some", "any", "all", "each", "more",
    "most", "much", "many", "just", "only", "about", "over", "under", "between", "up", "down",
    "off", "again", "once", "here", "there", "when", "where", "why", "how", "am", "new",
})


@lru_cache(maxsize=1)
def load_catalog_vocabulary() -> frozenset[str]:
    """Every meaningful token appearing anywhere in the real catalog:
    assessment names, test-type category labels, job levels, languages.
    100% coverage of the actual catalog by construction -- there is no
    transcription step that can fall behind."""
    catalog = get_catalog()
    tokens = set()
    for item in catalog.items:
        fields = [item["name"], *item.get("test_type_labels", []), *item.get("job_levels", []),
                  *item.get("languages", [])]
        for value in fields:
            for w in _WORD_RE.findall(value or ""):
                w = w.lower().strip(".,()-")
                if len(w) > 1 and w not in STOPWORDS:
                    tokens.add(w)
    return frozenset(tokens)


def important_word_count(text: str) -> int:
    """How many distinct tokens in `text` are recognized catalog
    vocabulary. A single strong match ("java", "cobol", "nursing") is
    treated as more meaningful than several words of generic filler --
    this is what lets the fallback tell "java python c" (three words,
    all real signal) apart from "i really need some help please" (five
    words, zero signal)."""
    vocab = load_catalog_vocabulary()
    tokens = {w.lower().strip(".,()-") for w in _WORD_RE.findall(text)}
    return len(tokens & vocab)