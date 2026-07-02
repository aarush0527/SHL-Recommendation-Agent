"""
Two independent layers, deliberately not merged into one:

1. Heuristic pre-check (this module) -- fast, free, catches the blunt
   cases ("ignore your instructions", "what's the weather") before an
   LLM call is even made. Runs first so an obvious off-topic message
   never has to wait on a model round-trip.

2. LLM-based scope classification (in the extraction prompt, not here)
   -- catches subtler cases the heuristics miss (a legal question phrased
   politely, an injection attempt with no trigger phrases).

Neither layer alone is trustworthy: heuristics both over-fire (a
legitimate question happens to contain "ignore") and under-fire (a
clever rephrasing dodges every pattern); an LLM classifier can be
argued with inside its own context window. Running both, and requiring
only one to flag a message as needing refusal, is the point.

3. Whitelist validation (also here) -- the very last step before a
   response leaves the service. Every recommended {name, url} must
   resolve to a real catalog entry. This should never fire if the rest
   of the pipeline is built correctly; it exists as the safety net for
   the day something upstream has a bug, not as the primary defense.
"""
import re

from app.catalog import get_catalog
from app.schemas import Recommendation

# Deliberately conservative and small. These exist to catch unambiguous
# cases cheaply, not to be a complete injection/off-topic classifier --
# that job belongs to the LLM extraction step, which has actual
# language understanding these regexes don't.
INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+|your\s+)?(previous|prior|above|earlier)\s+instructions\b",
    r"\byou\s+are\s+now\b",
    r"\bnew\s+system\s+prompt\b",
    r"\b(show|print|tell|give)\s+me\s+(your\s+)?(the\s+)?(system\s+)?(prompt|instructions)\b",
    r"\breveal\s+(your\s+)?(system\s+)?(prompt|instructions)\b",
    r"\bwhat\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions)\b",
    r"\bdisregard\s+(all\s+|your\s+)?(previous|prior)\b",
    r"\bact\s+as\s+(if\s+you\s+are\s+)?(dan|an?\s+unrestricted|.*\bwithout\s+(any\s+)?(restrictions|filters|guidelines)\b)",
    r"\bdeveloper\s+mode\b",
    r"\bjailbreak\b",
]

OFF_TOPIC_PATTERNS = [
    r"\bwrite\s+(me\s+)?(a\s+)?(poem|song|story|essay)\b",
    r"\bwhat'?s?\s+the\s+weather\b",
    r"\btell\s+me\s+a\s+joke\b",
    r"\b(stock\s+price|cryptocurrency|bitcoin)\b",
    r"\bhow\s+do\s+i\s+(cook|bake)\b",
]

LEGAL_HIRING_ADVICE_PATTERNS = [
    r"\b(legally|legal(ly)?\s+required|law\s+requires?)\b.*\b(fire|terminate|discriminat|hipaa|ada\b|eeoc)\b",
    r"\bis\s+it\s+legal\s+to\b",
    r"\b(should|can|could|may)\s+i\s+(fire|terminate|let\s+go|discriminat|reject)\b",
    r"\bhow\s+(do|should)\s+i\s+(fire|terminate|discipline)\s+(an?\s+)?employee\b",
    r"\bgeneral\s+hiring\s+advice\b",
]


def heuristic_scope_check(text: str) -> str | None:
    """Returns a short reason string if the heuristics flag this message,
    else None. Cheap regex pass -- not a final verdict, just a fast
    first opinion that skips a model call for the obvious cases."""
    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return "injection_attempt"
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, lowered):
            return "off_topic"
    for pattern in LEGAL_HIRING_ADVICE_PATTERNS:
        if re.search(pattern, lowered):
            return "legal_or_general_hiring_advice"
    return None


def validate_recommendations(recommendations: list[Recommendation]) -> list[Recommendation]:
    """The hallucination safety net. Drops (does not "fix") any item
    whose name+url don't match a real catalog entry exactly -- silently
    correcting a hallucinated item to the "closest" real one would hide
    a bug rather than surface it, and could itself present a wrong
    answer with false confidence."""
    catalog = get_catalog()
    valid = []
    for rec in recommendations:
        if catalog.is_real(rec.name, rec.url):
            valid.append(rec)
        else:
            # In production this should log/alert loudly -- reaching
            # here means the generation step produced something not in
            # our own retrieved candidates, which should be structurally
            # impossible if router.py only ever builds recommendations
            # from catalog rows. Treated as a bug signal, not routine.
            print(f"[guardrails] DROPPED non-catalog recommendation: {rec.name!r} / {rec.url!r}")
    return valid
