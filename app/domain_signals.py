"""
Hand-crafted domain-signal layer for the no-LLM fallback.

app/vocabulary.py answers "does this text mention something literally in
the catalog" (guaranteed complete, derived from data). This module
answers a different question: "what does this text imply, in ways that
don't literally appear in catalog text" -- a recruiter says "call
center", not "Biodata & Situational Judgment"; "fresher" or "just
graduated", not "Graduate"; "need it done in 15 minutes", not
"duration: 15 minutes". That gap is exactly what a real LLM's language
understanding closes and a keyword vocabulary can't -- so this is a
deliberately hand-curated (not catalog-derived) set of the common
recruiting phrasings mapped to structured signals, acting as a cheap
approximation of that understanding when no LLM is available.

This is the same idea as a "DOMAIN_SIGNALS dictionary" pattern seen in
similar SHL-recommender projects -- worth borrowing the *pattern*, but
built fresh against our own full 377-item catalog rather than adopted
from a reference that only covered ~50 items itself. Kept intentionally
scoped: it is not trying to enumerate the catalog (vocabulary.py already
does that with full coverage), only the layer of human phrasing that
sits on top of it.
"""
import re

# ----------------------------------------------------------------------
# Test-type signals: phrase -> SHL letter code(s).
# Legend: A=Ability & Aptitude, B=Biodata & Situational Judgment,
# C=Competencies, D=Development & 360, E=Assessment Exercises,
# K=Knowledge & Skills, P=Personality & Behavior, S=Simulations
# ----------------------------------------------------------------------
TEST_TYPE_SIGNALS: dict[str, list[str]] = {
    "P": ["personality", "behavioural", "behavioral", "opq", "temperament", "work style", "traits"],
    "A": ["cognitive", "aptitude", "numerical reasoning", "verbal reasoning", "abstract reasoning",
          "inductive reasoning", "logical reasoning", "ability test", "general ability", "iq test"],
    "K": ["technical test", "knowledge test", "coding test", "programming test", "skills test",
          "technical assessment", "coding assessment", "language test"],
    "S": ["simulation", "simulated", "role play", "roleplay", "in-tray", "in tray"],
    "B": ["situational judgement", "situational judgment", "sjt", "biodata", "work sample"],
    "D": ["360", "development report", "multi-rater", "multi rater"],
    "C": ["competency", "competencies", "competency framework"],
    "E": ["assessment centre", "assessment center", "development centre", "development center"],
}

# ----------------------------------------------------------------------
# Job-level signals: phrase -> exact catalog job_levels vocabulary value.
# Ordered so more specific phrases are checked before generic ones.
# ----------------------------------------------------------------------
JOB_LEVEL_SIGNALS: list[tuple[str, str]] = [
    ("fresh graduate", "Graduate"), ("fresher", "Graduate"), ("new grad", "Graduate"),
    ("recent graduate", "Graduate"), ("graduate", "Graduate"), ("campus hire", "Graduate"),
    ("entry level", "Entry-Level"), ("entry-level", "Entry-Level"), ("junior", "Entry-Level"),
    ("intern", "Entry-Level"),
    ("senior leadership", "Executive"), ("executive leadership", "Executive"),
    ("front line manager", "Front Line Manager"), ("frontline manager", "Front Line Manager"),
    ("team lead", "Front Line Manager"), ("team leader", "Front Line Manager"),
    ("supervisor", "Supervisor"), ("supervisory", "Supervisor"),
    ("director", "Director"), ("vp", "Director"), ("vice president", "Director"),
    ("executive", "Executive"), ("c-suite", "Executive"), ("chief", "Executive"),
    ("senior manager", "Manager"), ("manager", "Manager"), ("managerial", "Manager"),
    ("mid-level", "Mid-Professional"), ("mid level", "Mid-Professional"),
    ("experienced", "Mid-Professional"), ("senior", "Mid-Professional"),
    ("individual contributor", "Professional Individual Contributor"),
    ("ic role", "Professional Individual Contributor"),
]

# ----------------------------------------------------------------------
# Role-family signals: phrase -> extra terms injected into the search
# query (nudges retrieval toward the right part of the catalog even
# when the user's own words don't literally match assessment names).
# ----------------------------------------------------------------------
ROLE_FAMILY_SIGNALS: list[tuple[str, str]] = [
    ("call center", "customer service contact center phone simulation"),
    ("call centre", "customer service contact center phone simulation"),
    ("contact center", "customer service contact center phone simulation"),
    ("contact centre", "customer service contact center phone simulation"),
    ("customer service", "customer service simulation situational judgment"),
    ("customer support", "customer service simulation situational judgment"),
    ("sales", "sales solution behavioral personality"),
    ("account manager", "sales competencies personality"),
    ("finance", "finance accounting numerical reasoning"),
    ("accounting", "accounts payable receivable finance"),
    ("financial analyst", "finance numerical reasoning knowledge"),
    ("human resources", "HR competencies personality"),
    ("hr executive", "HR competencies personality"),
    ("hr manager", "HR competencies personality leadership"),
    ("leadership", "leadership OPQ development 360"),
    ("senior leadership", "leadership OPQ enterprise development"),
    ("executive", "leadership OPQ enterprise development"),
    ("administrative", "data entry administrative clerical"),
    ("admin assistant", "data entry administrative Excel Word"),
    ("clerical", "data entry administrative"),
    ("data entry", "data entry alphanumeric numeric"),
    ("technical support", "technical support desktop knowledge"),
    ("desktop support", "technical support desktop knowledge"),
    ("software engineer", "programming knowledge coding"),
    ("software developer", "programming knowledge coding"),
    ("backend", "programming knowledge coding"),
    ("frontend", "programming knowledge coding web"),
    ("full stack", "programming knowledge coding web"),
    ("devops", "cloud computing docker knowledge"),
    ("data scientist", "data science machine learning knowledge"),
    ("data analyst", "data science numerical reasoning knowledge"),
    ("manufacturing", "manufacturing industrial safety dependability"),
    ("warehouse", "manufacturing industrial safety dependability"),
    ("driver", "safety dependability"),
    ("nurse", "nursing healthcare knowledge"),
    ("nursing", "nursing healthcare knowledge"),
    ("pharma", "pharmaceutical science knowledge"),
    ("engineer", "engineering knowledge technical"),
]

# ----------------------------------------------------------------------
# Duration constraint: "under 20 minutes", "less than 30 min", "within
# 15 minutes", "no more than 10 minutes" -> integer minutes ceiling.
# ----------------------------------------------------------------------
_DURATION_RE = re.compile(
    r"(?:under|less than|no more than|within|max(?:imum)?(?:\s+of)?|shorter than)\s+(\d{1,3})\s*(?:min(?:ute)?s?)",
    re.IGNORECASE,
)


def extract_duration_ceiling(text: str) -> int | None:
    m = _DURATION_RE.search(text)
    return int(m.group(1)) if m else None


# ----------------------------------------------------------------------
# Negation-aware exclusion: "remove the personality test", "drop OPQ",
# "no longer need cognitive", "without a simulation" -> types to EXCLUDE
# rather than include, even though the type keyword itself is present.
# ----------------------------------------------------------------------
_NEGATION_WORDS = r"(?:remove|drop|exclude|without|no longer need|don'?t need|not interested in|skip)"
_NEGATION_RE = re.compile(_NEGATION_WORDS + r"\b[^.?!]{0,40}", re.IGNORECASE)


def extract_negated_types(text: str) -> set[str]:
    """Returns type codes explicitly being removed, so the caller can
    subtract them from whatever positive signals matched -- "remove the
    personality test" should not still boost P just because the word
    "personality" is present somewhere in the sentence."""
    negated = set()
    for m in _NEGATION_RE.finditer(text):
        window = m.group(0)
        for code, phrases in TEST_TYPE_SIGNALS.items():
            if any(p in window.lower() for p in phrases):
                negated.add(code)
    return negated


# ----------------------------------------------------------------------
# Closing-signal detection: plain acceptance/confirmation with nothing
# further pending.
# ----------------------------------------------------------------------
_CLOSING_PHRASES = [
    "thanks", "thank you", "perfect", "great, that works", "that works", "sounds good",
    "confirmed", "looks good", "that's all", "that is all", "all set", "good to go",
    "works for me", "appreciate it", "that'll do", "that will do",
]


def is_closing_signal(text: str) -> bool:
    lowered = text.lower().strip()
    if len(lowered) > 80:  # a long message is very unlikely to be *just* a closing remark
        return False
    return any(p in lowered for p in _CLOSING_PHRASES)


# ----------------------------------------------------------------------
# Compare-target extraction: "compare X and Y", "X vs Y", "difference
# between X and Y". Anchored to an explicit comparison trigger so a
# normal recommend request ("Java and Python developer") never gets
# misread as a comparison.
# ----------------------------------------------------------------------
_COMPARE_PATTERNS = [
    re.compile(r"compar(?:e|ing)\s+(.+?)\s+(?:and|vs\.?|versus|with|to)\s+(.+?)(?:\.(?!\d)|[?!]|$)", re.IGNORECASE),
    re.compile(r"difference(?:s)?\s+between\s+(.+?)\s+and\s+(.+?)(?:\.(?!\d)|[?!]|$)", re.IGNORECASE),
    re.compile(r"(.{2,40}?)\s+(?:vs\.?|versus)\s+(.{2,40}?)(?:\.(?!\d)|[?!]|$)", re.IGNORECASE),
    re.compile(r"is\s+(.+?)\s+(?:better|different)\s+(?:than|from)\s+(.+?)(?:\.(?!\d)|[?!]|$)", re.IGNORECASE),
]
_MAX_NAME_SPAN = 60  # a "name" longer than this is almost certainly a mis-extraction, not a product name

# If a captured span runs on past the product name into a qualifying
# clause ("Python which is better for backend"), cut it at the first of
# these -- they essentially never appear inside a real SHL product name,
# but commonly start the next clause of a question.
_CONTINUATION_STOPWORDS = re.compile(
    r"\s+\b(which|who|that|is|are|for|in|on|with|do|does|would|should|can)\b.*$", re.IGNORECASE
)


def _clean_span(span: str) -> str:
    span = span.strip(" ?.!")
    span = _CONTINUATION_STOPWORDS.sub("", span)
    return span.strip(" ?.!")


def extract_compare_targets(text: str) -> list[str]:
    for pattern in _COMPARE_PATTERNS:
        m = pattern.search(text)
        if m:
            a, b = _clean_span(m.group(1)), _clean_span(m.group(2))
            if 0 < len(a) <= _MAX_NAME_SPAN and 0 < len(b) <= _MAX_NAME_SPAN:
                return [a, b]
    return []


# ----------------------------------------------------------------------
# Combined extraction used by the fallback router.
# ----------------------------------------------------------------------
def extract_signals(text: str) -> dict:
    lowered = text.lower()
    positive_types = {code for code, phrases in TEST_TYPE_SIGNALS.items() if any(p in lowered for p in phrases)}
    negated_types = extract_negated_types(text)
    type_codes = sorted(positive_types - negated_types)

    job_level = None
    for phrase, level in JOB_LEVEL_SIGNALS:
        if phrase in lowered:
            job_level = level
            break

    extra_terms = [terms for phrase, terms in ROLE_FAMILY_SIGNALS if phrase in lowered]

    return {
        "test_types_wanted": type_codes,
        "job_level": job_level,
        "extra_search_terms": " ".join(extra_terms),
        "max_duration_minutes": extract_duration_ceiling(text),
        "compare_targets": extract_compare_targets(text),
        "is_closing_signal": is_closing_signal(text),
    }