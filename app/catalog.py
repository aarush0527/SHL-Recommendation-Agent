"""
Loads the normalized catalog once and exposes lookup helpers. Every other
module (retrieval, router, the final whitelist check) goes through this
-- there should be exactly one in-memory representation of "what's a real
catalog item," so a hallucinated name/URL has exactly one place to be
checked against.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


class Catalog:
    def __init__(self, items: list[dict]):
        self.items = items
        self.by_url = {item["url"]: item for item in items}
        self.by_entity_id = {item["entity_id"]: item for item in items}
        # normalized-name index for fuzzy compare-by-name lookups
        self._by_norm_name = {self._normalize(item["name"]): item for item in items}

    @staticmethod
    def _normalize(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    def get_by_url(self, url: str) -> dict | None:
        return self.by_url.get(url)

    def find_by_name(self, query_name: str, max_candidates: int = 3) -> list[dict]:
        """Returns ranked candidate matches for a `compare`-style name
        reference (e.g. "GSA", "OPQ"). Deliberately returns a *list*,
        not a single guess: SHL product families (OPQ32r, OPQ User
        Report, OPQ Leadership Report, ...) genuinely share abbreviations,
        and picking one silently risks grounding a comparison in the
        wrong product. The caller (router/generation) gets the ranked
        candidates plus their descriptions and can pick contextually, or
        ask a one-line disambiguating question if it's still unclear.

        Earlier version did plain substring matching on the fully
        concatenated normalized name, which is wrong for acronyms in a
        subtle way: "gsa" is not even a substring of
        "globalskillsassessment" (no word boundaries -> "l"+"s" where an
        acronym needs "g" immediately before "s"), while it accidentally
        WAS a substring of an unrelated product name where "...writin-G"
        ran into "SA-les..." across a word boundary. Word-aware acronym
        and token matching avoids both failure modes.
        """
        norm_query = self._normalize(query_name)
        if not norm_query:
            return []
        if norm_query in self._by_norm_name:
            return [self._by_norm_name[norm_query]]

        query_lower = query_name.lower().strip()
        stopwords = {"new", "the", "a", "of", "and", "for", "&"}
        scored = []
        for item in self.items:
            words = re.findall(r"[A-Za-z0-9]+", item["name"])
            significant = [w for w in words if w.lower() not in stopwords]
            if not significant:
                continue
            acronym = "".join(w[0] for w in significant).lower()
            name_lower = item["name"].lower()

            matched_token = None
            if any(w.lower() == query_lower for w in words):
                base_score = 50
                matched_token = next(w for w in words if w.lower() == query_lower)
            elif acronym == norm_query and len(norm_query) >= 2:
                base_score = 45
            elif len(query_lower) >= 3 and any(w.lower().startswith(query_lower) for w in words):
                base_score = 35
                matched_token = next(w for w in words if w.lower().startswith(query_lower))
            elif len(norm_query) >= 5 and norm_query in self._normalize(" ".join(significant)):
                base_score = 10
            else:
                continue

            score = base_score
            # A matched token carrying digits (OPQ32r, GSA1, Verify G+ "G+")
            # is usually the canonical product code in SHL's naming, not a
            # derived report -- worth more than exact-vs-prefix token match
            # itself, which is what lets "OPQ" resolve to the base OPQ32r
            # questionnaire instead of "OPQ User Report" (see conversation 5,
            # which asks to compare bare "OPQ" against "OPQ MQ Sales Report").
            if matched_token and any(c.isdigit() for c in matched_token):
                score += 25
            if "report" in name_lower:
                score -= 15
            if "profile" in name_lower:
                score -= 8

            scored.append((score, -len(item["name"]), item))  # shorter name breaks remaining ties

        scored.sort(key=lambda t: (-t[0], -t[1]))
        return [item for _, _, item in scored[:max_candidates]]

    def find_best_match(self, query_name: str) -> dict | None:
        """Convenience wrapper for when the caller just needs one item
        and is prepared to accept the top-ranked candidate (used by
        retrieval sanity checks / tests, not by the compare behavior
        itself, which should see all candidates)."""
        candidates = self.find_by_name(query_name)
        return candidates[0] if candidates else None

    def is_real(self, name: str, url: str) -> bool:
        """The whitelist check every outgoing recommendation must pass.
        URL is authoritative (names can legitimately vary in casing/
        punctuation); name is checked too as a cheap defense against a
        URL that matches by coincidence but a swapped/hallucinated name."""
        item = self.by_url.get(url)
        return item is not None and self._normalize(item["name"]) == self._normalize(name)

    def __len__(self):
        return len(self.items)


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    data = json.loads((DATA_DIR / "catalog.json").read_text(encoding="utf-8"))
    return Catalog(data)
