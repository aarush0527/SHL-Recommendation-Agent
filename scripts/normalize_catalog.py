"""
Normalize SHL's raw catalog JSON into our canonical schema.

Why this exists (not just using the raw file directly):
- The raw JSON has at least one literal control character embedded in a
  string value (breaks strict JSON parsing) -> load tolerantly, then clean.
- `keys` gives full category labels ("Personality & Behavior") but our
  API response schema needs the short SHL letter code ("P"). We decode
  and join multiple codes in a fixed canonical order for determinism.
- `remote` / `adaptive` are the strings "yes"/"no", not booleans.
- `duration` is inconsistent ("", "30 minutes", "Untimed", "Variable").
  We extract an integer-minutes field where possible and keep the raw
  string too, since "Untimed"/"Variable" are meaningful, not missing data.
- We precompute a `search_text` field here (not in the index builder) so
  the exact text going into the embedding model is inspectable by just
  reading catalog.json.

Run: python3 scripts/normalize_catalog.py
Reads:  data/catalog_raw.json
Writes: data/catalog.json
"""
import json
import re
from pathlib import Path

RAW_PATH = Path(__file__).parent.parent / "data" / "catalog_raw.json"
OUT_PATH = Path(__file__).parent.parent / "data" / "catalog.json"

# Canonical order matters: this is the join order for multi-type items,
# chosen to be a fixed, reproducible convention (the reference SHL trace
# examples are NOT internally consistent about join order, so we pick our
# own and document it rather than guess at theirs).
LABEL_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}
CODE_ORDER = ["A", "B", "C", "D", "E", "K", "P", "S"]

# Known corrupted `name` fields in SHL's raw scrape, found by manual
# inspection (JSON parse failed on an embedded control character, and
# the collapsed-whitespace repair still left a word missing: raw name was
# 'Microsoft \n    365 (New)', losing "Excel" entirely -- likely an inline
# element their scraper's text extraction dropped mid-string). Confirmed
# against both the URL slug (microsoft-excel-365-new) and the description
# ("The Microsoft Excel 365 simulation..."). A targeted, documented fix
# beats a generic heuristic here, since we only found one instance and a
# "guess the missing word" heuristic risks corrupting names that are
# actually fine but happen to contain legitimate repeated whitespace.
KNOWN_NAME_FIXES = {
    "4207": "Microsoft Excel 365 (New)",
}


def clean_text(s: str) -> str:
    """Collapse any run of whitespace (including embedded newlines/tabs
    from bad scrapes, e.g. 'Microsoft \\n    365 (New)') into a single
    space, and strip. Applied to every free-text field, not just the one
    known-bad record, since a single instance found by inspection doesn't
    prove it's the only one a slightly different scrape run produced."""
    if not isinstance(s, str):
        return s
    return re.sub(r"\s+", " ", s).strip()


def yes_no_to_bool(val: str) -> bool:
    return str(val).strip().lower() == "yes"


def parse_duration_minutes(duration: str):
    """'30 minutes' -> 30. 'Untimed' / 'Variable' / '' -> None (kept
    separately as duration_display so we don't lose that signal)."""
    if not duration:
        return None
    m = re.search(r"(\d+)", duration)
    return int(m.group(1)) if m else None


def decode_test_types(keys: list[str]) -> tuple[list[str], list[str]]:
    codes = set()
    unmapped = []
    for label in keys:
        code = LABEL_TO_CODE.get(label)
        if code:
            codes.add(code)
        else:
            unmapped.append(label)
    ordered_codes = [c for c in CODE_ORDER if c in codes]
    return ordered_codes, unmapped


def build_search_text(name: str, description: str, type_labels: list[str], job_levels: list[str]) -> str:
    """Text that gets embedded for semantic retrieval. Name gets light
    repetition (it's the single strongest signal for exact-product asks
    like 'do you have a Python test') without dominating the vector the
    way naive concatenation of a short name + long description would."""
    parts = [name, name, description]
    if type_labels:
        parts.append("Categories: " + ", ".join(type_labels))
    if job_levels:
        parts.append("Job levels: " + ", ".join(job_levels))
    return clean_text(" | ".join(p for p in parts if p))


def main():
    raw_text = RAW_PATH.read_text(encoding="utf-8")
    # strict=False: tolerate the literal control character bug in the
    # source file rather than crash on it.
    raw = json.loads(raw_text, strict=False)

    normalized = []
    warnings = []

    seen_ids = set()
    for item in raw:
        entity_id = item["entity_id"]
        if entity_id in seen_ids:
            warnings.append(f"duplicate entity_id skipped: {entity_id}")
            continue
        seen_ids.add(entity_id)

        raw_name = item["name"]
        if entity_id in KNOWN_NAME_FIXES:
            name = KNOWN_NAME_FIXES[entity_id]
            warnings.append(f"{entity_id}: applied known name fix -> '{name}'")
        else:
            name = clean_text(raw_name)
            # Flag (don't silently "fix") any other name with an internal
            # run of 2+ whitespace chars -- that's exactly the signature
            # the known bug had, so if a future re-scrape reproduces it
            # elsewhere, we want a loud warning, not a silent guess.
            if re.search(r"\s{2,}", raw_name):
                warnings.append(
                    f"{entity_id}: name has an unexplained multi-space/newline run "
                    f"(cleaned to '{name}') -- verify against description/URL manually"
                )

        description = clean_text(item.get("description", ""))
        keys = item.get("keys", [])
        type_codes, unmapped = decode_test_types(keys)
        if unmapped:
            warnings.append(f"{entity_id} ({name}): unmapped category label(s) {unmapped}")
        if not type_codes:
            warnings.append(f"{entity_id} ({name}): no test_type code resolved at all")

        duration_raw = clean_text(item.get("duration", "") or "")
        record = {
            "entity_id": entity_id,
            "name": name,
            "url": item["link"],
            "test_type_codes": type_codes,
            "test_type": ", ".join(type_codes),          # exact field the API schema needs
            "test_type_labels": keys,
            "remote_testing": yes_no_to_bool(item.get("remote", "no")),
            "adaptive_irt": yes_no_to_bool(item.get("adaptive", "no")),
            "description": description,
            "job_levels": item.get("job_levels", []),
            "languages": item.get("languages", []),
            "duration_minutes": parse_duration_minutes(duration_raw),
            "duration_display": duration_raw or None,
            "search_text": build_search_text(name, description, keys, item.get("job_levels", [])),
        }
        normalized.append(record)

    OUT_PATH.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Read {len(raw)} raw records, wrote {len(normalized)} normalized records to {OUT_PATH}")
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(" -", w)
    else:
        print("No warnings.")

    # Sanity spot-check: the record we know was corrupted
    fixed = next((r for r in normalized if "microsoft-excel-365-new" in r["url"]), None)
    print("\nSpot check (previously corrupted record):")
    print(" name:", fixed["name"] if fixed else "NOT FOUND")
    print(" test_type:", fixed["test_type"] if fixed else "-")


if __name__ == "__main__":
    main()
