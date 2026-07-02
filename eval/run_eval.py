"""
eval/run_eval.py -- quantitative evaluation harness for the SHL
Recommendation Agent, built to satisfy the assignment's second
requirement: "evaluation methods to measure retrieval quality,
recommendation relevance, groundedness, and overall response accuracy
and effectiveness." It complements run_tests2.py rather than
replacing it: run_tests2.py is a *conversational behavior* suite (does
the assistant clarify/refuse/compare/refine correctly across 50
hand-written scenarios, graded PASS/PARTIAL/FAIL by an LLM judge with
no numeric answer key). This script instead measures *how good the
recommendations themselves are*, scored against the 10 reference
conversations already parsed into eval/traces/ by
scripts/parse_traces.py -- each one carries a real, human-authored
"expected_shortlist" (SHL's own reference answer) to score against.

Five things get measured, one per named requirement (D covers
groundedness alongside C; the other three map one-to-one):

  A. RETRIEVAL QUALITY (offline -- no server, no GROQ_API_KEY needed)
     Calls app.retrieval.search() directly with each conversation's
     fully-specified request (every persona_facts turn joined into one
     query) and scores the ranked candidates against the reference
     shortlist with standard IR metrics: Precision/Recall/F1@k, MRR,
     MAP@k, nDCG@k (see eval/metrics.py). This isolates the TF-IDF +
     boost retriever's own ranking quality from the LLM selection step
     downstream of it -- a regression here means the *retriever*
     broke, not the prompt.

  B. END-TO-END RECOMMENDATION QUALITY (needs the live /chat server)
     Replays each conversation's real user turns against POST /chat
     one message at a time -- exactly the sequence a real user would
     type, including later refinements/removals -- and scores the
     FINAL turn's `recommendations` against the same reference
     shortlist with the same metrics. This is retrieval + LLM
     candidate-selection + multi-turn state management combined: what
     a user actually receives.

  C. DETERMINISTIC GROUNDEDNESS
     Every recommendation object returned across every turn of every
     replayed conversation is independently re-checked against
     app.catalog.Catalog.is_real(): does {name, url} resolve to a real
     catalog row, and does the reported test_type match that row
     exactly? app/guardrails.py already runs this exact check before a
     response leaves the service -- this re-runs it here as an
     external measurement of the live API's actual output, not a
     self-report, so a regression that bypassed guardrails would still
     be caught.

  D. LLM-JUDGED GROUNDEDNESS / FAITHFULNESS
     C only covers the *structured* recommendation objects, which are
     assembled in Python from real catalog rows and structurally can't
     hallucinate a name/URL (see prompts/generate.py's docstring). It
     says nothing about the free-text `reply` -- a model can correctly
     recommend a real assessment and still invent a false property of
     it in prose ("this one only takes 15 minutes" when the catalog
     says 45). This step feeds each final reply, plus the real catalog
     facts for its recommended items, to an LLM judge and asks it to
     flag any specific factual claim that isn't supported by those
     facts.

  E. LLM-JUDGED RECOMMENDATION RELEVANCE
     The reference shortlist is *a* good answer, not the *only*
     acceptable one -- a valid alternative assessment the system finds
     that SHL's reference simply didn't list would be unfairly
     punished by A/B's set-overlap metrics alone. This step asks an
     LLM judge to rate each recommended item's relevance to the
     conversation's stated requirements directly, independent of the
     reference list, giving a precision-like signal that doesn't
     depend on the reference being exhaustive.

Results are combined into one effectiveness summary (see
`build_overall_summary`) and everything -- including full
per-conversation detail -- is written to eval/results/ as both JSON
(machine-readable) and Markdown (for a human reviewer). Console output
is a compact summary only.

USAGE
  # 1. Retrieval quality only -- offline, no server, no GROQ_API_KEY.
  python3 eval/run_eval.py --retrieval-only

  # 2. Full run -- needs the server running in another terminal:
  export GROQ_API_KEY=your_key
  uvicorn app.main:app --port 8000 &
  python3 eval/run_eval.py           # GROQ_API_KEY must ALSO be set
                                      # in *this* shell -- it's used
                                      # again here for the judge calls

  # 3. Everything except the two LLM-judge sections (D, E) -- e.g. no
  #    GROQ_API_KEY handy right now, or saving Groq quota:
  python3 eval/run_eval.py --no-llm-judge

Every stage degrades independently and never crashes the run: if the
server isn't reachable, B/C/D/E are skipped (logged, not raised) and A
alone still runs and still gets written to the report; if
GROQ_API_KEY isn't set, D/E are skipped the same way -- the same
graceful-degradation spirit as router.py's own LLM-unavailable
fallback path.

Reads:  eval/traces/conversation_*.json  (see scripts/parse_traces.py)
Writes: eval/results/metrics.json, eval/results/report.md
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Literal

import requests
from pydantic import BaseModel, Field

# Same pattern as scripts/diagnose_llm.py -- this file lives one level
# below the repo root, so put the root on sys.path before importing
# anything under app/.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.catalog import get_catalog  # noqa: E402
from app.llm import LLMError, call_structured, new_deadline  # noqa: E402
from app.retrieval import Requirements, search, warm_up  # noqa: E402
from eval.metrics import DEFAULT_K_VALUES, aggregate, evaluate_ranking  # noqa: E402

TRACES_DIR = ROOT / "eval" / "traces"
RESULTS_DIR = ROOT / "eval" / "results"
DEFAULT_SERVER_URL = "http://localhost:8000"

# Generous, fixed per-judge-call budget. This is an offline batch
# script, not a live user-facing request, so there's no reason to use
# the app's tight 22s /chat budget -- we'd rather wait longer and get
# a real judgment than fall back early.
JUDGE_CALL_BUDGET_SECONDS = 60.0


# ======================================================================
# Judge output schemas (internal to this script -- not part of the
# public API contract in app/schemas.py)
# ======================================================================

class UnsupportedClaim(BaseModel):
    claim: str
    reason: str = ""


class GroundednessJudgment(BaseModel):
    fully_grounded: bool = True
    unsupported_claims: list[UnsupportedClaim] = Field(default_factory=list)
    groundedness_score: float = 1.0  # judge's own 0.0-1.0 holistic estimate


class ItemRelevance(BaseModel):
    name: str
    verdict: Literal["relevant", "partially_relevant", "not_relevant"] = "not_relevant"
    reason: str = ""


class RelevanceJudgment(BaseModel):
    items: list[ItemRelevance] = Field(default_factory=list)
    overall_relevance_score: int = 3  # 1-5
    overall_reason: str = ""


GROUNDEDNESS_JUDGE_SYSTEM_PROMPT = """You are auditing an AI assistant's reply for factual accuracy \
against ground-truth catalog data. You will be given the assistant's reply text, and, for each \
assessment it recommended, the ACTUAL verified facts about that assessment (test type, duration, \
remote-testing support, adaptive/IRT support, applicable job levels, languages, and description) \
pulled directly from SHL's own catalog.

Read the reply and identify every SPECIFIC, checkable factual claim it makes about an assessment's \
properties -- duration, remote/adaptive support, job-level fit, test type or content, languages, and \
similar. General framing or subjective language ("a strong fit for leadership roles") is not a \
checkable claim; "takes about 15 minutes" is. For each checkable claim, decide whether the provided \
catalog facts support it, contradict it, or don't mention it at all (in which case it's unverifiable, \
not automatically wrong -- only flag it if it actually conflicts with or clearly overstates the given \
facts).

Be strict but fair: reasonable paraphrase or an inference clearly implied by the facts is NOT a \
violation (describing a "Knowledge & Skills" test as testing "technical knowledge" is a fair \
paraphrase). Only flag a claim that actually goes beyond or conflicts with what was given.

Respond with ONLY a JSON object: {"fully_grounded": bool, "unsupported_claims": \
[{"claim": str, "reason": str}, ...], "groundedness_score": float} where groundedness_score is your \
own holistic 0.0-1.0 estimate of what fraction of the reply's checkable claims are properly grounded \
(1.0 = every claim supported, 0.0 = the reply is mostly fabricated)."""

RELEVANCE_JUDGE_SYSTEM_PROMPT = """You are grading the relevance of assessment recommendations \
against what a hiring professional actually asked for, independent of any fixed answer key -- there \
can be more than one reasonable shortlist for a given need.

You will be given the full conversation (the stated hiring need, role, and any constraints or \
refinements made across turns) and the list of assessments the assistant ultimately recommended, each \
with its real catalog description.

For each recommended assessment, decide: "relevant" (a sound fit for the stated need), \
"partially_relevant" (defensible but not a strong fit, or only fits part of the requirement), or \
"not_relevant" (doesn't fit the stated need). Give a one-sentence reason for each.

Then give an overall_relevance_score from 1 (poor -- most of the list doesn't fit) to 5 (excellent -- \
a hiring professional would be satisfied with exactly this shortlist), plus a one-sentence \
overall_reason.

Judge only relevance to what was actually asked. Do NOT penalize the list for omitting some other \
plausible assessment unless what IS included is itself a poor fit for what was asked -- completeness \
against a reference list is scored elsewhere; you are scoring fit only.

Respond with ONLY a JSON object: {"items": [{"name": str, "verdict": "relevant"|"partially_relevant"|\
"not_relevant", "reason": str}, ...], "overall_relevance_score": int, "overall_reason": str}"""


# ======================================================================
# Fixtures
# ======================================================================

def load_traces(traces_dir: Path) -> list[dict]:
    files = sorted(traces_dir.glob("conversation_*.json"),
                    key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise SystemExit(
            f"No trace fixtures found in {traces_dir}. Run "
            f"`python3 scripts/parse_traces.py` first (see its docstring for the source file it "
            f"expects)."
        )
    traces = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    return traces


# ======================================================================
# A. Retrieval quality (offline)
# ======================================================================

def run_retrieval_quality(traces: list[dict], k_values) -> tuple[list[dict], dict]:
    warm_up()
    per_conv = []
    for trace in traces:
        query = " ".join(trace["persona_facts"])
        relevant = {item["url"] for item in trace["expected_shortlist"]}
        results = search(query, requirements=Requirements(), top_k=max(k_values))
        predicted = [r["url"] for r in results]

        metrics = evaluate_ranking(predicted, relevant, k_values)
        metrics["conversation_id"] = trace["conversation_id"]
        metrics["query_preview"] = query[:160] + ("..." if len(query) > 160 else "")
        metrics["top_predicted_names"] = [r["name"] for r in results[:max(k_values)]]
        metrics["reference_names"] = [item["name"] for item in trace["expected_shortlist"]]
        per_conv.append(metrics)

    aggregated = aggregate([{k: v for k, v in m.items()
                              if k not in ("conversation_id", "query_preview",
                                           "top_predicted_names", "reference_names")}
                             for m in per_conv])
    return per_conv, aggregated


# ======================================================================
# Server connectivity + conversation replay (shared by B, C, D, E)
# ======================================================================

def check_server(server_url: str) -> bool:
    try:
        resp = requests.get(f"{server_url}/health", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def replay_conversation(trace: dict, server_url: str, http_timeout: float = 60.0) -> dict:
    """Sends each persona_facts turn to POST /chat in sequence, exactly
    as a real multi-turn user would, and returns everything downstream
    sections (C, D, E) need: the full turn-by-turn transcript, the
    final turn's recommendations/reply/eoc, and per-turn timing.

    A request-level failure (timeout, connection error, non-200) stops
    replay of THIS conversation early rather than raising -- the
    conversation is marked with `"error"` and whatever turns did
    complete are still returned, so one bad conversation can't take
    the whole run down."""
    history = []
    turns = []
    response_times = []
    error = None

    for user_msg in trace["persona_facts"]:
        history.append({"role": "user", "content": user_msg})
        t0 = time.perf_counter()
        try:
            resp = requests.post(f"{server_url}/chat", json={"messages": history}, timeout=http_timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            error = f"HTTP error on turn {len(turns) + 1}: {e!r}"
            break
        except (ValueError, KeyError) as e:
            error = f"Malformed /chat response on turn {len(turns) + 1}: {e!r}"
            break
        dt = time.perf_counter() - t0
        response_times.append(dt)

        reply = data.get("reply", "")
        recs = data.get("recommendations", [])
        eoc = bool(data.get("end_of_conversation", False))
        history.append({"role": "assistant", "content": reply})
        turns.append({"user": user_msg, "agent_reply": reply, "recommendations": recs,
                       "end_of_conversation": eoc, "response_time_s": dt})

    final = turns[-1] if turns else {"agent_reply": "", "recommendations": [], "end_of_conversation": False}
    return {
        "conversation_id": trace["conversation_id"],
        "turns": turns,
        "turns_used": len(turns),
        "turns_in_reference": trace["num_turns_in_reference"],
        "final_reply": final.get("agent_reply", ""),
        "final_recommendations": final.get("recommendations", []),
        "final_eoc": final.get("end_of_conversation", False),
        "expected_shortlist": trace["expected_shortlist"],
        "response_times": response_times,
        "error": error,
    }


# ======================================================================
# B. End-to-end recommendation quality
# ======================================================================

def run_end_to_end(replays: list[dict], k_values) -> tuple[list[dict], dict]:
    per_conv = []
    for r in replays:
        relevant = {item["url"] for item in r["expected_shortlist"]}
        predicted = [rec["url"] for rec in r["final_recommendations"]]
        metrics = evaluate_ranking(predicted, relevant, k_values)
        metrics["conversation_id"] = r["conversation_id"]
        metrics["turns_used"] = r["turns_used"]
        metrics["turns_in_reference"] = r["turns_in_reference"]
        metrics["final_eoc"] = r["final_eoc"]
        metrics["error"] = r["error"]
        # 1.0 if we finished in <= the reference's turn count, decaying
        # toward 0 the further over we run -- an "extra clarifying
        # question that wasn't really needed" signal, not a hard cutoff.
        if r["turns_in_reference"]:
            over = r["turns_used"] - r["turns_in_reference"]
            metrics["turn_efficiency"] = 1.0 if over <= 0 else max(0.0, 1.0 - over / r["turns_in_reference"])
        else:
            metrics["turn_efficiency"] = None
        per_conv.append(metrics)

    aggregated = aggregate([{k: v for k, v in m.items()
                              if k not in ("conversation_id", "turns_used", "turns_in_reference",
                                           "final_eoc", "error")}
                             for m in per_conv])
    return per_conv, aggregated


# ======================================================================
# C. Deterministic groundedness (whitelist re-check, no LLM)
# ======================================================================

def run_deterministic_groundedness(replays: list[dict]) -> dict:
    catalog = get_catalog()
    total = 0
    violations = []
    for r in replays:
        for turn_idx, turn in enumerate(r["turns"], start=1):
            for rec in turn["recommendations"]:
                total += 1
                item = catalog.by_url.get(rec.get("url", ""))
                is_real = catalog.is_real(rec.get("name", ""), rec.get("url", ""))
                type_matches = bool(item) and item.get("test_type") == rec.get("test_type")
                if not is_real or not type_matches:
                    violations.append({
                        "conversation_id": r["conversation_id"],
                        "turn": turn_idx,
                        "recommendation": rec,
                        "reason": "not in catalog (name/url mismatch)" if not is_real
                                  else f"test_type mismatch: reported {rec.get('test_type')!r}, "
                                       f"catalog has {item.get('test_type')!r}",
                    })
    return {
        "total_recommendations_checked": total,
        "num_violations": len(violations),
        "hallucination_rate": (len(violations) / total) if total else None,
        "violations": violations,
    }


# ======================================================================
# D. LLM-judged groundedness / faithfulness
# ======================================================================

def _catalog_evidence_block(item: dict) -> str:
    duration = f"{item['duration_minutes']} min" if item.get("duration_minutes") else (
        item.get("duration_display") or "not specified in catalog")
    return (
        f"- {item['name']}\n"
        f"    test_type: {item.get('test_type', 'n/a')}\n"
        f"    duration: {duration}\n"
        f"    remote_testing: {item.get('remote_testing')}\n"
        f"    adaptive_irt: {item.get('adaptive_irt')}\n"
        f"    job_levels: {', '.join(item.get('job_levels', [])) or 'not specified'}\n"
        f"    languages: {', '.join(item.get('languages', [])) or 'not specified'}\n"
        f"    description: {item.get('description', '')[:400]}"
    )


def run_llm_groundedness(replays: list[dict]) -> tuple[list[dict], dict]:
    catalog = get_catalog()
    per_conv = []
    for r in replays:
        if not r["final_recommendations"]:
            per_conv.append({"conversation_id": r["conversation_id"], "skipped": "no final recommendations"})
            continue

        evidence_items = []
        for rec in r["final_recommendations"]:
            item = catalog.by_url.get(rec.get("url", ""))
            if item:
                evidence_items.append(item)
        if not evidence_items:
            per_conv.append({"conversation_id": r["conversation_id"],
                              "skipped": "no recommended items resolved to real catalog rows"})
            continue

        user_content = (
            f"Assistant's reply:\n{r['final_reply']}\n\n"
            f"Verified catalog facts for the recommended assessments:\n"
            + "\n".join(_catalog_evidence_block(it) for it in evidence_items)
        )
        try:
            judgment = call_structured(
                GROUNDEDNESS_JUDGE_SYSTEM_PROMPT, user_content, GroundednessJudgment,
                temperature=0.0, deadline=new_deadline(JUDGE_CALL_BUDGET_SECONDS),
            )
            per_conv.append({"conversation_id": r["conversation_id"],
                              "fully_grounded": judgment.fully_grounded,
                              "groundedness_score": judgment.groundedness_score,
                              "unsupported_claims": [c.model_dump() for c in judgment.unsupported_claims]})
        except LLMError as e:
            per_conv.append({"conversation_id": r["conversation_id"], "skipped": f"judge call failed: {e}"})
        except Exception as e:
            # Deliberately broader than LLMError: app.llm wraps the failure
            # modes it anticipates (timeout, HTTP error status, rate limit)
            # but not every possible transport-level exception (e.g. a raw
            # connection failure). One conversation's judge call failing in
            # an unanticipated way should never cost the other 9 their
            # already-computed results.
            per_conv.append({"conversation_id": r["conversation_id"],
                              "skipped": f"judge call failed unexpectedly: {e!r}"})

    scored = [c for c in per_conv if "groundedness_score" in c]
    aggregated = {
        "num_judged": len(scored),
        "num_skipped": len(per_conv) - len(scored),
        "mean_groundedness_score": (sum(c["groundedness_score"] for c in scored) / len(scored)) if scored else None,
        "fraction_fully_grounded": (sum(1 for c in scored if c["fully_grounded"]) / len(scored)) if scored else None,
        "total_unsupported_claims": sum(len(c["unsupported_claims"]) for c in scored),
    }
    return per_conv, aggregated


# ======================================================================
# E. LLM-judged recommendation relevance
# ======================================================================

def run_llm_relevance(replays: list[dict]) -> tuple[list[dict], dict]:
    catalog = get_catalog()
    per_conv = []
    for r in replays:
        if not r["final_recommendations"]:
            per_conv.append({"conversation_id": r["conversation_id"], "skipped": "no final recommendations"})
            continue

        evidence_items = []
        for rec in r["final_recommendations"]:
            item = catalog.by_url.get(rec.get("url", ""))
            if item:
                evidence_items.append(item)
        if not evidence_items:
            per_conv.append({"conversation_id": r["conversation_id"],
                              "skipped": "no recommended items resolved to real catalog rows"})
            continue

        transcript = "\n".join(f"User: {t['user']}\nAssistant: {t['agent_reply']}" for t in r["turns"])
        user_content = (
            f"Conversation:\n{transcript}\n\n"
            f"Recommended assessments:\n" + "\n".join(_catalog_evidence_block(it) for it in evidence_items)
        )
        try:
            judgment = call_structured(
                RELEVANCE_JUDGE_SYSTEM_PROMPT, user_content, RelevanceJudgment,
                temperature=0.0, deadline=new_deadline(JUDGE_CALL_BUDGET_SECONDS),
            )
            items = [i.model_dump() for i in judgment.items]
            n = len(items) or 1
            relevance_precision = sum(
                1.0 if i["verdict"] == "relevant" else 0.5 if i["verdict"] == "partially_relevant" else 0.0
                for i in items
            ) / n
            per_conv.append({"conversation_id": r["conversation_id"], "items": items,
                              "overall_relevance_score": judgment.overall_relevance_score,
                              "overall_reason": judgment.overall_reason,
                              "relevance_precision": relevance_precision})
        except LLMError as e:
            per_conv.append({"conversation_id": r["conversation_id"], "skipped": f"judge call failed: {e}"})
        except Exception as e:
            per_conv.append({"conversation_id": r["conversation_id"],
                              "skipped": f"judge call failed unexpectedly: {e!r}"})

    scored = [c for c in per_conv if "relevance_precision" in c]
    aggregated = {
        "num_judged": len(scored),
        "num_skipped": len(per_conv) - len(scored),
        "mean_relevance_precision": (sum(c["relevance_precision"] for c in scored) / len(scored)) if scored else None,
        "mean_overall_relevance_score": (sum(c["overall_relevance_score"] for c in scored) / len(scored)) if scored else None,
    }
    return per_conv, aggregated


# ======================================================================
# Overall accuracy / effectiveness summary
# ======================================================================

def build_overall_summary(e2e_agg: dict, groundedness_det: dict, groundedness_llm_agg: dict,
                           relevance_agg: dict, e2e_per_conv: list[dict]) -> dict:
    """A single composite number is easy to game and easy to
    over-trust, so this is presented as a convenience summary
    *alongside* the raw metrics above, not a replacement for them (see
    the report's methodology section). Weights are a judgment call,
    documented here rather than left as unexplained magic numbers:

    - 40% end-to-end F1@10 vs the reference shortlist -- the closest
      single number to "did the user get the right answer."
    - 25% LLM-judged relevance precision -- catches valid
      recommendations the fixed reference wouldn't credit.
    - 25% LLM-judged groundedness score -- faithfulness of the prose,
      not just the structured recommendation objects.
    - 10% turn efficiency -- didn't drag out clarification well past
      what the reference needed.

    Any component that couldn't be computed (server unreachable, judge
    unavailable) is dropped and the remaining weights renormalized,
    rather than silently treated as 0 or as 1.

    HARD PENALTY: any deterministic hallucination (section C) halves
    the composite score and is called out explicitly, regardless of
    how well everything else scored -- a recommender that invents a
    catalog entry even once is a critical failure mode independent of
    average quality elsewhere.
    """
    components = {
        "end_to_end_f1": (e2e_agg.get("f1@10"), 0.40),
        "llm_relevance_precision": (relevance_agg.get("mean_relevance_precision"), 0.25),
        "llm_groundedness_score": (groundedness_llm_agg.get("mean_groundedness_score"), 0.25),
        "turn_efficiency": (e2e_agg.get("turn_efficiency"), 0.10),
    }
    available = {k: (v, w) for k, (v, w) in components.items() if v is not None}
    weight_sum = sum(w for _, w in available.values())
    composite = (sum(v * w for v, w in available.values()) / weight_sum) if weight_sum else None

    hallucination_rate = groundedness_det.get("hallucination_rate")
    penalized = False
    if composite is not None and hallucination_rate:
        composite *= 0.5
        penalized = True

    return {
        "components_used": {k: v for k, (v, _) in available.items()},
        "components_skipped": [k for k in components if k not in available],
        "composite_score": composite,
        "hallucination_penalty_applied": penalized,
        "deterministic_hallucination_rate": hallucination_rate,
    }


# ======================================================================
# Report writing
# ======================================================================

def _fmt(x, digits=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{x:.{digits}f}" if isinstance(x, float) else str(x)
    return str(x)


def write_reports(results: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    md = []
    md.append("# SHL Recommendation Agent -- Quantitative Evaluation Report\n\n")
    md.append(f"Generated: **{results['generated_at']}**\n\n")
    md.append(
        "Scores the live system against SHL's own reference conversations "
        f"(`eval/traces/`, n={results['num_traces']}). Companion to `run_tests2.py`'s "
        "conversational-behavior suite -- see `eval/README.md` for full methodology, metric "
        "definitions, and known limitations of a 10-conversation reference set.\n\n---\n\n"
    )

    # --- A. Retrieval quality ---
    md.append("## A. Retrieval Quality (offline retriever, isolated from the LLM)\n\n")
    ra = results["retrieval_quality"]["aggregated"]
    if ra:
        md.append("| Metric | Value |\n|---|---|\n")
        for k in ["precision@5", "recall@5", "f1@5", "ndcg@5", "ap@5",
                  "precision@10", "recall@10", "f1@10", "ndcg@10", "ap@10", "mrr"]:
            if k in ra:
                md.append(f"| {k} | {_fmt(ra[k])} |\n")
        md.append(f"\nAveraged over {ra.get('num_queries', 0)} reference conversations.\n\n")
    else:
        md.append("_Not run._\n\n")

    # --- B. End-to-end ---
    md.append("## B. End-to-End Recommendation Quality (live /chat, full conversation replay)\n\n")
    e2e = results["end_to_end"]
    if e2e["status"] == "ok":
        ea = e2e["aggregated"]
        md.append("| Metric | Value |\n|---|---|\n")
        for k in ["precision@5", "recall@5", "f1@5", "ndcg@5", "ap@5",
                  "precision@10", "recall@10", "f1@10", "ndcg@10", "ap@10", "mrr", "turn_efficiency"]:
            if k in ea:
                md.append(f"| {k} | {_fmt(ea[k])} |\n")
        md.append(f"\nAveraged over {ea.get('num_queries', 0)} replayed conversations.\n\n")
        md.append("### Per-conversation detail\n\n")
        md.append("| ID | P@10 | R@10 | F1@10 | MRR | Turns (used/ref) | Final EOC | Error |\n"
                   "|---|---|---|---|---|---|---|---|\n")
        for m in e2e["per_conversation"]:
            md.append(f"| {m['conversation_id']} | {_fmt(m.get('precision@10'))} | "
                       f"{_fmt(m.get('recall@10'))} | {_fmt(m.get('f1@10'))} | {_fmt(m.get('mrr'))} | "
                       f"{m['turns_used']}/{m['turns_in_reference']} | {m['final_eoc']} | "
                       f"{m['error'] or ''} |\n")
        md.append("\n")
    else:
        md.append(f"_Skipped: {e2e['status']}_\n\n")

    # --- C. Deterministic groundedness ---
    md.append("## C. Deterministic Groundedness (whitelist re-check against the live catalog)\n\n")
    gd = results["groundedness_deterministic"]
    if gd["status"] == "ok":
        d = gd["data"]
        md.append(f"- Recommendations checked: **{d['total_recommendations_checked']}**\n")
        md.append(f"- Violations (hallucinated / mismatched): **{d['num_violations']}**\n")
        md.append(f"- Hallucination rate: **{_fmt(d['hallucination_rate'], 4)}**\n\n")
        if d["violations"]:
            md.append("**Violations found:**\n\n")
            for v in d["violations"]:
                md.append(f"- Conversation {v['conversation_id']}, turn {v['turn']}: "
                           f"`{v['recommendation'].get('name')}` -- {v['reason']}\n")
            md.append("\n")
    else:
        md.append(f"_Skipped: {gd['status']}_\n\n")

    # --- D. LLM-judged groundedness ---
    md.append("## D. LLM-Judged Groundedness / Faithfulness (free-text reply vs. catalog facts)\n\n")
    gl = results["groundedness_llm"]
    if gl["status"] == "ok":
        a = gl["aggregated"]
        md.append(f"- Conversations judged: **{a['num_judged']}** (skipped: {a['num_skipped']})\n")
        md.append(f"- Mean groundedness score: **{_fmt(a['mean_groundedness_score'])}**\n")
        md.append(f"- Fraction fully grounded: **{_fmt(a['fraction_fully_grounded'])}**\n")
        md.append(f"- Total unsupported claims flagged: **{a['total_unsupported_claims']}**\n\n")
        flagged = [c for c in gl["per_conversation"] if c.get("unsupported_claims")]
        if flagged:
            md.append("**Unsupported claims flagged:**\n\n")
            for c in flagged:
                for claim in c["unsupported_claims"]:
                    md.append(f"- Conversation {c['conversation_id']}: \"{claim['claim']}\" -- {claim['reason']}\n")
            md.append("\n")
    else:
        md.append(f"_Skipped: {gl['status']}_\n\n")

    # --- E. LLM-judged relevance ---
    md.append("## E. LLM-Judged Recommendation Relevance (independent of the reference list)\n\n")
    rl = results["relevance_llm"]
    if rl["status"] == "ok":
        a = rl["aggregated"]
        md.append(f"- Conversations judged: **{a['num_judged']}** (skipped: {a['num_skipped']})\n")
        md.append(f"- Mean relevance precision (relevant=1.0, partial=0.5, not=0.0): "
                   f"**{_fmt(a['mean_relevance_precision'])}**\n")
        md.append(f"- Mean overall relevance score (judge's 1-5 holistic rating): "
                   f"**{_fmt(a['mean_overall_relevance_score'])}**\n\n")
        md.append("| ID | Relevance precision | Overall score (1-5) | Reason |\n|---|---|---|---|\n")
        for c in rl["per_conversation"]:
            if "relevance_precision" in c:
                md.append(f"| {c['conversation_id']} | {_fmt(c['relevance_precision'])} | "
                          f"{c['overall_relevance_score']} | {c['overall_reason']} |\n")
        md.append("\n")
    else:
        md.append(f"_Skipped: {rl['status']}_\n\n")

    # --- Latency ---
    md.append("## Latency (end-to-end replay, wall-clock per /chat call)\n\n")
    lat = results["latency"]
    if lat:
        md.append(f"- Mean: **{_fmt(lat['mean'])}s** | Median: **{_fmt(lat['median'])}s** | "
                   f"Max: **{_fmt(lat['max'])}s** | n={lat['n']}\n"
                   f"- Assignment's stated hard limit is 30s/call -- see app/llm.py's budget constants.\n\n")
    else:
        md.append("_Not available (end-to-end section did not run)._\n\n")

    # --- Overall ---
    md.append("## Overall Accuracy / Effectiveness Summary\n\n")
    ov = results["overall"]
    if ov["composite_score"] is not None:
        md.append(f"### Composite effectiveness score: **{_fmt(ov['composite_score'])}** / 1.0\n\n")
        if ov["hallucination_penalty_applied"]:
            md.append("> **CRITICAL:** deterministic hallucination(s) detected -- composite score was "
                       "halved regardless of other metrics. See section C above.\n\n")
        md.append("Components used (see run_eval.py's `build_overall_summary` docstring for weights "
                   "and rationale):\n\n| Component | Value |\n|---|---|\n")
        for k, v in ov["components_used"].items():
            md.append(f"| {k} | {_fmt(v)} |\n")
        if ov["components_skipped"]:
            md.append(f"\nSkipped (weights renormalized over the rest): {', '.join(ov['components_skipped'])}\n")
        md.append("\n")
    else:
        md.append("_Could not compute -- no sections with usable output ran. Run at least "
                   "`--retrieval-only`, or start the server for the full pipeline._\n\n")

    (out_dir / "report.md").write_text("".join(md), encoding="utf-8")


# ======================================================================
# Orchestration
# ======================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces-dir", default=str(TRACES_DIR))
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    ap.add_argument("--server-url", default=DEFAULT_SERVER_URL,
                     help="Base URL of a running `uvicorn app.main:app` instance.")
    ap.add_argument("--k", default="5,10", help="Comma-separated k values for @k metrics.")
    ap.add_argument("--retrieval-only", action="store_true",
                     help="Run section A only (offline; no server, no GROQ_API_KEY needed).")
    ap.add_argument("--no-llm-judge", action="store_true",
                     help="Run A/B/C but skip the LLM-judge sections D and E.")
    args = ap.parse_args()

    k_values = tuple(int(x) for x in args.k.split(","))
    traces = load_traces(Path(args.traces_dir))
    print(f"Loaded {len(traces)} reference conversations from {args.traces_dir}")

    results = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_traces": len(traces),
        "k_values": list(k_values),
    }

    # --- A ---
    print("\n[A] Retrieval quality (offline)...")
    ra_per_conv, ra_agg = run_retrieval_quality(traces, k_values)
    results["retrieval_quality"] = {"per_conversation": ra_per_conv, "aggregated": ra_agg}
    print(f"    mean recall@10={_fmt(ra_agg.get('recall@10'))}  "
          f"mean precision@10={_fmt(ra_agg.get('precision@10'))}  mean MRR={_fmt(ra_agg.get('mrr'))}")

    if args.retrieval_only:
        results["end_to_end"] = {"status": "skipped (--retrieval-only)"}
        results["groundedness_deterministic"] = {"status": "skipped (--retrieval-only)"}
        results["groundedness_llm"] = {"status": "skipped (--retrieval-only)"}
        results["relevance_llm"] = {"status": "skipped (--retrieval-only)"}
        results["latency"] = None
        results["overall"] = build_overall_summary(
            {}, {"hallucination_rate": None}, {}, {}, [])
        write_reports(results, Path(args.out_dir))
        print(f"\nWrote {args.out_dir}/metrics.json and report.md (retrieval-only run).")
        return

    server_up = check_server(args.server_url)
    if not server_up:
        print(f"\n[B-E] Server not reachable at {args.server_url} -- skipping sections B-E.\n"
              f"      Start it with: uvicorn app.main:app --port 8000\n"
              f"      (with GROQ_API_KEY set in that shell for real LLM-backed responses)")
        for key in ("end_to_end", "groundedness_deterministic", "groundedness_llm", "relevance_llm"):
            results[key] = {"status": f"skipped (server unreachable at {args.server_url})"}
        results["latency"] = None
        results["overall"] = build_overall_summary(
            {}, {"hallucination_rate": None}, {}, {}, [])
        write_reports(results, Path(args.out_dir))
        print(f"\nWrote {args.out_dir}/metrics.json and report.md (section A only).")
        return

    # --- Replay all conversations once, reuse for B, C, D, E ---
    print(f"\n[B] Replaying {len(traces)} conversations against {args.server_url}/chat...")
    replays = []
    for trace in traces:
        r = replay_conversation(trace, args.server_url)
        status = "error" if r["error"] else "ok"
        print(f"    conversation {r['conversation_id']}: {r['turns_used']} turns, "
              f"{len(r['final_recommendations'])} final recs [{status}]"
              + (f" -- {r['error']}" if r["error"] else ""))
        replays.append(r)

    e2e_per_conv, e2e_agg = run_end_to_end(replays, k_values)
    results["end_to_end"] = {"status": "ok", "per_conversation": e2e_per_conv, "aggregated": e2e_agg}
    print(f"    mean recall@10={_fmt(e2e_agg.get('recall@10'))}  "
          f"mean precision@10={_fmt(e2e_agg.get('precision@10'))}  mean F1@10={_fmt(e2e_agg.get('f1@10'))}")

    all_times = [t for r in replays for t in r["response_times"]]
    results["latency"] = ({"mean": statistics.mean(all_times), "median": statistics.median(all_times),
                            "max": max(all_times), "n": len(all_times)} if all_times else None)

    print("\n[C] Deterministic groundedness re-check...")
    det = run_deterministic_groundedness(replays)
    results["groundedness_deterministic"] = {"status": "ok", "data": det}
    print(f"    {det['num_violations']} violation(s) out of {det['total_recommendations_checked']} "
          f"recommendations checked.")

    if args.no_llm_judge:
        results["groundedness_llm"] = {"status": "skipped (--no-llm-judge)"}
        results["relevance_llm"] = {"status": "skipped (--no-llm-judge)"}
    elif not os.environ.get("GROQ_API_KEY"):
        print("\n[D/E] GROQ_API_KEY not set in this shell -- skipping LLM-judge sections. "
              "Export it and re-run (server can keep running).")
        results["groundedness_llm"] = {"status": "skipped (GROQ_API_KEY not set)"}
        results["relevance_llm"] = {"status": "skipped (GROQ_API_KEY not set)"}
    else:
        print("\n[D] LLM-judged groundedness / faithfulness...")
        gl_per_conv, gl_agg = run_llm_groundedness(replays)
        results["groundedness_llm"] = {"status": "ok", "per_conversation": gl_per_conv, "aggregated": gl_agg}
        print(f"    mean groundedness score={_fmt(gl_agg.get('mean_groundedness_score'))}  "
              f"({gl_agg.get('num_judged', 0)} judged, {gl_agg.get('num_skipped', 0)} skipped)")

        print("\n[E] LLM-judged recommendation relevance...")
        rl_per_conv, rl_agg = run_llm_relevance(replays)
        results["relevance_llm"] = {"status": "ok", "per_conversation": rl_per_conv, "aggregated": rl_agg}
        print(f"    mean relevance precision={_fmt(rl_agg.get('mean_relevance_precision'))}  "
              f"({rl_agg.get('num_judged', 0)} judged, {rl_agg.get('num_skipped', 0)} skipped)")

    groundedness_llm_agg = results["groundedness_llm"].get("aggregated", {})
    relevance_agg = results["relevance_llm"].get("aggregated", {})
    results["overall"] = build_overall_summary(e2e_agg, det, groundedness_llm_agg, relevance_agg, e2e_per_conv)

    write_reports(results, Path(args.out_dir))
    print(f"\nWrote {args.out_dir}/metrics.json and report.md")
    if results["overall"]["composite_score"] is not None:
        print(f"Overall effectiveness score: {_fmt(results['overall']['composite_score'])} / 1.0"
              + (" (HALLUCINATION PENALTY APPLIED)" if results["overall"]["hallucination_penalty_applied"] else ""))


if __name__ == "__main__":
    main()