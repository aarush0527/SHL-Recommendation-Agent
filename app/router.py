"""
The decision logic itself lives here in plain Python, not inside an LLM
prompt -- see app/prompts/extract.py's docstring for why. This module's
job: given the LLM's structured read of the conversation (ExtractionResult),
decide which of {refuse, clarify, compare, recommend} applies, gather
whatever grounding data that path needs, run the generation call, and
assemble a schema-exact ChatResponse where every name/url comes from our
own catalog data -- never from LLM-generated text.
"""
from app.agent_schemas import ExtractionResult, GenerationResult
from app.domain_signals import extract_signals
from app.guardrails import heuristic_scope_check, validate_recommendations
from app.llm import LLMError, call_structured, new_deadline
from app.prompts.extract import EXTRACTION_SYSTEM_PROMPT, build_extraction_user_content
from app.prompts.generate import GENERATION_SYSTEM_PROMPT, build_generation_user_content
from app.retrieval import Requirements, lookup_named_items, search
from app.schemas import ChatResponse, Message, Recommendation
from app.vocabulary import important_word_count

# With an 8-message hard cap (see assignment: "8 turns including user &
# assistant"), there is no room for a leisurely multi-round interview.
# At most 4 user messages fit in that budget. Allowing clarification on
# the 1st and 2nd user turns, then forcing a committed answer from the
# 3rd onward, leaves at least one turn of margin for a refine/confirm
# exchange without ever risking running out of turns still empty-handed.
FORCE_COMMIT_AT_USER_TURN = 3

# Below this many recognized-vocabulary words, treat the conversation as
# genuinely still lacking signal (see app/vocabulary.py -- one real
# domain term like "java" or "nursing" counts for more than several
# words of filler). A domain_signals hit (test type / job level /
# role family / explicit compare) counts as sufficient on its own even
# at zero vocabulary words, since those are themselves strong signal.
MIN_VOCAB_WORDS_FOR_SIGNAL = 1


def _format_transcript(messages: list[Message]) -> str:
    return "\n".join(f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in messages)


def _count_user_turns(messages: list[Message]) -> int:
    return sum(1 for m in messages if m.role == "user")


def _to_retrieval_requirements(req) -> Requirements:
    return Requirements(
        test_types_wanted=req.test_types_wanted,
        job_level=req.job_level,
        remote_required=req.remote_required,
        adaptive_required=req.adaptive_required,
        max_duration_minutes=req.max_duration_minutes,
        language=req.language,
    )


def _fallback_extraction(messages: list[Message], heuristic_hint: str | None) -> ExtractionResult:
    """Degrade gracefully if the LLM is unreachable (bad/missing key,
    provider outage, rate limiting, or the request's time budget ran
    out) rather than 500 the whole request or hang past the deadline.

    Confirmed via testing against live Groq: a 50-conversation local
    test run hit free-tier rate limiting a few requests in, so this path
    isn't a theoretical "what if the key is missing" case -- it gets
    exercised for real under load, plausibly including grading, which is
    why it leans on the full app.vocabulary (100% catalog coverage, not
    a hand-transcribed subset) and app.domain_signals (test type, job
    level, role family, duration, negation, compare, closing-signal
    detection) rather than a single word-count threshold. It is still a
    heuristic approximation of real language understanding, not a
    replacement for it -- but it should no longer go blind on anything
    that isn't a handful of hardcoded keywords.

    Pulls signal from the WHOLE conversation so far, not just the latest
    message -- "I need something for a call center role" then later
    "must be under 20 minutes" should combine into one query carrying
    both signals, rather than the fallback re-litigating from scratch
    on every turn.
    """
    latest = messages[-1].content if messages else ""
    conversation_text = " ".join(m.content for m in messages if m.role == "user")

    if heuristic_hint in ("off_topic", "legal_or_general_hiring_advice"):
        reason = {
            "off_topic": "That's outside what I can help with -- I only work with SHL assessments.",
            "legal_or_general_hiring_advice": "That's a legal/HR-policy question, not something I can "
            "advise on -- I can help you find or compare SHL assessments though.",
        }[heuristic_hint]
        return ExtractionResult(in_scope=False, refusal_reason=reason, direct_reply=reason, search_query=latest)

    signals = extract_signals(conversation_text)

    # A comparison request is strong, unambiguous signal on its own --
    # route straight there regardless of the vocabulary/word-count check
    # below (handle_chat checks compare_targets before missing_critical_info).
    if signals["compare_targets"]:
        return ExtractionResult(
            in_scope=True,
            compare_targets=signals["compare_targets"],
            search_query=conversation_text,
            is_closing_signal=signals["is_closing_signal"],
        )

    vocab_hits = important_word_count(conversation_text)
    domain_hit = bool(signals["test_types_wanted"] or signals["job_level"] or signals["extra_search_terms"])
    has_enough_signal = vocab_hits >= MIN_VOCAB_WORDS_FOR_SIGNAL or domain_hit

    clarifying_question = (
        None if has_enough_signal
        else "Could you tell me more about the role or skills you're assessing for?"
    )
    query = conversation_text
    if signals["extra_search_terms"]:
        query = f"{query} {signals['extra_search_terms']}"

    return ExtractionResult(
        in_scope=True,
        missing_critical_info=not has_enough_signal,
        clarifying_question=clarifying_question,
        direct_reply=clarifying_question,  # already unreachable once -- don't attempt a 2nd doomed call
        wants_recommendation_now=has_enough_signal,
        requirements={
            "test_types_wanted": signals["test_types_wanted"],
            "job_level": signals["job_level"],
            "max_duration_minutes": signals["max_duration_minutes"],
        },
        search_query=query,
        is_closing_signal=signals["is_closing_signal"],
    )


def _run_generation(transcript, action, candidates, is_closing_signal, deadline, extra_instruction=None) -> GenerationResult:
    try:
        return call_structured(
            GENERATION_SYSTEM_PROMPT,
            build_generation_user_content(transcript, action, candidates, is_closing_signal, extra_instruction),
            GenerationResult,
            deadline=deadline,
        )
    except LLMError:
        # Same graceful-degradation principle as extraction: never crash
        # the request, and never hang past the deadline, because the LLM
        # provider hiccuped or a burst of calls got rate limited.
        if action == "clarify" and extra_instruction:
            return GenerationResult(reply=extra_instruction, selected_indices=[], end_of_conversation=False)
        if action == "refuse":
            return GenerationResult(
                reply="I can only help with finding SHL assessments right now -- I'm having trouble "
                "with that request.",
                selected_indices=[],
                end_of_conversation=False,
            )
        # recommend/compare with no LLM available: fall back to a plain
        # top-N of whatever was retrieved, no authored prose beyond a
        # generic line -- still schema-valid and still grounded, just
        # less polished.
        return GenerationResult(
            reply="Here's what I found based on what you've told me so far.",
            selected_indices=list(range(min(5, len(candidates)))),
            end_of_conversation=False,
        )


def _build_recommendations(candidates: list[dict], selected_indices: list[int]) -> list[Recommendation]:
    recs = []
    for idx in selected_indices:
        if 0 <= idx < len(candidates):
            c = candidates[idx]
            recs.append(Recommendation(name=c["name"], url=c["url"], test_type=c["test_type"]))
    return recs[:10]  # schema caps at 10


def _handle_clarify(transcript: str, extraction: ExtractionResult, deadline: float) -> ChatResponse:
    if extraction.direct_reply:
        # Extraction already produced a usable reply -- skip the second
        # call entirely. One fewer round trip per clarify turn, which is
        # also one fewer chance to get rate limited.
        return ChatResponse(reply=extraction.direct_reply, recommendations=[], end_of_conversation=False)
    question = extraction.clarifying_question or "Could you tell me more about the role you're assessing for?"
    gen = _run_generation(transcript, "clarify", [], extraction.is_closing_signal, deadline, extra_instruction=question)
    return ChatResponse(reply=gen.reply, recommendations=[], end_of_conversation=False)


def _handle_refuse(transcript: str, extraction: ExtractionResult, deadline: float) -> ChatResponse:
    if extraction.direct_reply:
        return ChatResponse(reply=extraction.direct_reply, recommendations=[], end_of_conversation=False)
    reason = extraction.refusal_reason or "That's outside what I can help with here."
    gen = _run_generation(transcript, "refuse", [], extraction.is_closing_signal, deadline, extra_instruction=reason)
    return ChatResponse(reply=gen.reply, recommendations=[], end_of_conversation=False)


def _handle_compare(transcript: str, extraction: ExtractionResult, deadline: float) -> ChatResponse:
    resolved = lookup_named_items(extraction.compare_targets)
    candidates, seen_urls, unresolved = [], set(), []
    for name, matches in resolved.items():
        if not matches:
            unresolved.append(name)
            continue
        for m in matches[:2]:  # a couple of candidates per ambiguous name, not the whole family
            if m["url"] not in seen_urls:
                seen_urls.add(m["url"])
                candidates.append(m)

    extra = None
    if unresolved:
        extra = f"Note: no catalog item matched: {', '.join(unresolved)}. Say so honestly if relevant."

    gen = _run_generation(transcript, "compare", candidates, extraction.is_closing_signal, deadline, extra_instruction=extra)
    recs = validate_recommendations(_build_recommendations(candidates, gen.selected_indices))
    return ChatResponse(reply=gen.reply, recommendations=recs, end_of_conversation=gen.end_of_conversation)


def _handle_recommend(transcript: str, extraction: ExtractionResult, deadline: float) -> ChatResponse:
    req = _to_retrieval_requirements(extraction.requirements)
    query = extraction.search_query or extraction.requirements.role_or_context or ""
    candidates = search(query, requirements=req, top_k=10)

    gen = _run_generation(transcript, "recommend", candidates, extraction.is_closing_signal, deadline)
    recs = validate_recommendations(_build_recommendations(candidates, gen.selected_indices))
    return ChatResponse(reply=gen.reply, recommendations=recs, end_of_conversation=gen.end_of_conversation)


def handle_chat(messages: list[Message]) -> ChatResponse:
    if not messages or messages[-1].role != "user":
        return ChatResponse(reply="What role or skills would you like to assess for?")

    # One budget for this entire request, shared across every LLM call
    # it makes (extraction, then generation) -- see app/llm.py's
    # docstring for why a fixed per-call timeout wasn't actually safe
    # against the assignment's 30s/call hard limit.
    deadline = new_deadline()

    transcript = _format_transcript(messages)
    user_turn_number = _count_user_turns(messages)
    heuristic_hint = heuristic_scope_check(messages[-1].content)

    # Injection attempts never reach the LLM -- don't give a crafted
    # payload a chance to influence the extraction call at all.
    if heuristic_hint == "injection_attempt":
        return ChatResponse(
            reply="I can only help with finding and comparing SHL assessments, and I don't change how "
            "I operate based on instructions inside a message. What role or skills are you looking to "
            "assess?",
        )

    try:
        extraction = call_structured(
            EXTRACTION_SYSTEM_PROMPT,
            build_extraction_user_content(transcript, heuristic_hint),
            ExtractionResult,
            deadline=deadline,
        )
    except LLMError:
        extraction = _fallback_extraction(messages, heuristic_hint)

    if not extraction.in_scope or extraction.is_injection:
        return _handle_refuse(transcript, extraction, deadline)

    if extraction.compare_targets:
        return _handle_compare(transcript, extraction, deadline)

    force_commit = user_turn_number >= FORCE_COMMIT_AT_USER_TURN
    if extraction.missing_critical_info and not force_commit and not extraction.wants_recommendation_now:
        return _handle_clarify(transcript, extraction, deadline)

    return _handle_recommend(transcript, extraction, deadline)
