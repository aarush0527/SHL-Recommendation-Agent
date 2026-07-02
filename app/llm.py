"""
Thin LLM client wrapper. Design choices worth calling out:

1. Provider is Groq by default (OpenAI-compatible endpoint, free tier,
   fast enough to matter against a 30s hard timeout) but nothing else in
   the codebase imports the Groq SDK directly -- everything goes through
   `call_structured()` below, so swapping providers (Gemini, OpenAI,
   whatever) is a one-file change.

2. We use plain `json_object` response-format mode, not the stricter
   provider-native `json_schema` mode, and validate the *shape* ourselves
   with Pydantic. Whatever the provider guarantees, we don't propagate an
   invalid structure into the rest of the pipeline.

3. Every call is deadline-aware, not fixed-timeout. handle_chat() makes
   up to two LLM calls per turn (extraction, then generation); a fixed
   12s-per-call budget with an internal "retry once on bad JSON" meant
   a single turn's worst case was 4 HTTP calls x 12s = 48s -- already
   over the assignment's 30s/call hard limit before even considering a
   rate-limit retry. That surfaced for real: testing against live Groq
   hit free-tier rate limiting partway through a 50-conversation run,
   and the fix needs to be time-budget-aware, not just "retry more."
   Every call site now takes a `deadline` (an absolute time.time()
   value); each attempt uses whatever budget is actually left, and the
   function bails out to LLMError (triggering the caller's fallback)
   the moment there isn't enough time left to be worth trying again --
   never blocks past the deadline hoping one more attempt pays off.

4. A module-global, thread-safe pacing gate keeps Groq call volume safe
   without needing to give up early (FastAPI runs sync `def` endpoints
   in a threadpool, so concurrent /chat requests really do call this
   module from different threads at once). Every real outgoing call --
   first attempt or retry, this request or a concurrent one -- funnels
   through the same lock and minimum spacing, so no combination of
   retries can burst past Groq's free-tier RPM ceiling. Because volume
   is already safe at the pacing layer, rate-limit retries don't need a
   tight cap of their own: they keep trying (with backoff, widening the
   pacing gap for a bit after a fresh 429) for as long as the request's
   deadline allows, which maximizes the chance of a real answer instead
   of an early fallback. (An earlier version capped rate-limit retries
   at exactly one and added a hard cooldown that blocked all calls the
   instant any 429 was seen -- that stopped the runaway-retry bug but
   over-corrected into surrendering to the non-LLM fallback far more
   than necessary. Pacing alone was already sufficient to prevent the
   storm; the hard cooldown was pure downside once pacing existed.)

Requires GROQ_API_KEY in the environment. Nothing in this file will work
until that's set -- see .env.example.
"""
import json
import logging
import os
import threading
import time
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

# Overall per-/chat-request budget for ALL LLM work combined (extraction +
# generation + any retries), leaving real margin under the assignment's
# 30s hard cutoff for retrieval, JSON serialization, and network jitter
# on top of whatever we spend here.
DEFAULT_TOTAL_BUDGET_SECONDS = 22.0
MIN_USEFUL_SECONDS = 1.5   # below this, don't bother starting another attempt
MAX_SINGLE_CALL_SECONDS = 10.0  # cap one attempt so it can't eat the whole remaining budget
MAX_JSON_RETRY_ATTEMPTS = 2     # retries for "model returned invalid JSON", not rate limits

# ---- Process-wide Groq call pacing ----
# Groq's free tier RPM applies at the account level, not per request, so
# this is process-global (protected by a lock -- FastAPI runs sync `def`
# endpoints in a threadpool, meaning concurrent /chat requests really do
# call this module from different threads at once).
#
# This is the ONLY thing that needs to prevent a request storm, and it
# does so unconditionally: every real outgoing call -- first attempt or
# retry, this request or a concurrent one -- funnels through the same
# lock and minimum spacing. That means retries don't need a tight cap of
# their own to stay safe; they can keep trying for as long as the
# request's time budget allows, which is what actually maximizes the
# chance of getting a real Groq answer instead of falling back. We just
# get a little more cautious (wider spacing) for a bit right after a 429,
# since that's a live signal we're close to the ceiling.
MIN_CALL_INTERVAL_SECONDS = 2.2
WIDENED_INTERVAL_AFTER_429_SECONDS = 4.0
RECENT_429_WINDOW_SECONDS = 12.0

# Rate-limit retries are bounded by the deadline in practice; this is a
# generous safety valve, not the real limiter -- it should basically
# never be the thing that stops a retry loop.
RATE_LIMIT_BACKOFF_BASE_SECONDS = 1.5
MAX_RATE_LIMIT_ATTEMPTS = 5

_call_lock = threading.Lock()
_last_call_time = [0.0]
_last_429_time = [0.0]

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    pass


class RateLimitError(LLMError):
    def __init__(self, message: str, retry_after: float | None):
        super().__init__(message)
        self.retry_after = retry_after


def new_deadline(budget_seconds: float = DEFAULT_TOTAL_BUDGET_SECONDS) -> float:
    """Called once per incoming /chat request (router.py), then threaded
    through every LLM call that request makes."""
    return time.time() + budget_seconds


def _respect_pacing():
    """Enforces a minimum spacing between real outgoing Groq calls,
    process-wide, so normal operation doesn't even approach the
    free-tier RPM ceiling regardless of how many /chat requests -- or
    retries within one request -- are in flight. Widens the spacing for
    a short window after a 429 as soft backpressure; never refuses to
    place the call outright, since a stale rate-limit signal shouldn't
    permanently block a call that might well succeed now."""
    with _call_lock:
        now = time.time()
        interval = MIN_CALL_INTERVAL_SECONDS
        if now - _last_429_time[0] < RECENT_429_WINDOW_SECONDS:
            interval = WIDENED_INTERVAL_AFTER_429_SECONDS
        wait = interval - (now - _last_call_time[0])
        if wait > 0:
            time.sleep(wait)
        _last_call_time[0] = time.time()


def _note_rate_limited():
    with _call_lock:
        _last_429_time[0] = time.time()


def _client(timeout: float) -> httpx.Client:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMError(
            "GROQ_API_KEY is not set. Export it before running the service "
            "(see .env.example) -- this file makes no live calls without it."
        )
    return httpx.Client(
        base_url="https://api.groq.com/openai/v1",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )


def _chat_completion(system_prompt: str, user_content: str, temperature: float, timeout: float,
                      response_format_json: bool = True) -> str:
    body = {
        "model": DEFAULT_MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    if response_format_json:
        body["response_format"] = {"type": "json_object"}
    with _client(timeout) as client:
        _respect_pacing()
        try:
            resp = client.post("/chat/completions", json=body)
            resp.raise_for_status()
        except httpx.TimeoutException as e:
            raise LLMError(f"LLM call timed out after {timeout:.1f}s") from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                retry_after_header = e.response.headers.get("retry-after")
                retry_after = float(retry_after_header) if retry_after_header else None
                raise RateLimitError(f"Rate limited: {e.response.text[:200]}", retry_after) from e
            raise LLMError(f"LLM call failed: {e.response.status_code} {e.response.text[:300]}") from e
    return resp.json()["choices"][0]["message"]["content"]


def _remaining(deadline: float) -> float:
    return deadline - time.time()


def call_structured(system_prompt: str, user_content: str, schema: type[T], temperature: float = 0.2,
                     deadline: float | None = None) -> T:
    """Calls the LLM expecting JSON matching `schema`, validates it, and
    retries (with the validation error fed back to the model) on bad
    JSON, or with backoff on a 429 rate limit -- for as long as the time
    budget before `deadline` allows. Rate-limit retries are deliberately
    not capped tightly: pacing (see _respect_pacing) is what keeps Groq
    call volume safe, so retries here are free to keep trying for a real
    answer rather than surrendering to the fallback path early. Raises
    LLMError the moment continuing wouldn't fit the budget (or the
    generous MAX_RATE_LIMIT_ATTEMPTS safety valve trips, which should be
    rare); callers are expected to have a non-LLM fallback (see
    router.py)."""
    if deadline is None:
        deadline = new_deadline()

    last_error = None
    json_attempts = 0
    rate_limit_attempts = 0
    while True:
        remaining = _remaining(deadline)
        if remaining < MIN_USEFUL_SECONDS:
            raise LLMError(
                f"Out of time budget ({remaining:.1f}s left) calling for {schema.__name__}; "
                f"last error: {last_error}"
            )
        call_timeout = min(remaining - 0.3, MAX_SINGLE_CALL_SECONDS)  # 0.3s slack for our own overhead

        prompt = system_prompt
        if json_attempts:
            prompt += (
                f"\n\nYour previous response was invalid JSON for the required schema. "
                f"Error: {last_error}\nReturn ONLY valid JSON matching the schema, nothing else."
            )

        try:
            raw = _chat_completion(prompt, user_content, temperature, call_timeout)
        except RateLimitError as e:
            _note_rate_limited()
            rate_limit_attempts += 1
            if rate_limit_attempts >= MAX_RATE_LIMIT_ATTEMPTS:
                logger.warning(f"[llm] Still rate limited for {schema.__name__} after "
                                f"{rate_limit_attempts} attempts -- falling back.")
                raise LLMError(f"Rate limited for {schema.__name__} after {rate_limit_attempts} "
                                f"attempts: {e}") from e
            remaining_after = _remaining(deadline)
            if remaining_after < MIN_USEFUL_SECONDS + 0.2:
                # Genuinely not enough time budget left for another
                # attempt to be worth starting -- distinct from Groq's
                # retry-after just happening to be small, which is a
                # *good* sign (means try again soon) and should be
                # honored, not treated as "give up."
                raise LLMError(f"Rate limited with no time budget left to retry: {e}") from e
            # Honor what Groq actually told us if it told us anything;
            # otherwise back off a bit more with each consecutive hit on
            # this specific call.
            backoff = e.retry_after if e.retry_after is not None else RATE_LIMIT_BACKOFF_BASE_SECONDS * rate_limit_attempts
            wait = max(0.0, min(backoff, remaining_after - MIN_USEFUL_SECONDS))
            logger.info(f"[llm] 429 for {schema.__name__} (attempt {rate_limit_attempts}) -- "
                        f"retrying in {wait:.1f}s.")
            time.sleep(wait)
            last_error = f"rate limited, attempt {rate_limit_attempts}"
            continue

        if rate_limit_attempts:
            logger.info(f"[llm] {schema.__name__} succeeded after {rate_limit_attempts} rate-limit retr"
                        f"{'y' if rate_limit_attempts == 1 else 'ies'}.")

        json_attempts += 1
        try:
            data = json.loads(raw)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
            if json_attempts >= MAX_JSON_RETRY_ATTEMPTS:
                raise LLMError(
                    f"LLM did not return valid {schema.__name__} JSON after "
                    f"{json_attempts} attempts: {last_error}"
                )


def call_text(system_prompt: str, user_content: str, temperature: float = 0.4,
               deadline: float | None = None) -> str:
    """Plain text completion, no JSON constraint. Same deadline-bound,
    keep-trying-until-it's-not-worth-it rate-limit handling as
    call_structured, minus the JSON-shape retry (nothing to validate)."""
    if deadline is None:
        deadline = new_deadline()

    rate_limit_attempts = 0
    while True:
        remaining = _remaining(deadline)
        if remaining < MIN_USEFUL_SECONDS:
            raise LLMError(f"Out of time budget ({remaining:.1f}s left) for call_text")
        call_timeout = min(remaining - 0.3, MAX_SINGLE_CALL_SECONDS)

        try:
            return _chat_completion(system_prompt, user_content, temperature, call_timeout, response_format_json=False)
        except RateLimitError as e:
            _note_rate_limited()
            rate_limit_attempts += 1
            if rate_limit_attempts >= MAX_RATE_LIMIT_ATTEMPTS:
                logger.warning(f"[llm] Still rate limited for call_text after {rate_limit_attempts} "
                                f"attempts -- falling back.")
                raise LLMError(f"Rate limited for call_text after {rate_limit_attempts} attempts: {e}") from e
            remaining_after = _remaining(deadline)
            if remaining_after < MIN_USEFUL_SECONDS + 0.2:
                raise LLMError(f"Rate limited with no time budget left to retry: {e}") from e
            backoff = e.retry_after if e.retry_after is not None else RATE_LIMIT_BACKOFF_BASE_SECONDS * rate_limit_attempts
            wait = max(0.0, min(backoff, remaining_after - MIN_USEFUL_SECONDS))
            logger.info(f"[llm] 429 for call_text (attempt {rate_limit_attempts}) -- retrying in {wait:.1f}s.")
            time.sleep(wait)
            continue