"""
Internal schemas for the two LLM calls the router makes. These are NOT
the public API contract (that's schemas.py) -- they're how the LLM
communicates structured judgments back to our Python router logic.
"""
from pydantic import BaseModel, Field


class ExtractedRequirements(BaseModel):
    role_or_context: str | None = None
    test_types_wanted: list[str] = Field(default_factory=list)  # letter codes, e.g. ["P"]
    job_level: str | None = None
    remote_required: bool = False
    adaptive_required: bool = False
    max_duration_minutes: int | None = None
    language: str | None = None


class ExtractionResult(BaseModel):
    in_scope: bool
    is_injection: bool = False
    refusal_reason: str | None = None
    requirements: ExtractedRequirements = Field(default_factory=ExtractedRequirements)
    missing_critical_info: bool = False
    wants_recommendation_now: bool = False
    compare_targets: list[str] = Field(default_factory=list)
    search_query: str = ""
    clarifying_question: str | None = None  # only used if missing_critical_info
    is_closing_signal: bool = False  # latest user message reads like acceptance/confirmation
    direct_reply: str | None = None  # see build_extraction_user_content docstring: lets
    # clarify/refuse skip the second (generation) LLM call entirely -- those two actions
    # don't need retrieved candidates, so there's nothing a second call adds except latency
    # and another chance to get rate limited. Null for recommend/compare/anything the
    # router doesn't yet know is heading toward clarify or refuse.


class GenerationResult(BaseModel):
    reply: str
    selected_indices: list[int] = Field(default_factory=list)  # into the candidate list passed in
    end_of_conversation: bool = False