"""
Request/response models -- kept intentionally minimal and exactly
matched to the assignment's spec, since "the schema is non-negotiable."

POST /chat
  request:  {"messages": [{"role": "user"|"assistant", "content": str}, ...]}
  response: {"reply": str, "recommendations": [{"name","url","test_type"}],
             "end_of_conversation": bool}

Design choices worth flagging (documented here, not just in my head):
- `recommendations` defaults to [] (not None). The spec says empty when
  gathering context or refusing -- an empty list satisfies that and is
  strictly easier for a client to iterate over than a nullable field.
- `test_type` is a single string even for multi-category items, e.g.
  "K, P" -- joined in a fixed canonical order (see
  scripts/normalize_catalog.py) since the field is typed as a string in
  the spec, not a list.
- Role is constrained to exactly "user"/"assistant" via Literal, so a
  malformed request fails fast with a clear 422 instead of propagating a
  bad role string into a prompt somewhere downstream.
"""
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
