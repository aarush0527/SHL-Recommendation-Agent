"""
FastAPI entrypoint. Catalog and the TF-IDF index are loaded once at
import time (via the lru_cache'd getters in catalog.py/retrieval.py),
not per-request -- /chat only ever does in-memory lookups plus at most
two LLM calls, nothing touches disk or the network except those calls.
"""
import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.catalog import get_catalog
from app.retrieval import warm_up
from app.router import handle_chat
from app.schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl_recommender")

app = FastAPI(title="SHL Assessment Recommender")


@app.on_event("startup")
def _warm_up():
    t0 = time.time()
    catalog = get_catalog()
    warm_up()
    logger.info(f"Loaded {len(catalog)} catalog items and TF-IDF index in {time.time() - t0:.2f}s")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    return handle_chat(request.messages)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # A single unexpected exception anywhere in the pipeline must not
    # take the evaluator's whole run down mid-conversation -- return a
    # schema-valid, in-scope-sounding fallback instead of a raw 500.
    logger.exception("Unhandled error in request")
    return JSONResponse(
        status_code=200,
        content=ChatResponse(
            reply="Something went wrong on my end -- could you rephrase what role or skills you're looking to assess?",
            recommendations=[],
            end_of_conversation=False,
        ).model_dump(),
    )
