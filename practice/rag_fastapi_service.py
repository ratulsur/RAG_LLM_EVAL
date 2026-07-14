"""
RAG API ENDPOINTS — Reference (Module File 4, study material).
Exposes the runnable Pinecone RAG pipeline (`rag_langchain_pinecone.py`:
OpenAI embeddings + Pinecone + ChatOpenAI) as a FastAPI service.

This is the ONE file where we allow more structure than usual, because APIs ARE
structure and deployment questions are interview-relevant. Still self-contained:
no separate logging/config/loader modules. One file, a few classes/models.

WHAT AN INTERVIEWER IS CHECKING WITH "wrap your RAG in an API":
  - Do you separate INGEST (write path) from QUERY (read path)? They have
    different latency, auth, and scaling profiles.
  - Do you return CITATIONS/sources, not just an answer string? (grounding +
    auditability — critical for the compliance/finance use cases you target).
  - Do you handle the EMPTY-RETRIEVAL path (the "I don't know" contract) instead
    of hallucinating or 500-ing?
  - Typed contracts (Pydantic) at the boundary.
  - Do you know WHERE streaming, auth, and multi-tenancy slot in? (your Synapse
    FastAPI + SSE experience is the soundbite here.)

MAPS TO YOUR WORK: Synapse ships a FastAPI backend with SSE streaming + a web UI
over a LangGraph pipeline. This is the same shape, one layer simpler (RAG chain
instead of a graph).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field


# ===========================================================================
# PYDANTIC CONTRACTS  (the typed API boundary)
# ===========================================================================
# Request/response models are the contract. They validate input BEFORE it hits
# your pipeline (bad request -> 422 automatically, never reaches the LLM) and
# document the API for free. This is the boundary discipline interviewers want.
class IngestRequest(BaseModel):
    doc_id: str = Field(..., description="Stable id used for citations.")
    text: str = Field(..., min_length=1)
    # tenant would drive multi-tenant isolation (see note in /query).
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    doc_id: str
    chunks_indexed: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(6, ge=1, le=20)  # bound it: unbounded k = context blowup
    # Optional metadata filter for EXHAUSTIVE retrieval on aggregation/filter
    # questions (e.g. {"status": "Returned"}) — the API-layer version of the
    # --filter fix in rag_langchain_pinecone.py. Pure dense top-k under-recalls.
    filter: dict | None = Field(
        default=None, description='e.g. {"status": "Returned"}'
    )


class Source(BaseModel):
    doc_id: str
    chunk_index: int
    score: float | None = None
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    grounded: bool  # False on the empty-retrieval / "I don't know" path


# ===========================================================================
# THE RAG SERVICE  (wraps the file-2 pipeline; injected, not global)
# ===========================================================================
# We wrap the pipeline in a small service object so the endpoints stay thin and
# the pipeline can be swapped/mocked. Built once at startup (embedding model +
# index are expensive to construct) and injected via Depends — never rebuilt
# per request.
# Must match the abstention string the pipeline's prompt enforces so the
# grounded flag below is computed correctly.
ABSTAIN = "I don't know based on the order records."


class RagService:
    def __init__(self, embeddings, breakpoint_percentile: int = 90) -> None:
        # Reuse the runnable Pinecone harness's builders — one source of truth for
        # the splitter, the (persistent) Pinecone store, and the LCEL chain.
        from rag_langchain_pinecone import (
            build_rag_chain,
            ensure_index,
            get_vectorstore,
        )

        ensure_index()  # idempotent: create the serverless index if missing
        self._embeddings = embeddings
        self._build_rag_chain = build_rag_chain
        self._splitter = None  # built lazily on first ingest (embedding cost)
        self._breakpoint_percentile = breakpoint_percentile
        # Pinecone PERSISTS, so the store is available immediately — no lazy
        # FAISS-style creation, no "nothing ingested yet" crash.
        self._vs = get_vectorstore(embeddings)

    def _get_splitter(self):
        if self._splitter is None:
            from rag_langchain_pinecone import (  # local import keeps startup cheap
                EMBED_MODEL,  # noqa: F401  (documents which model the splitter uses)
            )
            from langchain_experimental.text_splitter import SemanticChunker

            self._splitter = SemanticChunker(
                embeddings=self._embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=self._breakpoint_percentile,
            )
        return self._splitter

    # ---- write path ----
    def ingest(self, req: IngestRequest) -> int:
        # API ingest takes arbitrary PROSE documents, so semantic chunking IS the
        # right tool here (unlike the tabular CSV, which we index row-per-order).
        docs = self._get_splitter().create_documents(
            [req.text], metadatas=[{**req.metadata, "doc_id": req.doc_id}]
        )
        for i, d in enumerate(docs):
            d.metadata["chunk_index"] = i
        self._vs.add_documents(docs)  # upsert into the persistent Pinecone index
        return len(docs)

    # ---- read path ----
    def query(self, question: str, top_k: int, meta_filter: dict | None = None) -> QueryResponse:
        search_kwargs: dict = {"k": top_k}
        if meta_filter:
            # {"status": "Returned"} -> Pinecone {"status": {"$eq": "Returned"}}
            search_kwargs["filter"] = {
                k: {"$eq": v} for k, v in meta_filter.items()
            }
        retriever = self._vs.as_retriever(search_kwargs=search_kwargs)
        docs = retriever.invoke(question)

        # EMPTY-RETRIEVAL CONTRACT: if nothing came back, don't call the LLM —
        # abstain. This is the API-layer version of your Synapse source grader
        # catching retrieval failure at the boundary.
        if not docs:
            return QueryResponse(answer=ABSTAIN, sources=[], grounded=False)

        chain = self._build_rag_chain(retriever, self._llm())
        answer = chain.invoke(question)

        sources = [
            Source(
                doc_id=d.metadata.get("doc_id") or d.metadata.get("order_id", "?"),
                chunk_index=d.metadata.get("chunk_index", i),
                snippet=d.page_content[:200],
            )
            for i, d in enumerate(docs)
        ]
        # grounded flag lets the client distinguish a real answer from abstention
        # even when the model phrases uncertainty differently.
        grounded = ABSTAIN not in answer
        return QueryResponse(answer=answer, sources=sources, grounded=grounded)

    def _llm(self):
        from langchain_openai import ChatOpenAI
        from rag_langchain_pinecone import CHAT_MODEL

        return ChatOpenAI(model=CHAT_MODEL, temperature=0)


# ===========================================================================
# APP WIRING + DEPENDENCY INJECTION
# ===========================================================================
# lifespan builds the expensive service ONCE at startup and stashes it on
# app.state. Endpoints pull it via Depends -> testable (override the dep with a
# fake service in tests) and no per-request construction.
@asynccontextmanager
async def lifespan(app: FastAPI):
    from rag_langchain_pinecone import get_embeddings, load_env

    load_env()  # OPENAI_API_KEY + PINECONE_API_KEY from .env
    app.state.rag = RagService(embeddings=get_embeddings())
    yield
    # (teardown: close DB/index handles here in a real service)


app = FastAPI(title="RAG Service", version="1.0", lifespan=lifespan)


def get_rag(app_ref: FastAPI = Depends(lambda: app)) -> RagService:
    return app_ref.state.rag


# ---- endpoints -------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    # Liveness probe for ECS/EKS. A real readiness check would also verify the
    # index/embedder are initialized.
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, rag: RagService = Depends(get_rag)) -> IngestResponse:
    try:
        n = rag.ingest(req)
    except Exception as e:  # ingest can fail on splitter/embedder errors
        raise HTTPException(status_code=500, detail=f"ingest failed: {e}") from e
    return IngestResponse(doc_id=req.doc_id, chunks_indexed=n)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, rag: RagService = Depends(get_rag)) -> QueryResponse:
    return rag.query(req.question, req.top_k, req.filter)


# ===========================================================================
# WHERE THE PRODUCTION CONCERNS SLOT IN  (say these; don't build them)
# ===========================================================================
# STREAMING (SSE): add GET /query/stream returning a StreamingResponse
#   (media_type="text/event-stream") that iterates chain.stream(question),
#   yielding tokens as they arrive. This is exactly your Synapse SSE pattern —
#   first-token latency drops dramatically for long answers.
#
# AUTH + MULTI-TENANCY: a dependency (Depends(verify_token)) validates a bearer
#   token and resolves a tenant_id. Two isolation strategies:
#     - metadata filter: one index, retriever filters on metadata["tenant"] ==
#       tenant_id (cheap, but shared blast radius).
#     - namespace/collection per tenant (Pinecone namespaces / separate FAISS
#       index): hard isolation, the right call for compliance/finance data.
#   You'd inject tenant_id into ingest metadata and into the retriever filter.
#
# OBSERVABILITY: wrap the chain with tracing (LangSmith) + per-request latency /
#   token / cost metrics to CloudWatch; log the retrieved doc_ids for audit.
#
# To run:  uvicorn rag_fastapi_service:app --reload
#          then POST /ingest, POST /query, GET /health.
