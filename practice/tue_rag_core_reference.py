"""
RAG CORE PIPELINE — Reference implementation (study material, Tuesday).

Goal of this file: show the *mechanics* of RAG with no LangChain, no framework
magic. Every step a framework hides — chunking, embedding, similarity search,
context assembly, prompt construction — is explicit here so you can explain it
in an interview in 30 seconds.

Architecture (the shape interviewers expect from an OOP RAG round):

    Document ──> Chunker(ABC) ──> [Chunk]
                                    │
                                    ▼
                              Embedder(ABC) ──> vectors
                                    │
                                    ▼
                              VectorStore(ABC) ──> add / search
                                    │
              query ──> Embedder ──┘
                                    ▼
                              RAGPipeline (orchestrator)
                                  ├─ retrieve top-k chunks
                                  ├─ assemble context (with token budget)
                                  └─ build grounded prompt ──> LLM ──> answer

The four abstractions are independent and swappable. That is the whole point of
the ABC design: in the room you say "I can swap FixedSizeChunker for a
semantic chunker, or FAISS for Pinecone, without touching the orchestrator."
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
# A Chunk is the atomic retrievable unit. It carries its text, its source
# metadata, and (once embedded) its vector. Keeping metadata on the chunk is
# what lets you do citations, source grading, and multi-tenant filtering later
# — it is not decoration. In Synapse, this metadata is what the grounding
# grader cites a section against.
@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)
    embedding: np.ndarray | None = None  # filled in by the Embedder


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # similarity score; higher = more relevant (cosine here)


# ---------------------------------------------------------------------------
# 1. Chunker
# ---------------------------------------------------------------------------
class Chunker(ABC):
    """Splits a raw document into retrievable units.

    INTERVIEW POINT — why chunking even exists:
      - Embedding models have a max context; you cannot embed a 50-page PDF as
        one vector and expect useful retrieval.
      - Retrieval granularity = answer granularity. One huge vector retrieves
        the whole doc (no precision); too-small chunks lose context (the answer
        spans chunks). Chunk size is a precision/recall dial.
    """

    @abstractmethod
    def split(self, text: str, doc_id: str, metadata: dict | None = None) -> list[Chunk]:
        ...


class FixedSizeChunker(Chunker):
    """Fixed-size sliding window with overlap, measured in *words*.

    EXPERT DECISIONS TO ARTICULATE:
      1. OVERLAP. The window steps forward by (chunk_size - overlap), so
         adjacent chunks share `overlap` words. This is the single most
         important RAG-quality knob most juniors miss: without overlap, a fact
         that straddles a boundary ("...the SLA is" | "99.9% uptime...") is
         split across two chunks and retrieval recall craters. Typical: 10-20%
         overlap.
      2. WORD-BASED, NOT CHAR-BASED. Words approximate tokens far better than
         characters, so chunk size stays roughly aligned to the embedder's
         token budget. In production you'd use the model's real tokenizer
         (tiktoken); words are the pragmatic from-scratch proxy.
      3. WHY FIXED-SIZE AT ALL. It's the robust baseline: deterministic, cheap,
         no dependency on document structure. You upgrade to recursive /
         semantic / parent-document chunking ONLY when you can name the failure
         it fixes (that's Wednesday). Don't reach for clever chunking first.
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 40) -> None:
        if overlap >= chunk_size:
            # A guard worth stating out loud: overlap >= size means the window
            # never advances -> infinite/duplicate chunks. Cheap invariant,
            # saves you from an embarrassing live bug.
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str, doc_id: str, metadata: dict | None = None) -> list[Chunk]:
        words = text.split()
        if not words:
            return []

        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        idx = 0
        for start in range(0, len(words), step):
            window = words[start : start + self.chunk_size]
            if not window:
                break
            chunks.append(
                Chunk(
                    text=" ".join(window),
                    doc_id=doc_id,
                    chunk_index=idx,
                    metadata=dict(metadata or {}),
                )
            )
            idx += 1
            # Stop once the window has consumed the tail; prevents a final
            # tiny duplicate chunk when start + chunk_size overshoots length.
            if start + self.chunk_size >= len(words):
                break
        return chunks


# ---------------------------------------------------------------------------
# 2. Embedder
# ---------------------------------------------------------------------------
class Embedder(ABC):
    """Turns text into dense vectors. Same model MUST embed both documents and
    queries — mixing models means the vectors live in different spaces and
    cosine similarity is meaningless. State this; it's a classic gotcha."""

    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array, one row per input text."""
        ...


class OpenAIEmbedder(Embedder):
    """Wraps OpenAI's embedding endpoint.

    EXPERT DECISIONS:
      1. BATCHING. We send texts in batches, not one HTTP call per chunk. For a
         10k-chunk corpus, per-chunk calls = 10k round trips = dead pipeline.
         Batching is the difference between a toy and something that ingests a
         real corpus.
      2. MODEL CHOICE. text-embedding-3-small: cheap, 1536-dim, strong recall.
         Go to -3-large only when eval shows small is the bottleneck — don't
         pay 6x by default.
      3. NORMALIZATION. We L2-normalize vectors so an inner-product index gives
         cosine similarity directly (see FAISSStore). Cosine, not raw dot
         product, because we care about *direction* (semantic meaning), not
         magnitude (which tracks text length).

    NOTE: the import + client live inside __init__ so this file is readable
    without the SDK installed. In a live round you'd lift the import to the top.
    """

    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 128) -> None:
        from openai import OpenAI  # local import: keeps the module importable offline

        self._client = OpenAI()
        self._model = model
        self._batch_size = batch_size
        self._dim = 1536  # text-embedding-3-small

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = list(texts[i : i + self._batch_size])
            resp = self._client.embeddings.create(model=self._model, input=batch)
            # The API guarantees response order matches input order — relied on
            # here to keep vectors aligned with their chunks.
            vectors.extend(d.embedding for d in resp.data)
        arr = np.asarray(vectors, dtype=np.float32)
        return self._l2_normalize(arr)

    @staticmethod
    def _l2_normalize(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12  # guard against divide-by-zero on empty text
        return arr / norms


# ---------------------------------------------------------------------------
# 3. VectorStore
# ---------------------------------------------------------------------------
class VectorStore(ABC):
    """Holds vectors + their chunks and answers nearest-neighbour queries.

    The store owns the chunk<->vector mapping. The orchestrator never touches
    raw vectors after ingestion — it asks the store "give me the k most similar
    chunks to this query vector." That separation is what makes FAISS -> Pinecone
    a one-class swap."""

    @abstractmethod
    def add(self, chunks: Sequence[Chunk]) -> None:
        ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        ...


class FAISSStore(VectorStore):
    """In-memory FAISS index.

    EXPERT DECISIONS:
      1. INDEX TYPE = IndexFlatIP (exact, inner product). Because vectors are
         L2-normalized, inner product == cosine similarity. Flat = brute force,
         exact recall, perfect for <~1M vectors. At larger scale you swap to an
         ANN index (IVF, HNSW) and trade a little recall for speed — name that
         tradeoff, don't pretend Flat scales forever.
      2. FAISS STORES VECTORS, NOT TEXT. So we keep a parallel Python list of
         chunks; the index returns integer row positions and we map them back.
         Forgetting this mapping is the #1 from-scratch FAISS bug.
      3. DIM MISMATCH GUARD. The index is created for a fixed dim; pushing a
         differently-sized vector fails loudly. Good — it catches the
         "queried with a different embedder" mistake immediately.
    """

    def __init__(self, dim: int) -> None:
        import faiss  # local import, same reasoning as the embedder

        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)  # IP on normalized vectors = cosine
        self._chunks: list[Chunk] = []        # row i in index  <->  _chunks[i]

    def add(self, chunks: Sequence[Chunk]) -> None:
        vectors = []
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"chunk {c.doc_id}:{c.chunk_index} has no embedding")
            vectors.append(c.embedding)
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape[1] != self._dim:
            raise ValueError(f"expected dim {self._dim}, got {matrix.shape[1]}")
        self._index.add(matrix)
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        # FAISS expects a 2D (n_queries, dim) array.
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        k = min(k, len(self._chunks))  # asking for more than we have -> -1 ids
        if k == 0:
            return []
        scores, ids = self._index.search(q, k)
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:  # FAISS pads with -1 when fewer than k results exist
                continue
            results.append(RetrievedChunk(chunk=self._chunks[idx], score=float(score)))
        return results


# ---------------------------------------------------------------------------
# 4. RAGPipeline (orchestrator)
# ---------------------------------------------------------------------------
class RAGPipeline:
    """Wires the four pieces together and owns the two flows: INGEST and QUERY.

    The orchestrator is deliberately thin. It does NOT know how chunking,
    embedding, or indexing work — it coordinates. That is the design sense an
    interviewer is grading: dependencies injected, single responsibility,
    swappable parts."""

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
        *,
        top_k: int = 4,
        max_context_words: int = 1500,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.max_context_words = max_context_words

    # ---- INGEST ----------------------------------------------------------
    def ingest(self, text: str, doc_id: str, metadata: dict | None = None) -> int:
        """Chunk -> embed -> store. Returns number of chunks indexed."""
        chunks = self.chunker.split(text, doc_id, metadata)
        if not chunks:
            return 0
        vectors = self.embedder.embed([c.text for c in chunks])
        for chunk, vec in zip(chunks, vectors):
            chunk.embedding = vec
        self.store.add(chunks)
        return len(chunks)

    # ---- QUERY -----------------------------------------------------------
    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Embed the query with the SAME embedder, then nearest-neighbour search."""
        q_vec = self.embedder.embed([query])[0]
        return self.store.search(q_vec, self.top_k)

    def _assemble_context(self, retrieved: Sequence[RetrievedChunk]) -> str:
        """Concatenate retrieved chunks into a context block under a word budget.

        EXPERT DECISIONS:
          1. TOKEN BUDGET. You cannot dump unbounded context into the prompt —
             you'll blow the context window and pay for tokens that don't help.
             We greedily add chunks (already ranked best-first) until the budget
             is hit. Real systems budget in tokens; words here for clarity.
          2. CITATIONS. Each chunk is labelled with its source so the model can
             attribute, and so a downstream grounding grader (your Synapse
             pattern) can check each claim against a specific chunk.
          3. ORDER = RELEVANCE. Best chunk first. Models attend more to the
             head/tail of context ("lost in the middle"), so ranking order
             inside the prompt matters, not just which chunks you picked.
        """
        parts: list[str] = []
        used = 0
        for r in retrieved:
            words = len(r.chunk.text.split())
            if used + words > self.max_context_words:
                break
            tag = f"[source: {r.chunk.doc_id}#{r.chunk.chunk_index}]"
            parts.append(f"{tag}\n{r.chunk.text}")
            used += words
        return "\n\n".join(parts)

    def build_prompt(self, query: str, context: str) -> list[dict]:
        """Construct the grounded chat prompt.

        EXPERT DECISIONS:
          1. GROUNDING INSTRUCTION. We explicitly tell the model to answer ONLY
             from context and to say "I don't know" otherwise. This is your
             first and cheapest hallucination control — before any grader. It
             converts the failure mode from "confident wrong answer" to
             "honest abstention," which is what compliance use cases need.
          2. SEPARATION OF CONTEXT AND QUESTION. Context in the system message,
             question in the user message. Clear roles reduce prompt-injection
             surface and make the model treat context as reference, not
             instructions.
        """
        system = (
            "You are a retrieval-grounded assistant. Answer the user's question "
            "using ONLY the context below. If the answer is not in the context, "
            "reply exactly: 'I don't know based on the provided documents.' "
            "Cite sources using their [source: ...] tags.\n\n"
            f"CONTEXT:\n{context}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]

    def answer(self, query: str, llm) -> dict:
        """End-to-end query flow. `llm` is any callable taking messages -> str,
        injected so this stays model-agnostic and unit-testable (you can pass a
        fake LLM in a test — no network needed).

        Returns the answer AND the retrieved chunks, so the caller can show
        citations and so an eval harness can score faithfulness/context-recall.
        """
        retrieved = self.retrieve(query)
        if not retrieved:
            # No evidence -> don't even call the LLM. Retrieval failure handled
            # at the boundary (your source-grader instinct from Synapse).
            return {
                "answer": "I don't know based on the provided documents.",
                "retrieved": [],
            }
        context = self._assemble_context(retrieved)
        messages = self.build_prompt(query, context)
        return {"answer": llm(messages), "retrieved": retrieved}


# ---------------------------------------------------------------------------
# Usage sketch (read it, don't run it — it needs OpenAI + faiss + a key)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def openai_chat(messages: list[dict]) -> str:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return resp.choices[0].message.content

    embedder = OpenAIEmbedder()
    pipeline = RAGPipeline(
        chunker=FixedSizeChunker(chunk_size=200, overlap=40),
        embedder=embedder,
        store=FAISSStore(dim=embedder.dim),
        top_k=4,
    )

    pipeline.ingest(
        text="... your document text ...",
        doc_id="policy_handbook",
        metadata={"tenant": "acme", "section": "SLA"},
    )

    result = pipeline.answer("What is the uptime SLA?", llm=openai_chat)
    print(result["answer"])
    for r in result["retrieved"]:
        print(f"  cited {r.chunk.doc_id}#{r.chunk.chunk_index} (score={r.score:.3f})")
