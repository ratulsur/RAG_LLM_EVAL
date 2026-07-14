"""
RAG WITH LANGCHAIN — Reference (Part 2, study material).
Leads with SEMANTIC CHUNKING, then shows how LangChain composes the pipeline you
hand-wrote in Part 1.

WHY THIS PART EXISTS
--------------------
In Part 1 you built the mechanics by hand so you understand what's underneath.
LangChain's value is NOT magic — it's *composition*: standard interfaces
(Document, Embeddings, VectorStore, Retriever) plus LCEL to wire them into a
chain. Interviewers want to hear BOTH: "I know the mechanics (Part 1) AND I know
the idiomatic framework (Part 2), and I can say exactly which piece maps to
which."

THE MAP (say this in the room):
    Part 1 class          LangChain idiom
    ------------          ---------------
    Chunk                 langchain_core.documents.Document
    FixedSizeChunker      a TextSplitter  (here: SemanticChunker)
    Embedder / .embed     an Embeddings object (.embed_documents / .embed_query)
    FAISSStore            a VectorStore (FAISS) + .as_retriever()
    RAGPipeline.retrieve  the Retriever (Runnable)
    RAGPipeline.answer    an LCEL chain: retriever | prompt | llm | parser

============================================================================
SEMANTIC CHUNKING — the headline topic
============================================================================
WHAT IT IS
  Fixed-size chunking cuts every N words regardless of meaning. Semantic
  chunking instead cuts at *topic boundaries*. Mechanism:
    1. Split the document into sentences.
    2. Embed each sentence (optionally a small sliding window of sentences, to
       smooth noise).
    3. Walk adjacent sentences and compute the distance (1 - cosine sim)
       between neighbours.
    4. A "breakpoint" is where that distance spikes above a threshold — a topic
       shift. Cut there. Sentences between breakpoints form one chunk.
  So chunk *boundaries* are placed by embedding similarity, and chunk *size*
  becomes variable: coherent topics stay whole, unrelated topics get separated.

HOW LANGCHAIN DOES IT (langchain_experimental.text_splitter.SemanticChunker)
  - Takes an Embeddings object (it embeds sentences during splitting — that's
    the cost).
  - breakpoint_threshold_type: how the cut threshold is chosen —
      "percentile"  (default): cut where distance > Xth percentile of all
                     distances (X = breakpoint_threshold_amount, default 95).
      "standard_deviation": cut where distance > mean + k*std.
      "interquartile": robust to outliers via IQR.
      "gradient": for domains where distances change smoothly (e.g. legal).
  - Tuning knob = the percentile/amount. LOWER percentile -> more breakpoints ->
    smaller, more numerous chunks. HIGHER -> fewer, larger chunks.

WHY / WHEN IT BEATS FIXED-SIZE
  - HETEROGENEOUS docs: a policy handbook where SLA, pricing, and security sit
    in one file. Fixed-size splices unrelated topics into one chunk, so a
    retrieval hit drags in noise and dilutes the embedding. Semantic keeps each
    topic clean -> higher precision.
  - TOPIC-SHIFT boundaries: fixed-size cuts mid-argument; semantic respects the
    author's structure, so retrieved chunks read as complete thoughts -> better
    grounding, fewer "answer straddles two chunks" misses.
  - NO ARBITRARY OVERLAP GUESSING: overlap in fixed-size is a blunt fix for
    boundary-straddling. Semantic reduces the need because boundaries land in
    the natural gaps.

COSTS / WHEN NOT TO USE (the senior-level honesty)
  - EMBEDDING COST AT INGEST: it embeds every sentence to decide cuts, on top of
    embedding the final chunks. For a huge corpus that's real money + latency.
    Fixed-size ingest is nearly free.
  - TUNING: the percentile is corpus-dependent. Wrong threshold -> either giant
    chunks (blow the context window, lose precision) or confetti chunks (lose
    context). You must eval it, not eyeball it.
  - VARIABLE CHUNK SIZE: some chunks can exceed the embedder/LLM token budget;
    you often still need a max-size cap on top (hybrid).
  - FAILURE MODES: poor sentence splitting (bad punctuation, tables, code)
    wrecks the boundaries; dense uniform prose has no clear breakpoints so it
    degenerates toward arbitrary cuts anyway. On short/uniform docs, fixed-size
    is cheaper and just as good — don't reach for semantic reflexively.

DECISION RULE to state: "Fixed-size is my baseline. I move to semantic chunking
when documents are long and topically heterogeneous and eval shows retrieval
precision is my bottleneck — and I budget for the extra ingest embedding cost."
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ===========================================================================
# STUB EMBEDDINGS — implements LangChain's Embeddings interface without network.
# ===========================================================================
# LangChain's contract for an embeddings object is just two methods:
#   embed_documents(list[str]) -> list[list[float]]
#   embed_query(str)           -> list[float]
# SemanticChunker, FAISS, and the retriever all program against THIS interface —
# which is exactly why swapping OpenAIEmbeddings for a stub (or for Bedrock,
# HuggingFace, etc.) is a one-line change. We give the stub *deterministic*
# vectors seeded by token hashing so semantically similar sentences land near
# each other and breakpoints are reproducible for study.
class StubEmbeddings:
    """Drop-in Embeddings replacement. In a live round this is
    `from langchain_openai import OpenAIEmbeddings; OpenAIEmbeddings(
    model="text-embedding-3-small")` — identical interface, real vectors."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        # Bag-of-hashed-tokens -> deterministic pseudo-embedding. NOT a real
        # embedding; just enough structure that shared vocabulary => similar
        # vectors, so semantic breakpoints are demonstrable offline.
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            v[hash(tok) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()  # L2-normalize -> cosine via dot

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


# ===========================================================================
# BUILDING THE PIPELINE WITH LANGCHAIN
# ===========================================================================
# All LangChain imports are local to build_* functions so this file stays
# importable/readable even without the packages installed. In a real session
# you'd hoist them to the top.


def build_semantic_splitter(embeddings, breakpoint_percentile: int = 95):
    """Chunker (Part 1)  ->  SemanticChunker (Part 2).

    EXPERT NOTES:
      - The splitter *consumes an embeddings object*. That coupling is the whole
        idea of semantic chunking: chunk boundaries are an embedding decision,
        not a character-count decision.
      - breakpoint_threshold_amount is THE tuning dial. Start at 95th percentile;
        lower it if chunks are too coarse (topics bleeding together), raise it if
        chunks are too fragmented.
      - In production you often wrap this: SemanticChunker to find topic
        boundaries, then a RecursiveCharacterTextSplitter pass to enforce a hard
        max size on any oversized semantic chunk. Mention that hybrid — it's the
        mature answer.
    """
    from langchain_experimental.text_splitter import SemanticChunker

    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=breakpoint_percentile,
    )


def build_vectorstore(documents, embeddings):
    """FAISSStore.add (Part 1)  ->  FAISS.from_documents (Part 2).

    EXPERT NOTES:
      - FAISS.from_documents does in ONE call what you hand-wrote: embed every
        doc, build the index, AND keep the doc<->vector mapping (the docstore).
        The parallel-list bookkeeping you did manually in Part 1 is now internal.
      - LangChain's FAISS defaults to L2 distance. Because our stub vectors are
        normalized, L2 ranking and cosine ranking agree; with real embeddings you
        may pass distance_strategy=MAX_INNER_PRODUCT for true cosine. Know that
        default so you're not surprised by scores.
    """
    from langchain_community.vectorstores import FAISS

    return FAISS.from_documents(documents, embeddings)


def build_rag_chain(retriever, llm):
    """RAGPipeline.answer (Part 1)  ->  an LCEL chain (Part 2).

    LCEL (LangChain Expression Language) composes Runnables with `|`. The chain
    below is the declarative version of your imperative Part 1 `answer()`:

        retrieve -> assemble context -> fill prompt -> call llm -> parse

    EXPERT NOTES:
      - RunnableParallel runs {context: retriever|format, question: passthrough}
        concurrently, so retrieval and query pass-through are one composable step.
      - format_docs is where CONTEXT ASSEMBLY lives — the same job as your
        _assemble_context (token budget + source tags). LCEL doesn't do this for
        you; you still own the grounding discipline. This is a common blind spot.
      - StrOutputParser just unwraps the message to text. The whole chain is
        itself a Runnable -> .invoke / .stream / .batch for free (streaming +
        batching you'd have hand-rolled in Part 1).
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                # Same grounding + abstention instruction as Part 1 — your
                # cheapest hallucination control, framework or not.
                "You are a retrieval-grounded assistant. Answer using ONLY the "
                "context. If the answer isn't in it, reply exactly: "
                "'I don't know based on the provided documents.' Cite [source] tags.\n\n"
                "CONTEXT:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    def format_docs(docs) -> str:
        # CONTEXT ASSEMBLY: best-first (retriever already ranked), tagged for
        # citation + downstream grounding grader. Add a token budget here for a
        # real corpus.
        return "\n\n".join(
            f"[source: {d.metadata.get('doc_id', '?')}#{d.metadata.get('chunk_index', i)}]\n"
            f"{d.page_content}"
            for i, d in enumerate(docs)
        )

    # retriever|format_docs feeds {context}; the raw question passes through.
    setup = RunnableParallel(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    )
    return setup | prompt | llm | StrOutputParser()


def make_documents(text: str, doc_id: str, metadata: dict | None = None):
    """Chunk (Part 1 dataclass) -> Document (Part 2). Document is LangChain's
    universal unit: page_content + metadata. Metadata still carries doc_id for
    citations, tenant for isolation, etc. — same discipline as Part 1."""
    from langchain_core.documents import Document

    md = dict(metadata or {})
    md["doc_id"] = doc_id
    return Document(page_content=text, metadata=md)


# ===========================================================================
# END-TO-END SKETCH (read it; runnable with a fake LLM if packages installed)
# ===========================================================================
if __name__ == "__main__":
    # A tiny heterogeneous doc: three unrelated topics in one file — exactly the
    # case where semantic chunking should separate topics that fixed-size would
    # blend.
    raw = (
        "Our uptime SLA guarantees 99.9% availability measured monthly. "
        "Credits apply if we miss it. "
        "Pricing is tiered: Starter is $49, Growth is $199, Enterprise is custom. "
        "Annual billing gives two months free. "
        "All data is encrypted at rest with AES-256 and in transit with TLS 1.3. "
        "We are SOC 2 Type II certified."
    )

    embeddings = StubEmbeddings()

    # 1. SEMANTIC CHUNKING: split at topic shifts (SLA | pricing | security).
    splitter = build_semantic_splitter(embeddings, breakpoint_percentile=90)
    docs = splitter.create_documents([raw], metadatas=[{"doc_id": "handbook"}])
    for i, d in enumerate(docs):
        d.metadata["chunk_index"] = i

    # 2. Index + retriever (top-k=2).
    vs = build_vectorstore(docs, embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": 2})

    # 3. Fake LLM so this runs with no key. Any Runnable that maps prompt->str
    #    works; in a live round this is ChatOpenAI(model="gpt-4o-mini").
    from langchain_core.runnables import RunnableLambda

    fake_llm = RunnableLambda(lambda prompt_value: prompt_value.to_string()[:400])

    chain = build_rag_chain(retriever, fake_llm)
    print(chain.invoke("What is the uptime SLA?"))
