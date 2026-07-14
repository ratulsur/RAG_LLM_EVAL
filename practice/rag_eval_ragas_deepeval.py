"""
RAG EVALUATION — Reference (Module File 3, study material).
Evaluates the runnable LangChain RAG pipeline from `rag_langchain_pinecone.py`
(OpenAI embeddings + Pinecone + ChatOpenAI over the customer-orders CSV). The
metric theory below is vector-store-agnostic — it applied to the FAISS study
file too; only the pipeline we point the sampler at, and the gold set, changed.

WHICH TOOL DO I TEACH?  ->  DeepEval is the PRIMARY practice + CI harness here;
RAGAS is the second lens for reasoning about metric definitions.
  - DeepEval frames every metric as an object with a THRESHOLD and a pass/fail
    (is_successful) — a CI quality gate you can run per commit. You SHIPPED
    DeepEval in Universal Document Ingestor, so it's your project soundbite. This
    file wires the FULL suite (Faithfulness, Answer Relevancy, Contextual
    Precision/Recall/Relevancy, Hallucination, and a custom GEval citation judge)
    plus a pytest-style `assert_test` gate, and the pipeline's `eval` command
    calls straight into it (`python rag_langchain_pinecone.py eval`).
  - RAGAS is purpose-built for RAG and its four metrics (faithfulness, answer
    relevance, context precision, context recall) map 1:1 onto the pipeline
    stages you built — retrieval quality vs generation quality. It's the fastest
    way to reason about "is my RETRIEVER wrong or is my GENERATOR wrong."

THE ONE MENTAL MODEL TO CARRY IN
--------------------------------
Every RAG failure is either a RETRIEVAL failure or a GENERATION failure. The four
metrics split cleanly along that line — this is the sentence that makes you sound
senior:

    RETRIEVAL quality        GENERATION quality
    -----------------        ------------------
    context precision        faithfulness
    context recall           answer relevance

If faithfulness is high but answer relevance is low -> model is grounded but
off-topic. If context recall is low -> retriever never fetched the evidence, so
no prompt trick can save you. Diagnose the stage, then fix the stage.

============================================================================
THE FOUR CORE METRICS  (crisp defs + how each is computed)
============================================================================
Each metric is scored per (question, answer, contexts[, ground_truth]) sample,
then averaged. Most are computed BY AN LLM JUDGE ("LLM-as-judge") that decomposes
text into claims and checks them — so evals cost LLM calls too.

1) FAITHFULNESS  (generation; needs: answer + contexts)
   DEF: of the claims the ANSWER makes, what fraction are supported by the
   retrieved CONTEXT? = |claims entailed by context| / |claims in answer|.
   HOW: judge extracts atomic claims from the answer, then checks each for
   entailment against the context. 1.0 = every claim is grounded.
   => This is your HALLUCINATION metric. hallucination rate ≈ 1 - faithfulness.
   THIS IS YOUR SYNAPSE GROUNDING GRADER, formalized: your grader scores each
   drafted section against its source citations and gates pass/fail — that IS a
   per-section faithfulness check driving a revision loop.

2) ANSWER RELEVANCE  (generation; needs: question + answer)
   DEF: does the answer actually address the QUESTION (not verbose, not evasive,
   not off-topic)? HOW: judge generates N synthetic questions that the answer
   *would* be the answer to, embeds them, and measures mean cosine similarity to
   the ORIGINAL question. High sim = the answer is on-point. Note: this does NOT
   check correctness — only relevance/on-topic-ness.

3) CONTEXT PRECISION  (retrieval; needs: question + contexts + ground_truth)
   DEF: of the retrieved chunks, are the RELEVANT ones ranked at the top?
   (signal-to-noise + ranking quality). HOW: for each retrieved chunk, judge
   whether it's relevant to the ground-truth answer, then compute a rank-weighted
   score (relevant chunks near the top score higher). Low precision => reranking
   or better chunking will help.

4) CONTEXT RECALL  (retrieval; needs: contexts + ground_truth)
   DEF: of the claims in the GROUND-TRUTH answer, what fraction are covered by
   the retrieved context? = |gt claims found in context| / |gt claims|.
   HOW: judge breaks the ground truth into claims, checks each against the
   retrieved context. Low recall => your retriever/chunking is MISSING evidence;
   no generation fix helps. This is the metric that needs ground truth most.

HALLUCINATION RATE: not a separate RAGAS metric — it's the complement of
faithfulness (1 - faithfulness). Say it that way in the room.

WHAT NEEDS GROUND TRUTH?
   faithfulness       : NO  (answer vs context only)
   answer relevance   : NO  (question vs answer only)
   context precision  : YES (needs relevance judged against gt)
   context recall     : YES (needs gt claims to check coverage)
=> You can run faithfulness + relevance online in prod with NO labels; precision
   + recall need a labelled eval set. State this — it's a real deployment insight.

============================================================================
BUILDING AN EVAL DATASET
============================================================================
The unit is a sample: {question, answer, contexts, ground_truth}.
  - question     : the user query
  - answer       : what YOUR pipeline generated (chain.invoke)
  - contexts     : the chunks YOUR retriever returned (list[str])
  - ground_truth : the human-labelled correct answer (for precision/recall)
You generate answer + contexts by RUNNING your file-2 pipeline over each
question; you author ground_truth by hand (or with a strong model + review).
"""

from __future__ import annotations

from dataclasses import dataclass


# ===========================================================================
# EVAL SAMPLE + a tiny gold set
# ===========================================================================
@dataclass
class EvalSample:
    question: str
    answer: str            # produced by the pipeline under test
    contexts: list[str]    # retrieved chunks the pipeline used
    ground_truth: str      # human label


def build_eval_samples_from_pipeline(chain, retriever, gold: list[tuple[str, str]]) -> list[EvalSample]:
    """Run the file-2 pipeline over each (question, ground_truth) pair to capture
    what the system ACTUALLY answered and retrieved.

    EXPERT NOTE: you must record the SAME contexts the generator saw — capture
    them from the retriever, don't re-retrieve differently later, or your
    faithfulness/precision scores won't reflect the real run.
    """
    samples: list[EvalSample] = []
    for question, gt in gold:
        docs = retriever.invoke(question)  # the actual retrieved context
        answer = chain.invoke(question)
        samples.append(
            EvalSample(
                question=question,
                answer=answer,
                contexts=[d.page_content for d in docs],
                ground_truth=gt,
            )
        )
    return samples


# ===========================================================================
# OPTION A — RAGAS  (the primary tool)
# ===========================================================================
def evaluate_with_ragas(samples: list[EvalSample], judge_llm=None, judge_embeddings=None):
    """Run the four core RAGAS metrics.

    EXPERT NOTES:
      - RAGAS needs a JUDGE llm + embeddings (the LLM-as-judge doing claim
        extraction/entailment, embeddings for the answer-relevance similarity).
        In prod that's ChatOpenAI + OpenAIEmbeddings; pass them explicitly so the
        judge model is PINNED — an unpinned judge makes scores drift run-to-run,
        which destroys comparability. Pinning the judge is a senior-level detail.
      - Metric selection is intentional: report all four so you can localize a
        regression to retrieval vs generation.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    ds = Dataset.from_dict(
        {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "ground_truth": [s.ground_truth for s in samples],
        }
    )
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,               # PIN this
        embeddings=judge_embeddings,  # PIN this
    )
    # result is a dict-like of metric -> mean score in [0,1].
    return result


# ===========================================================================
# OPTION B — DeepEval  (the PRIMARY practice + CI harness; Universal Doc soundbite)
# ===========================================================================
# DeepEval's model: each metric is an object with a THRESHOLD and a
# measure(test_case) -> score + is_successful() pass/fail. That threshold model
# IS a CI quality gate ("faithfulness >= 0.7 on the gold set") — exactly what you
# ran in Universal Document Ingestor. We expose the FULL suite so both the
# pipeline's `eval` command and the pytest gate below use it.
#
# DIRECTION: Faithfulness / AnswerRelevancy / Contextual{Precision,Recall,
# Relevancy} / the citation GEval are HIGHER-IS-BETTER (pass if score >=
# threshold). Hallucination is LOWER-IS-BETTER (pass if score <= its own max) —
# each metric's is_successful() encodes its own direction, so the report is
# uniform: {score, passed}.


def _to_test_case(s: EvalSample):
    """EvalSample -> DeepEval LLMTestCase.

    Field wiring is deliberate:
      - retrieval_context = the chunks the retriever returned (Faithfulness /
        Contextual{Precision,Recall,Relevancy} judge against these).
      - context           = the FACTUAL REFERENCE (ground truth). HallucinationMetric
        checks the answer for CONTRADICTION against `context`; feeding it the noisy
        retrieved chunks (many irrelevant) makes it count off-topic chunks as
        hallucinations. The labelled answer is the correct factual anchor."""
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
        input=s.question,
        actual_output=s.answer,
        retrieval_context=s.contexts,
        expected_output=s.ground_truth,
        context=[s.ground_truth],
    )


def build_deepeval_metrics(threshold: float = 0.7):
    """Construct the DeepEval metric suite. Each metric is created defensively so
    a version mismatch skips ONE metric with a note rather than crashing the run.
    Hallucination gets an inverted gate (max acceptable hallucination), and a
    GEval metric adds a RAG-specific custom judge: correct order-id citation."""
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        GEval,
        HallucinationMetric,
    )
    from deepeval.test_case import LLMTestCaseParams

    metrics = []

    def add(factory, label):
        try:
            metrics.append(factory())
        except Exception as e:  # version drift / metric unavailable
            print(f"  [skip] {label}: {e}")

    # Generation quality
    add(lambda: FaithfulnessMetric(threshold=threshold), "Faithfulness")
    add(lambda: AnswerRelevancyMetric(threshold=threshold), "AnswerRelevancy")
    # Retrieval quality
    add(lambda: ContextualPrecisionMetric(threshold=threshold), "ContextualPrecision")
    add(lambda: ContextualRecallMetric(threshold=threshold), "ContextualRecall")
    # NOTE: Contextual Relevancy = signal-to-noise of the retrieved context. It is
    # LOW BY DESIGN at the pipeline's default top_k=6 on single-order questions
    # (5 of 6 chunks are off-topic). Read it as a DIAGNOSTIC that argues for lower
    # k or a reranker — not a broken metric. It's the honest cost of high recall.
    add(lambda: ContextualRelevancyMetric(threshold=threshold), "ContextualRelevancy")
    # Direct hallucination (lower is better -> invert the gate)
    add(lambda: HallucinationMetric(threshold=round(1 - threshold, 2)), "Hallucination")
    # Custom RAG judge: does the answer cite the order id(s) that back its claims?
    add(
        lambda: GEval(
            name="CitationCorrectness",
            criteria=(
                "Given the question, the answer, and the retrieved order records, "
                "decide whether the answer cites the correct [order NNNN] id(s) for "
                "the facts it states. Penalize missing or wrong citations."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            threshold=threshold,
        ),
        "CitationCorrectness(GEval)",
    )
    return metrics


def evaluate_with_deepeval(samples: list[EvalSample], threshold: float = 0.7, metrics=None):
    """Score each sample with the full DeepEval suite. Returns a report:
    list of {question, MetricName: {score, passed}} — the contract the pipeline's
    `eval` command consumes, so enriching the suite enriches that command too."""
    metrics = metrics if metrics is not None else build_deepeval_metrics(threshold)
    report = []
    for s in samples:
        tc = _to_test_case(s)
        row = {"question": s.question}
        for m in metrics:
            m.measure(tc)  # runs the judge
            name = getattr(m, "__name__", None) or type(m).__name__
            row[name] = {"score": m.score, "passed": bool(m.is_successful())}
        report.append(row)
    return report


def build_deepeval_dataset(samples: list[EvalSample]):
    """Bundle samples into a DeepEval EvaluationDataset (batch/CI ergonomics).
    Then: `from deepeval import evaluate; evaluate(dataset, build_deepeval_metrics())`."""
    from deepeval.dataset import EvaluationDataset

    return EvaluationDataset(test_cases=[_to_test_case(s) for s in samples])


def test_orders_pipeline_quality():
    """pytest-discoverable DeepEval GATE. Run either:
        deepeval test run practice/rag_eval_ragas_deepeval.py
        pytest practice/rag_eval_ragas_deepeval.py
    Fails the build if any metric on any gold sample is below threshold. Skips
    (never falsely fails) when live keys / the pipeline aren't available."""
    import os

    from rag_langchain_pinecone import load_env

    load_env()
    if not (os.getenv("OPENAI_API_KEY") and os.getenv("PINECONE_API_KEY")):
        try:
            import pytest

            pytest.skip("live OPENAI/PINECONE keys not set; skipping DeepEval gate")
        except ImportError:
            return
    from deepeval import assert_test

    chain, retriever = build_orders_pipeline()
    samples = build_eval_samples_from_pipeline(chain, retriever, ORDERS_GOLD)
    metrics = build_deepeval_metrics(threshold=0.7)
    for s in samples:
        assert_test(_to_test_case(s), metrics)


# ===========================================================================
# OFFLINE STUB — see the SHAPE of scoring with no keys/network.
# ===========================================================================
# This is NOT how RAGAS works internally (real metrics use an LLM judge); it's a
# lexical proxy so you can run this file offline and see the numbers move. It
# demonstrates the *definitions* concretely.
def _claims(text: str) -> set[str]:
    # crude "claim" proxy = content words
    stop = {"the", "is", "a", "an", "of", "to", "and", "in", "with", "for", "at", "on"}
    return {w.strip(".,").lower() for w in text.split() if w.lower() not in stop}


def evaluate_offline_stub(samples: list[EvalSample]) -> dict:
    """Lexical approximation of the four metrics — for intuition only."""
    faith, arel, cprec, crec = [], [], [], []
    for s in samples:
        ctx = " ".join(s.contexts)
        ctx_claims = _claims(ctx)
        ans_claims = _claims(s.answer)
        gt_claims = _claims(s.ground_truth)

        # faithfulness: answer claims supported by context
        faith.append(len(ans_claims & ctx_claims) / max(len(ans_claims), 1))
        # answer relevance (proxy): answer overlap with question
        q_claims = _claims(s.question)
        arel.append(len(ans_claims & q_claims) / max(len(q_claims), 1))
        # context recall: gt claims covered by context
        crec.append(len(gt_claims & ctx_claims) / max(len(gt_claims), 1))
        # context precision (proxy): fraction of context that is relevant to gt
        cprec.append(len(ctx_claims & gt_claims) / max(len(ctx_claims), 1))

    mean = lambda xs: round(sum(xs) / max(len(xs), 1), 3)
    return {
        "faithfulness": mean(faith),
        "answer_relevancy": mean(arel),
        "context_precision": mean(cprec),
        "context_recall": mean(crec),
        "hallucination_rate": round(1 - mean(faith), 3),  # = 1 - faithfulness
    }


# ===========================================================================
# WIRING THE EVAL TO THE ACTUAL PINECONE PIPELINE
# ===========================================================================
# Gold set for the customer-orders corpus (question -> human-labelled answer).
# Contexts + answers are captured by RUNNING the real pipeline, not hand-filled,
# so the scores reflect the deployed system.
ORDERS_GOLD: list[tuple[str, str]] = [
    (
        "What happened with order 1011?",
        "Order 1011 (Wireless Mouse) was returned; three units failed within a "
        "week and it is under investigation for a possible batch defect.",
    ),
    (
        "Which orders were returned, and why?",
        "Order 1011 (Wireless Mouse, batch-defect investigation) and order 1016 "
        "(Ergonomic Chair, unsuitable height) were returned.",
    ),
    (
        "What is the status of order 1003?",
        "Order 1003 is Processing; the 27-inch Monitor is on backorder with "
        "restock expected in about two weeks.",
    ),
]


def build_orders_pipeline(top_k: int = 6):
    """Build (chain, retriever) from the runnable Pinecone harness so we can
    capture the REAL answers/contexts under eval. Requires the orders index to
    have been ingested (`python rag_langchain_pinecone.py ingest`)."""
    from langchain_openai import ChatOpenAI
    from rag_langchain_pinecone import (
        CHAT_MODEL,
        build_rag_chain,
        get_embeddings,
        get_vectorstore,
        load_env,
    )

    load_env()
    embeddings = get_embeddings()
    vs = get_vectorstore(embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": top_k})
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    return build_rag_chain(retriever, llm), retriever


def run_ragas_over_orders():
    """End-to-end real eval: run the Pinecone pipeline over ORDERS_GOLD, then
    score with RAGAS using a PINNED ChatOpenAI judge + OpenAIEmbeddings."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    chain, retriever = build_orders_pipeline()
    samples = build_eval_samples_from_pipeline(chain, retriever, ORDERS_GOLD)
    return evaluate_with_ragas(
        samples,
        judge_llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        judge_embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    )


def run_deepeval_over_orders(threshold: float = 0.7, top_k: int = 6, gold=None):
    """End-to-end DeepEval: run the Pinecone pipeline over the gold set (defaults
    to ORDERS_GOLD), capture each real answer + retrieved context, then score with
    DeepEval's threshold metrics. Returns (samples, report) so the caller can both
    inspect what the pipeline produced AND the per-metric pass/fail.

    This is the wiring the RAG pipeline's `eval` command calls — DeepEval is the
    default practice harness because its threshold + pass/fail model is exactly a
    CI quality gate (Universal Document Ingestor)."""
    gold = gold or ORDERS_GOLD
    chain, retriever = build_orders_pipeline(top_k=top_k)
    samples = build_eval_samples_from_pipeline(chain, retriever, gold)
    report = evaluate_with_deepeval(samples, threshold=threshold)
    return samples, report


if __name__ == "__main__":
    # Default: offline lexical stub over the orders gold set — no keys/network,
    # just to see the metric SHAPE. Contexts here are hand-filled proxies.
    samples = [
        EvalSample(
            question=q,
            answer=gt,          # stand-in; real run captures the pipeline's answer
            contexts=[gt],      # stand-in; real run captures retrieved chunks
            ground_truth=gt,
        )
        for q, gt in ORDERS_GOLD
    ]
    print(evaluate_offline_stub(samples))
    # Real eval against the live Pinecone pipeline:
    #   DeepEval (primary practice harness):
    #     python rag_langchain_pinecone.py eval        # or: from rag_eval_ragas_deepeval import run_deepeval_over_orders
    #   DeepEval CI gate (pytest / assert_test):
    #     deepeval test run practice/rag_eval_ragas_deepeval.py
    #   RAGAS (second lens):
    #     from rag_eval_ragas_deepeval import run_ragas_over_orders; print(run_ragas_over_orders())
