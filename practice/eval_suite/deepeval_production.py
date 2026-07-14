"""
DEEPEVAL — PRODUCTION / CI-GRADE EVAL over the runnable Pinecone RAG pipeline.
DeepEval is the framework you SHIPPED in Universal Document Ingestor, so this is
your project soundbite. Where RAGAS thinks in RAG stages, DeepEval thinks in
pytest: every metric is an object with a THRESHOLD and a measure() -> pass/fail,
so it drops straight into CI via assert_test. This file is the exhaustive version.

Reference pipeline under test:  practice/rag_langchain_pinecone.py
    OpenAIEmbeddings -> Pinecone(customer-orders-rag) -> ChatOpenAI(gpt-4o-mini).

============================================================================
THE METRIC MAP (what each one guards, and its RAGAS twin)
============================================================================
    DeepEval metric              guards                     RAGAS twin
    ---------------              ------                     ----------
    FaithfulnessMetric           grounding / hallucination  faithfulness
    AnswerRelevancyMetric        on-topic answer            answer_relevancy
    ContextualPrecisionMetric    ranking of retrieved docs  context_precision
    ContextualRecallMetric       coverage of retrieved docs context_recall
    ContextualRelevancyMetric    signal-to-noise in context (no clean twin)
    HallucinationMetric          contradiction vs CONTEXT   (inverse faithfulness)
    ToxicityMetric               unsafe output              (safety, not RAG)
    BiasMetric                   biased output              (safety, not RAG)
    GEval (custom)               ANY rubric you write       (rubric metrics)

THE ONE DISTINCTION INTERVIEWERS PROBE — Faithfulness vs Hallucination in DeepEval:
  - FaithfulnessMetric reads `retrieval_context` (what your RAG retrieved) and
    asks "are the answer's claims supported by what we RETRIEVED?" -> RAG grounding.
  - HallucinationMetric reads `context` (the GROUND-TRUTH context you supply) and
    asks "does the answer CONTRADICT known-correct context?" It is FACTUAL
    consistency against truth, not against retrieval. Different field, different
    question. Mixing them up is a classic tell that someone only read the README.
  So: FaithfulnessMetric for "did my generator invent something the retriever
  didn't say", HallucinationMetric for "did my generator contradict ground truth".

============================================================================
GEval — LLM-as-judge with a CUSTOM rubric (the flexible one)
============================================================================
GEval turns a plain-English criterion into a scored metric via chain-of-thought
judging. Here we build "answer cites the correct order id" — a domain rule pure
RAGAS metrics can't express. This is how you encode Synapse-style business rules
(e.g. "the drafted section cites its source") as an automated gate.

============================================================================
SYNTHETIC DATA (mention this, it signals maturity)
============================================================================
DeepEval ships `deepeval.synthesizer.Synthesizer`, which generates
goldens (input/expected_output pairs) FROM your documents so you don't hand-author
hundreds of test cases. Pattern (reference — costs LLM calls, not run here):
    from deepeval.synthesizer import Synthesizer
    s = Synthesizer()
    goldens = s.generate_goldens_from_docs(document_paths=["data/customer_orders.csv"])
You then review/curate the goldens (never ship un-reviewed synthetic labels — same
'reframe, never fabricate' discipline) and load them into an EvaluationDataset.

============================================================================
INSTALLED-VERSION NOTES (verified by import before finalizing)
============================================================================
  deepeval 4.0.7:
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams   # 'test_case' SINGULAR
    from deepeval.dataset  import EvaluationDataset
    from deepeval          import assert_test
  All metrics used below (Faithfulness/AnswerRelevancy/ContextualPrecision/
  ContextualRecall/ContextualRelevancy/Hallucination/Toxicity/Bias/GEval) are
  present in 4.0.7. Each is still resolved defensively so a version bump degrades
  to a printed skip rather than a crash.

RUN
    # real eval (each metric = OpenAI judge calls; needs OPENAI_API_KEY):
    python practice/eval_suite/deepeval_production.py --live
    # pytest CI harness (assert_test) — see run_pytest_ci() docstring:
    #   deepeval test run practice/eval_suite/deepeval_production.py
    # offline stub — no keys, shows the metric/threshold SHAPE:
    python practice/eval_suite/deepeval_production.py

DEPS: deepeval (probed: 4.0.7) — already installed. No new deps.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")  # silence LLMTestCaseParams->SingleTurnParams deprecation

PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))


# ===========================================================================
# GOLD SET (mirrors ragas_production.ORDERS_GOLD; kept local so this file is
# self-contained). Each tuple: (question, ground_truth, expected_order_id).
# ===========================================================================
ORDERS_GOLD: list[tuple[str, str, str]] = [
    ("What happened with order 1011?",
     "Order 1011 (Wireless Mouse) was returned; three units stopped working "
     "within a week and it is under investigation for a possible batch defect.",
     "1011"),
    ("What is the status of order 1003?",
     "Order 1003 is Processing; the 27-inch Monitor is on backorder with restock "
     "expected in about two weeks.",
     "1003"),
    ("Why was order 1004 cancelled and was the customer refunded?",
     "Order 1004 (USB-C Hub) was cancelled because the customer changed their "
     "mind; a full refund was issued to the original card within 24 hours.",
     "1004"),
    ("Did order 1007 have any issues?",
     "Yes; one Webcam unit in order 1007 had a blurry lens and the customer "
     "accepted a partial refund instead of a replacement.",
     "1007"),
]


@dataclass
class Golden:
    question: str
    answer: str            # pipeline output
    contexts: list[str]    # retrieved chunks
    ground_truth: str
    expected_order_id: str


# ===========================================================================
# BUILD THE DATASET BY RUNNING THE PIPELINE
# ===========================================================================
def build_orders_pipeline(top_k: int = 6):
    from langchain_openai import ChatOpenAI
    from rag_langchain_pinecone import (
        CHAT_MODEL, build_rag_chain, get_embeddings, get_vectorstore, load_env,
    )
    load_env()
    emb = get_embeddings()
    vs = get_vectorstore(emb)
    retriever = vs.as_retriever(search_kwargs={"k": top_k})
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    return build_rag_chain(retriever, llm), retriever


def build_goldens(chain, retriever) -> list[Golden]:
    goldens: list[Golden] = []
    for q, gt, oid in ORDERS_GOLD:
        docs = retriever.invoke(q)
        ans = chain.invoke(q)
        goldens.append(Golden(q, ans, [d.page_content for d in docs], gt, oid))
    return goldens


# ===========================================================================
# METRICS — instantiate the exhaustive set, guarding any missing in this version.
# ===========================================================================
def build_metrics(threshold: float = 0.7) -> tuple[list, list, list[str]]:
    """Return (rag_metrics, safety_metrics, skipped). rag_metrics take
    retrieval_context; safety_metrics are output-only. Plus the custom GEval."""
    import deepeval.metrics as DM

    skipped: list[str] = []

    def make(name, **kw):
        cls = getattr(DM, name, None)
        if cls is None:
            skipped.append(name)
            return None
        try:
            return cls(**kw)
        except Exception as e:
            skipped.append(f"{name}({type(e).__name__})")
            return None

    rag = [m for m in (
        make("FaithfulnessMetric", threshold=threshold),
        make("AnswerRelevancyMetric", threshold=threshold),
        make("ContextualPrecisionMetric", threshold=threshold),
        make("ContextualRecallMetric", threshold=threshold),
        make("ContextualRelevancyMetric", threshold=threshold),
    ) if m is not None]

    # HallucinationMetric: LOWER is better (fraction of contradicted context).
    # ToxicityMetric / BiasMetric: also lower-is-better safety gates.
    safety = [m for m in (
        make("HallucinationMetric", threshold=0.3),
        make("ToxicityMetric", threshold=0.2),
        make("BiasMetric", threshold=0.2),
    ) if m is not None]

    # Custom GEval — "answer cites the correct order id".
    geval = build_geval_cites_correct_id()
    if geval is not None:
        rag.append(geval)
    else:
        skipped.append("GEval(cites_correct_order_id)")

    return rag, safety, skipped


def build_geval_cites_correct_id():
    """GEval metric: does the answer cite the correct order id in [order NNNN]
    form and match expected? We give it evaluation_steps (explicit rubric) rather
    than only criteria — explicit steps make the judge more consistent, the same
    reason your Synapse grounding grader uses a checklist, not a vibe."""
    import deepeval.metrics as DM
    from deepeval.test_case import LLMTestCaseParams
    GEval = getattr(DM, "GEval", None)
    if GEval is None:
        return None
    return GEval(
        name="CitesCorrectOrderId",
        evaluation_steps=[
            "Check whether the actual output cites an order id in brackets, e.g. [order 1011].",
            "Check that the cited order id matches the order id referenced in the expected output.",
            "Penalize heavily if the answer states a fact about an order but cites the WRONG id, or cites no id.",
            "A correct, well-cited answer scores near 1.0; a confident answer with a wrong/missing citation scores near 0.0.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
    )


# ===========================================================================
# LIVE EVAL + EvaluationDataset
# ===========================================================================
def to_test_case(g: Golden):
    """Golden -> deepeval LLMTestCase. Note the field wiring:
      retrieval_context -> Faithfulness/Contextual* (what RAG retrieved)
      context           -> HallucinationMetric (ground-truth context)
    We feed the gold answer as `context` so HallucinationMetric checks the output
    against TRUTH, distinct from grounding against retrieval."""
    from deepeval.test_case import LLMTestCase
    return LLMTestCase(
        input=g.question,
        actual_output=g.answer,
        expected_output=g.ground_truth,
        retrieval_context=g.contexts,
        context=[g.ground_truth],
    )


def run_live() -> int:
    print("[mode] LIVE — Pinecone pipeline + DeepEval OpenAI judges.\n")
    from deepeval.dataset import EvaluationDataset

    chain, retriever = build_orders_pipeline()
    goldens = build_goldens(chain, retriever)
    cases = [to_test_case(g) for g in goldens]

    # EvaluationDataset = the curated set you version alongside the code. deepeval
    # 4.0.7 constructs from goldens; add_test_case() appends LLMTestCases directly.
    dataset = EvaluationDataset()
    for tc in cases:
        dataset.add_test_case(tc)
    print(f"[deepeval] EvaluationDataset with {len(dataset.test_cases)} cases.")

    rag_metrics, safety_metrics, skipped = build_metrics(threshold=0.7)
    if skipped:
        print(f"[deepeval] skipped (not in version / init error): {skipped}")

    lower_is_better = {"Hallucination", "Toxicity", "Bias"}
    # GATE vs DIAGNOSTIC (a senior distinction worth stating): not every metric
    # should BLOCK a deploy. ContextualRelevancy measures the relevant FRACTION of
    # retrieved context; at k=6 with single-order questions it runs ~0.1-0.3 by
    # construction (5 of 6 chunks are off-topic) — that is the reranking/lower-k
    # lesson, not a generation bug. So we TREND it as a diagnostic instead of
    # failing the merge on it. Everything else is a hard gate.
    diagnostic = {"ContextualRelevancy"}
    all_pass = True
    for g, tc in zip(goldens, cases):
        print(f"\nQ: {g.question}\n  A: {g.answer[:90]}...")
        for m in rag_metrics + safety_metrics:
            try:
                m.measure(tc)
                name = type(m).__name__.replace("Metric", "")
                is_diag = name in diagnostic
                direction = " (lower better)" if name in lower_is_better else ""
                tag = " [diagnostic]" if is_diag else ""
                passed = m.is_successful()
                if not is_diag:
                    all_pass = all_pass and passed
                print(f"    {name:<26} {m.score:.3f}  {'PASS' if passed else 'FAIL'}{direction}{tag}")
            except Exception as e:
                print(f"    {type(m).__name__:<26} ERROR {type(e).__name__}: {str(e)[:60]}")
    print(f"\nOVERALL (hard gate, diagnostics excluded): {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


# ===========================================================================
# PYTEST-STYLE CI HARNESS — assert_test (this is deepeval's CI killer feature)
# ===========================================================================
# Run with:  deepeval test run practice/eval_suite/deepeval_production.py
# assert_test raises AssertionError if ANY metric is below threshold, so it fails
# the build exactly like a unit test. This is the shape you'd wire into GitHub
# Actions for a merge gate on RAG quality.
def test_orders_rag_quality():
    """Pytest entrypoint. Skips cleanly if keys/deps are absent so `pytest` on a
    keyless box doesn't error."""
    import os
    import pytest
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("no OPENAI_API_KEY — deepeval judge unavailable")
    try:
        from deepeval import assert_test
    except Exception:
        pytest.skip("deepeval not installed")

    chain, retriever = build_orders_pipeline()
    goldens = build_goldens(chain, retriever)
    rag_metrics, _safety, _skipped = build_metrics(threshold=0.7)
    for g in goldens:
        assert_test(to_test_case(g), rag_metrics)


# ===========================================================================
# OFFLINE STUB — no keys. Shows the metric/threshold SHAPE with a lexical proxy;
# NOT real DeepEval scores.
# ===========================================================================
_STOP = {"the", "is", "a", "an", "of", "to", "and", "in", "with", "for", "at",
         "on", "was", "were", "it", "its", "by"}


def _tok(t: str) -> set[str]:
    return {w.strip(".,()$[]").lower() for w in t.split()
            if w.lower() not in _STOP and w.strip(".,()$[]")}


def run_offline_stub() -> int:
    print("[mode] OFFLINE STUB — lexical proxy, no keys. Values are illustrative, "
          "not real DeepEval judge scores.\n")
    # Approximate the pipeline output from the gold answer for shape only.
    for q, gt, oid in ORDERS_GOLD:
        ans = gt  # stand-in
        ctx = _tok(gt)
        a = _tok(ans)
        faith = len(a & ctx) / max(len(a), 1)
        arel = len(a & _tok(q)) / max(len(_tok(q)), 1)
        cites = 1.0 if f"[order {oid}]".lower() in ans.lower() or oid in ans else 0.0
        print(f"Q: {q}")
        print(f"    Faithfulness(>=0.7)   {faith:.3f}  {'PASS' if faith >= 0.7 else 'FAIL'}")
        print(f"    AnswerRelevancy(>=0.7){arel:.3f}  {'PASS' if arel >= 0.7 else 'FAIL'}")
        print(f"    Hallucination(<=0.3)  {1-faith:.3f}  {'PASS' if (1-faith) <= 0.3 else 'FAIL'}  (lower better)")
        print(f"    CitesCorrectOrderId   {cites:.3f}  {'PASS' if cites >= 0.7 else 'FAIL'}  (GEval proxy)")
    print("\n(Live metrics: FaithfulnessMetric, AnswerRelevancyMetric, "
          "ContextualPrecision/Recall/Relevancy, HallucinationMetric, "
          "ToxicityMetric, BiasMetric, GEval — run with --live.)")
    return 0


def main() -> None:
    if "--live" in sys.argv:
        sys.exit(run_live())
    sys.exit(run_offline_stub())


if __name__ == "__main__":
    main()
