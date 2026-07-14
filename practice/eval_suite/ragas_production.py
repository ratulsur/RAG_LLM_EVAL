"""
RAGAS — PRODUCTION / CI-GRADE EVAL over the runnable Pinecone RAG pipeline.
Supersedes the intro `practice/rag_eval_ragas_deepeval.py`: same mental model,
but exhaustive metrics, a pinned judge, a real dataset built by RUNNING the
pipeline, a regression GATE (thresholds -> pass/fail -> nonzero exit), and a
per-sample failure export for debugging.

Reference pipeline under test:  practice/rag_langchain_pinecone.py
    OpenAIEmbeddings(text-embedding-3-small) -> Pinecone(customer-orders-rag)
    -> ChatOpenAI(gpt-4o-mini) via an LCEL chain over 16 customer orders.

============================================================================
THE MENTAL MODEL YOU WALK IN WITH (say this in the room)
============================================================================
Every RAG failure is a RETRIEVAL failure or a GENERATION failure. The metrics
split cleanly on that line, and that split is how you localize a regression:

    RETRIEVAL quality                 GENERATION quality
    -----------------                 ------------------
    context_precision (ranking)       faithfulness  (grounded? -> hallucination)
    context_recall    (coverage)      answer_relevancy (on-topic?)
    context_entity_recall (entities)  answer_correctness (right vs gold)
    noise_sensitivity (robustness)    answer_similarity (semantic closeness)

Faithfulness is your HALLUCINATION metric: hallucination_rate ~= 1 - faithfulness.
That is exactly your Synapse GROUNDING GRADER formalized — it scores each drafted
section against its cited sources and gates pass/fail, i.e. a per-section
faithfulness check driving a revision loop. context_recall failing is your Synapse
SOURCE GRADER territory — the retriever never fetched the evidence, so no prompt
trick saves you; you re-query / widen retrieval instead.

WHAT NEEDS GROUND TRUTH (a real deployment insight — state it unprompted):
    faithfulness, answer_relevancy, noise_sensitivity ......... NO labels needed
    context_precision, context_recall, context_entity_recall,
    answer_correctness, answer_similarity .................... need ground_truth
=> faithfulness + relevance run ONLINE on live traffic with no labels (see
   llm_optimization/03_llm_monitoring_observability.py); precision/recall/
   correctness need a labelled gold set and run OFFLINE in CI (this file).

============================================================================
WHY PIN THE JUDGE (the senior detail interviewers dig for)
============================================================================
RAGAS metrics are LLM-as-judge: a model decomposes text into atomic claims and
checks entailment. If the judge model/temperature drifts between runs, your
scores drift too and week-over-week comparisons become meaningless — you can no
longer tell a real regression from judge noise. So we PIN:
    judge LLM  = ChatOpenAI(gpt-4o-mini, temperature=0)   (deterministic-ish)
    judge emb  = OpenAIEmbeddings(text-embedding-3-small) (relevancy similarity)
and wrap them in ragas' Langchain wrappers so the SAME judge scores every run.
Pin the judge version in your eval config the way you pin a dependency.

RUN
    # real eval against the live Pinecone pipeline + OpenAI judge (needs keys):
    python practice/eval_suite/ragas_production.py --live
    # CI gate (nonzero exit on threshold breach), machine-readable failure export:
    python practice/eval_suite/ragas_production.py --live --gate
    # offline lexical stub — no keys, no network, just the metric SHAPE:
    python practice/eval_suite/ragas_production.py

DEPS: ragas (probed: 0.4.3), datasets, langchain-openai — all already installed.
      Metrics are probed at runtime and any not present in the installed ragas
      are skipped with a printed note, so this file survives a version bump.
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")  # ragas 0.4.x emits deprecation noise on legacy metric imports

# Make the reference pipeline importable from this subfolder.
PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))

FAILURE_EXPORT = Path(__file__).resolve().parent / "ragas_failures.json"


# ===========================================================================
# GOLD SET — the labelled eval dataset (question -> human-authored answer).
# ===========================================================================
# Every ground_truth below is traceable to practice/data/customer_orders.csv —
# same discipline as the master-resume rule: never invent a fact to score against.
# We deliberately mix: single-fact lookups, an aggregation (returns), a
# multi-hop-ish status question, and a numeric total — so the metrics exercise
# different failure modes.
ORDERS_GOLD: list[tuple[str, str]] = [
    (
        "What happened with order 1011?",
        "Order 1011 (Wireless Mouse) was returned; the customer reported three "
        "units stopped working within a week and it is under investigation for a "
        "possible batch defect.",
    ),
    (
        "Which orders were returned, and why?",
        "Order 1011 (Wireless Mouse) was returned for a suspected batch defect, "
        "and order 1016 (Ergonomic Chair) was returned because the height was "
        "unsuitable with a refund pending inspection.",
    ),
    (
        "What is the status of order 1003?",
        "Order 1003 is Processing; the 27-inch Monitor is on backorder with a "
        "restock expected in about two weeks.",
    ),
    (
        "Why was order 1004 cancelled and was the customer refunded?",
        "Order 1004 (USB-C Hub) was cancelled because the customer changed their "
        "mind before shipment; a full refund was issued to the original card "
        "within 24 hours.",
    ),
    (
        "What was the total value of order 1013 and what did it contain?",
        "Order 1013 totalled $499.00 for two 27-inch Monitors for a dual-monitor "
        "home-office setup.",
    ),
    (
        "Did order 1007 have any issues?",
        "Yes; one Webcam unit in order 1007 had a blurry lens and the customer "
        "accepted a partial refund instead of a replacement.",
    ),
]


# ===========================================================================
# EVAL SAMPLE + dataset builder (run the pipeline to capture REAL behavior).
# ===========================================================================
@dataclass
class EvalSample:
    question: str
    answer: str            # what the pipeline actually generated
    contexts: list[str]    # the chunks the retriever actually returned
    ground_truth: str      # human label


def build_orders_pipeline(top_k: int = 6):
    """(chain, retriever) from the runnable Pinecone harness. Requires the index
    to have been ingested already (it has — 16 orders)."""
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


def build_eval_samples(chain, retriever, gold: list[tuple[str, str]]) -> list[EvalSample]:
    """Run the pipeline over each gold question and capture the SAME contexts the
    generator saw. Capturing contexts from the retriever (not re-retrieving later
    with different params) is what makes faithfulness/precision reflect the real
    deployed run — a subtle correctness point interviewers probe."""
    samples: list[EvalSample] = []
    for question, gt in gold:
        docs = retriever.invoke(question)
        answer = chain.invoke(question)
        samples.append(
            EvalSample(question, answer, [d.page_content for d in docs], gt)
        )
    return samples


# ===========================================================================
# METRIC REGISTRY — probe the installed ragas and take whatever it exposes.
# ===========================================================================
# The task: "use whatever the installed ragas version exposes — probe it and
# adapt; guard any metric not present." ragas 0.4.x moved metrics to
# `ragas.metrics.collections` and deprecated the legacy lowercase singletons,
# but the singletons still resolve. We resolve each metric defensively so a
# version bump degrades gracefully (skip + note) instead of crashing the gate.

# Per-metric regression thresholds = the CI GATE. THE SENIOR MOVE: calibrate these
# against a KNOWN-GOOD baseline run (a bit below observed, leaving headroom for
# judge noise), NOT arbitrary round numbers — a gate that false-alarms on the good
# pipeline gets muted, and a muted gate catches nothing. Observed on this pipeline
# (gpt-4o-mini judge, k=6): faithfulness 1.0, answer_relevancy 0.86, ctx_precision
# 0.61, ctx_recall 1.0, entity_recall 0.55, correctness 0.71, similarity 0.82,
# noise 0.20. Thresholds sit just under those. Tune per use case — a compliance
# doc-QA system runs faithfulness >= 0.95. NOTE: ctx_precision 0.61 is genuinely
# mediocre (k=6 pulls ~5 irrelevant orders on single-order questions) -> the real
# fix is lower k or a reranker, a retrieval-side lesson, not a gate tweak.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.75,
    "context_precision": 0.55,       # calibrated below observed 0.61 (weak by design; rerank to raise)
    "context_recall": 0.80,
    "context_entity_recall": 0.45,   # entity recall is strict; observed ~0.55
    "answer_correctness": 0.60,      # factual+semantic blend; mid bar
    "answer_similarity": 0.75,       # embedding cosine; text-embedding-3-small runs ~0.8
    "noise_sensitivity": 0.30,       # LOWER is better (see note) -> treated as a ceiling
}

# noise_sensitivity is INVERTED: it measures how often the answer picks up claims
# from IRRELEVANT retrieved chunks. High = bad. So its gate is a CEILING, not a
# floor. We flag which metrics are "lower-is-better".
LOWER_IS_BETTER = {"noise_sensitivity"}


def resolve_ragas_metrics() -> tuple[list, list[str]]:
    """Return (metric_objects, skipped_names). Tries legacy singletons first,
    falls back to instantiating the collections class, else records a skip."""
    from ragas import metrics as RM

    wanted = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "context_entity_recall",
        "answer_correctness",
        "answer_similarity",
        "noise_sensitivity",
    ]
    resolved, skipped = [], []
    # class-name fallbacks for metrics with no ready-made lowercase singleton
    class_fallbacks = {
        "noise_sensitivity": "NoiseSensitivity",
        "answer_similarity": "SemanticSimilarity",
    }
    for name in wanted:
        obj = getattr(RM, name, None)
        if obj is None:
            cls_name = class_fallbacks.get(name)
            cls = getattr(RM, cls_name, None) if cls_name else None
            if cls is not None:
                try:
                    obj = cls()
                except Exception:
                    obj = None
        if obj is None:
            skipped.append(name)
        else:
            resolved.append(obj)
    return resolved, skipped


def _metric_name(m) -> str:
    return getattr(m, "name", type(m).__name__).lower()


# ===========================================================================
# LIVE RAGAS EVAL (pinned judge) + per-sample failure export + CI gate.
# ===========================================================================
def evaluate_with_ragas(samples: list[EvalSample]):
    """Score the samples with the exhaustive metric set, using a PINNED judge.

    Returns (aggregate: dict[str,float], per_sample_df). ragas' EvaluationResult
    exposes .to_pandas() giving one row per sample per metric — that per-sample
    granularity is what turns a red gate into an actionable bug ('sample 2 failed
    context_recall' -> go look at the retriever for that query)."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    # PIN the judge. Wrap once, reuse for every metric so scoring is consistent.
    judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    judge_emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    metrics, skipped = resolve_ragas_metrics()
    if skipped:
        print(f"[ragas] metrics not in installed version, skipped: {skipped}")
    print(f"[ragas] scoring {len(samples)} samples on: "
          f"{[_metric_name(m) for m in metrics]}")

    # ragas 0.4.x canonical shape: EvaluationDataset of SingleTurnSample.
    ds = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=s.question,
                response=s.answer,
                retrieved_contexts=s.contexts,
                reference=s.ground_truth,
            )
            for s in samples
        ]
    )
    result = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=judge_llm,             # PINNED
        embeddings=judge_emb,      # PINNED
        show_progress=False,
    )
    df = result.to_pandas()
    # ragas suffixes some columns, e.g. 'noise_sensitivity(mode=relevant)'. Strip the
    # '(...)' so column names line up with our threshold keys, and rename in-place so
    # export_failures sees the clean names too.
    df = df.rename(columns={c: c.split("(")[0] for c in df.columns})
    # Aggregate = mean over samples for each metric column present in the frame.
    metric_cols = [c for c in df.columns
                   if c not in ("user_input", "response", "retrieved_contexts",
                                "reference", "retrieved_context_ids", "reference_contexts")]
    aggregate = {c: float(df[c].dropna().mean()) for c in metric_cols
                 if df[c].dropna().shape[0] > 0}
    aggregate["hallucination_rate"] = round(1 - aggregate.get("faithfulness", 0.0), 3)
    return aggregate, df


def export_failures(df, thresholds: dict[str, float]) -> list[dict]:
    """Per-sample failure export: which SAMPLE failed which METRIC. Writes JSON to
    ragas_failures.json so a CI job can attach it to the build and you can jump
    straight to the offending query instead of re-running the whole suite."""
    failures = []
    for i, row in df.iterrows():
        for metric, thr in thresholds.items():
            if metric not in df.columns:
                continue
            score = row[metric]
            if score is None or (isinstance(score, float) and score != score):  # NaN
                continue
            breached = (score > thr) if metric in LOWER_IS_BETTER else (score < thr)
            if breached:
                failures.append({
                    "sample_index": int(i),
                    "question": row.get("user_input", ""),
                    "metric": metric,
                    "score": round(float(score), 3),
                    "threshold": thr,
                    "direction": "ceiling" if metric in LOWER_IS_BETTER else "floor",
                })
    FAILURE_EXPORT.write_text(json.dumps(failures, indent=2))
    return failures


def gate(aggregate: dict[str, float], thresholds: dict[str, float]) -> bool:
    """CI regression gate. Returns True if ALL present metrics clear their bar.
    Prints a PASS/FAIL table. Caller maps False -> sys.exit(1)."""
    print("\n=== RAGAS REGRESSION GATE ===")
    ok = True
    for metric, thr in thresholds.items():
        if metric not in aggregate:
            print(f"  {metric:<24} SKIP (metric not scored)")
            continue
        score = aggregate[metric]
        if metric in LOWER_IS_BETTER:
            passed = score <= thr
            rel = f"<= {thr} (ceiling)"
        else:
            passed = score >= thr
            rel = f">= {thr}"
        ok = ok and passed
        print(f"  {metric:<24} {score:.3f}  {rel:<18} {'PASS' if passed else 'FAIL'}")
    print(f"  {'hallucination_rate':<24} {aggregate.get('hallucination_rate', float('nan')):.3f}  (= 1 - faithfulness)")
    print(f"GATE: {'PASS' if ok else 'FAIL'}")
    return ok


# ===========================================================================
# OFFLINE LEXICAL STUB — runs with NO keys/network. NOT how ragas works
# internally (real metrics use an LLM judge); this is a token-overlap proxy so
# you can see the metric SHAPE and exercise the gate offline.
# ===========================================================================
_STOP = {"the", "is", "a", "an", "of", "to", "and", "in", "with", "for", "at",
         "on", "was", "were", "it", "its", "by", "as", "that", "this"}


def _tokens(text: str) -> set[str]:
    return {w.strip(".,()$").lower() for w in text.split()
            if w.lower() not in _STOP and w.strip(".,()$")}


def evaluate_offline_stub(samples: list[EvalSample]) -> tuple[dict, list[dict]]:
    """Lexical approximations of the metrics + a fake per-sample frame (list of
    dicts) so the gate/export code paths run with no network."""
    rows = []
    for s in samples:
        ctx = _tokens(" ".join(s.contexts))
        ans = _tokens(s.answer)
        gt = _tokens(s.ground_truth)
        q = _tokens(s.question)
        row = {
            "user_input": s.question,
            "faithfulness": len(ans & ctx) / max(len(ans), 1),
            "answer_relevancy": len(ans & q) / max(len(q), 1),
            "context_precision": len(ctx & gt) / max(len(ctx), 1),
            "context_recall": len(gt & ctx) / max(len(gt), 1),
            "context_entity_recall": len(gt & ctx) / max(len(gt), 1),
            "answer_correctness": len(ans & gt) / max(len(ans | gt), 1),
            "answer_similarity": len(ans & gt) / max(len(ans | gt), 1),
            "noise_sensitivity": len(ans - ctx - q) / max(len(ans), 1),  # unsupported frac
        }
        rows.append(row)
    metric_keys = [k for k in rows[0] if k != "user_input"]
    aggregate = {k: round(sum(r[k] for r in rows) / len(rows), 3) for k in metric_keys}
    aggregate["hallucination_rate"] = round(1 - aggregate["faithfulness"], 3)
    return aggregate, rows


def _stub_export_and_gate(rows, aggregate, thresholds):
    """Mirror the live export/gate over the list-of-dict stub frame."""
    class _DF:  # minimal shim so export_failures/gate reuse works
        def __init__(self, rows): self._rows = rows; self.columns = list(rows[0].keys())
        def iterrows(self):
            for i, r in enumerate(self._rows): yield i, r
    failures = export_failures(_DF(rows), thresholds)
    ok = gate(aggregate, thresholds)
    return failures, ok


# ===========================================================================
# ENTRYPOINT
# ===========================================================================
def main() -> None:
    live = "--live" in sys.argv
    do_gate = "--gate" in sys.argv
    thresholds = DEFAULT_THRESHOLDS

    if live:
        print("[mode] LIVE — running the Pinecone pipeline + pinned OpenAI judge.\n")
        chain, retriever = build_orders_pipeline()
        samples = build_eval_samples(chain, retriever, ORDERS_GOLD)
        aggregate, df = evaluate_with_ragas(samples)
        print("\nAGGREGATE:", json.dumps(aggregate, indent=2))
        failures = export_failures(df, thresholds)
    else:
        print("[mode] OFFLINE STUB — lexical proxy, no keys/network. Metric VALUES "
              "are NOT real ragas scores; the SHAPE and the gate are.\n")
        samples = build_eval_samples_offline()
        aggregate, rows = evaluate_offline_stub(samples)
        print("AGGREGATE:", json.dumps(aggregate, indent=2))
        failures, ok = _stub_export_and_gate(rows, aggregate, thresholds)
        print(f"\nExported {len(failures)} per-sample failures -> {FAILURE_EXPORT.name}")
        if do_gate and not ok:
            sys.exit(1)
        return

    ok = gate(aggregate, thresholds)
    print(f"\nExported {len(failures)} per-sample failures -> {FAILURE_EXPORT.name}")
    if do_gate and not ok:
        sys.exit(1)


def build_eval_samples_offline() -> list[EvalSample]:
    """Stand-in samples for the offline path: answer/contexts approximated from
    the gold answer (a real run captures the pipeline's actual output)."""
    return [
        EvalSample(question=q, answer=gt, contexts=[gt], ground_truth=gt)
        for q, gt in ORDERS_GOLD
    ]


if __name__ == "__main__":
    main()
