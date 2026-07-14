"""
LLM OPTIMIZATION 1/3 — PROMPT & OUTPUT OPTIMIZATION for reliability + cost.
Applied to the orders RAG prompt (practice/rag_langchain_pinecone.py). The theme:
you rarely need a bigger model — you need a tighter prompt, a typed output, a
repair loop, and a token budget. These are the cheapest reliability wins and the
ones interviewers use to separate "calls the API" from "runs it in production".

============================================================================
WHAT THIS FILE DRILLS (each is a runnable/measured block)
============================================================================
  1. ZERO-SHOT vs FEW-SHOT     when examples earn their token cost (and when they
                               don't). Measured in tokens.
  2. STRUCTURED OUTPUT         Pydantic schema + with_structured_output so the
                               model returns a typed object, not prose you regex.
  3. VALIDATION + REPAIR       validate against the schema; on failure, feed the
                               error back for ONE repair attempt (bounded).
  4. PROMPT COMPRESSION        strip redundancy from context; measure quality-vs-
                               tokens so you cut fat without cutting evidence.
  5. TOKEN BUDGETING           count tokens (tiktoken) and fit context to a budget
                               by dropping lowest-ranked chunks, not truncating mid-doc.

============================================================================
THE MENTAL MODEL (say this)
============================================================================
Every token in the prompt is paid for on EVERY call and inflates latency. So
prompt optimization is cost optimization. But you cannot cut evidence a grounded
answer needs (that raises hallucination — see eval_suite/hallucination_suite.py).
The job is to maximize ANSWER QUALITY PER TOKEN: cut instructions/duplication,
keep evidence. Structured output + validation then removes the SECOND cost —
re-calls caused by unparseable answers.

TIE TO YOUR WORK: your Synapse grounding grader gates on faithfulness; that same
gate is why structured output matters — a typed {answer, cited_order_ids,
grounded: bool} lets the grader read `grounded` and `cited_order_ids` as fields
instead of parsing prose. Typed output makes the downstream grader deterministic.

RUN
    python practice/llm_optimization/01_prompt_and_output_optimization.py          # offline: token math + compression + validation
    python practice/llm_optimization/01_prompt_and_output_optimization.py --live   # + real structured-output call

DEPS: tiktoken (installed) for exact token counts; falls back to a ~4-chars/token
      estimate if absent (guarded). pydantic (installed). langchain-openai for --live.
      No new deps.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))


# ===========================================================================
# TOKEN COUNTING — exact via tiktoken, else a hand-rolled estimate.
# ===========================================================================
def make_token_counter(model: str = "gpt-4o-mini"):
    """Return count_tokens(text)->int. tiktoken gives exact BPE counts; without it
    we estimate ~4 chars/token (English), which is close enough for budgeting.
    ALWAYS budget in tokens, never characters — that's the unit you're billed in."""
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text)), "tiktoken"
    except Exception:
        return (lambda text: max(1, len(text) // 4)), "estimate(~4 chars/token)"


COUNT, COUNTER_KIND = make_token_counter()


# ===========================================================================
# 1) ZERO-SHOT vs FEW-SHOT — measured in tokens.
# ===========================================================================
ZERO_SHOT_SYSTEM = (
    "You are a customer-orders assistant. Answer using ONLY the context. If the "
    "answer isn't in it, reply exactly: 'I don't know based on the order records.' "
    "Cite the order id(s) like [order 1011]."
)

# Few-shot adds worked examples. They buy FORMAT DISCIPLINE + edge-case behavior
# (e.g. teaching abstention by example), at a recurring token cost paid every call.
FEW_SHOT_EXAMPLES = [
    ("What happened with order 1011?",
     "Order 1011 (Wireless Mouse) was returned; three units failed within a week, "
     "under investigation for a batch defect. [order 1011]"),
    ("What is the capital of France?",
     "I don't know based on the order records."),   # teaches abstention by example
]


def build_few_shot_system() -> str:
    ex = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in FEW_SHOT_EXAMPLES)
    return ZERO_SHOT_SYSTEM + "\n\nEXAMPLES:\n" + ex


def demo_zero_vs_few_shot() -> None:
    zs, fs = ZERO_SHOT_SYSTEM, build_few_shot_system()
    print(f"  zero-shot system prompt : {COUNT(zs):>4} tokens")
    print(f"  few-shot  system prompt : {COUNT(fs):>4} tokens  "
          f"(+{COUNT(fs) - COUNT(zs)} tokens EVERY call)")
    print("  RULE: use few-shot only when zero-shot fails a behavior you can teach "
          "by example (format, abstention). Otherwise the example tokens are pure cost.")


# ===========================================================================
# 2) STRUCTURED OUTPUT — Pydantic schema (typed, not prose).
# ===========================================================================
def order_answer_model():
    """The typed contract the RAG generator should return. A schema turns
    'parse the prose' into 'read a field' and lets the grounding gate be a boolean."""
    from pydantic import BaseModel, Field

    class OrderAnswer(BaseModel):
        answer: str = Field(description="natural-language answer, grounded in context")
        cited_order_ids: list[str] = Field(default_factory=list,
                                            description="order ids the answer relies on")
        grounded: bool = Field(description="True if fully supported by the context")

    return OrderAnswer


# ===========================================================================
# 3) VALIDATION + REPAIR — bounded self-correction.
# ===========================================================================
def validate_answer(raw: str, model_cls) -> tuple[bool, object | str]:
    """Try to coerce raw text -> the schema. Returns (ok, obj_or_error)."""
    from pydantic import ValidationError
    try:
        obj = model_cls.model_validate_json(raw)
        return True, obj
    except ValidationError as e:
        return False, str(e)


def repair_prompt(raw: str, error: str, schema_json: str) -> str:
    """ONE bounded repair attempt: hand the model its own broken output + the
    validation error + the schema, ask for corrected JSON. Bounded because an
    unbounded repair loop is a cost/latency footgun — cap at 1-2 tries then fail
    closed (abstain), never spin."""
    return (
        "Your previous output failed schema validation.\n"
        f"OUTPUT:\n{raw}\n\nERROR:\n{error}\n\nSCHEMA:\n{schema_json}\n\n"
        "Return ONLY corrected JSON matching the schema."
    )


def demo_validation_repair() -> None:
    model_cls = order_answer_model()
    schema_json = json.dumps(model_cls.model_json_schema())
    good = '{"answer":"Order 1011 was returned.","cited_order_ids":["1011"],"grounded":true}'
    bad = '{"answer":"Order 1011 was returned.","cited_order_ids":"1011"}'  # wrong type + missing field
    for label, raw in [("valid", good), ("invalid", bad)]:
        ok, res = validate_answer(raw, model_cls)
        print(f"  {label:<8} -> ok={ok}" + ("" if ok else f"  (would send 1 repair prompt of "
              f"{COUNT(repair_prompt(raw, str(res), schema_json))} tokens)"))


# ===========================================================================
# 4) PROMPT COMPRESSION — cut redundancy, measure quality-vs-tokens.
# ===========================================================================
def compress_context(chunks: list[str], question: str, keep_ratio: float = 0.6) -> list[str]:
    """Cheap extractive compression: keep the sentences whose content-word overlap
    with the QUESTION is highest, drop boilerplate. Real systems use LLMLingua or an
    LLM summarizer; the PRINCIPLE is the same — spend tokens on evidence, not filler.
    Returns compressed chunks. (Guardrail: never compress below the evidence the
    answer needs — measure faithfulness before/after, see eval_suite.)"""
    import re
    def words(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3}

    q_words = words(question)

    def score(sentence: str) -> int:
        return len(q_words & words(sentence))

    # split on sentence-final '.'/';' followed by whitespace+capital — this does NOT
    # break decimals like $249.50 (no space after the dot), which a naive split(".")
    # would shatter into "$249" / "50" and wreck ranking.
    boundary = re.compile(r"(?<=[.;])\s+(?=[A-Z])")

    out = []
    for ch in chunks:
        sents = [s.strip() for s in boundary.split(ch) if s.strip()]
        keep_n = max(1, round(len(sents) * keep_ratio))
        ranked_idx = sorted(range(len(sents)), key=lambda i: score(sents[i]),
                            reverse=True)[:keep_n]
        kept = [sents[i] for i in sorted(ranked_idx)]  # preserve original order
        out.append(" ".join(kept))
    return out


def demo_compression() -> None:
    chunks = [
        "Order 1003 was placed on 2026-01-08 by Wei Chen in region APAC-North. It is "
        "for 1 x 27-inch Monitor (Displays) at $249.50 each, totalling $249.50. Current "
        "status: Processing. Notes: Payment cleared but the item is on backorder. "
        "Expected restock in two weeks. Customer notified by email.",
    ]
    q = "What is the status of order 1003 and when is restock?"
    before = sum(COUNT(c) for c in chunks)
    comp = compress_context(chunks, q, keep_ratio=0.5)
    after = sum(COUNT(c) for c in comp)
    print(f"  context tokens: {before} -> {after}  "
          f"({100*(before-after)//max(before,1)}% cut)")
    print(f"  kept: {comp[0][:120]}...")
    print("  QUALITY CHECK: the restock+status sentences survived -> a faithful "
          "answer is still possible. Verify with faithfulness before shipping a cut.")


# ===========================================================================
# 5) TOKEN BUDGETING — fit retrieved context to a budget by dropping lowest-
#    ranked chunks (never truncate mid-document — that corrupts evidence).
# ===========================================================================
def fit_to_budget(ranked_chunks: list[str], budget_tokens: int,
                  reserved: int = 300) -> tuple[list[str], int]:
    """Greedily pack top-ranked chunks until the token budget (minus a reserve for
    the system prompt + answer) is exhausted. Returns (kept, tokens_used).
    Dropping WHOLE low-rank chunks preserves per-chunk integrity and citations;
    truncating a chunk mid-sentence would silently break grounding."""
    available = budget_tokens - reserved
    kept, used = [], 0
    for ch in ranked_chunks:
        t = COUNT(ch)
        if used + t > available:
            break
        kept.append(ch)
        used += t
    return kept, used


def demo_budget() -> None:
    chunks = [f"Order {1000+i} record with notes and status and totals verbalized "
              f"as prose sentence number {i}." for i in range(1, 9)]
    kept, used = fit_to_budget(chunks, budget_tokens=200, reserved=60)
    print(f"  8 candidate chunks, budget=200 (reserve 60) -> kept {len(kept)} "
          f"chunks using {used} tokens")
    print("  TRADEOFF: smaller budget = cheaper+faster but risks dropping evidence "
          "-> higher context_recall failure. Budget is a recall/cost dial.")


# ===========================================================================
# LIVE — real structured-output call over the orders pipeline.
# ===========================================================================
def demo_live_structured_output() -> None:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from rag_langchain_pinecone import (
        CHAT_MODEL, get_embeddings, get_vectorstore, load_env,
    )

    load_env()
    model_cls = order_answer_model()
    vs = get_vectorstore(get_embeddings())
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    q = "What happened with order 1011?"
    docs = retriever.invoke(q)
    context = "\n\n".join(f"[order {d.metadata.get('order_id')}] {d.page_content}" for d in docs)

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0).with_structured_output(model_cls)
    prompt = ChatPromptTemplate.from_messages([
        ("system", ZERO_SHOT_SYSTEM + "\n\nReturn a typed object.\n\nCONTEXT:\n{context}"),
        ("human", "{q}"),
    ])
    obj = (prompt | llm).invoke({"context": context, "q": q})
    print(f"  typed output: answer={obj.answer[:60]!r}")
    print(f"                cited_order_ids={obj.cited_order_ids}  grounded={obj.grounded}")
    print("  -> downstream grounding grader reads .grounded / .cited_order_ids as "
          "FIELDS, no prose parsing, no regex, deterministic gate.")


def main() -> None:
    live = "--live" in sys.argv
    print(f"[token counter] {COUNTER_KIND}\n")
    print("=== 1) ZERO-SHOT vs FEW-SHOT (token cost) ===")
    demo_zero_vs_few_shot()
    print("\n=== 3) VALIDATION + REPAIR (bounded self-correction) ===")
    demo_validation_repair()
    print("\n=== 4) PROMPT COMPRESSION (quality-vs-tokens) ===")
    demo_compression()
    print("\n=== 5) TOKEN BUDGETING (fit context to a budget) ===")
    demo_budget()
    print("\n=== 2) STRUCTURED OUTPUT ===")
    if live:
        demo_live_structured_output()
    else:
        print("  (offline) schema:",
              json.dumps(order_answer_model().model_json_schema()["properties"]))
        print("  run with --live for a real with_structured_output() call.")


if __name__ == "__main__":
    main()
