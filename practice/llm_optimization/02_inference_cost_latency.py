"""
LLM OPTIMIZATION 2/3 — INFERENCE COST & LATENCY.
The levers that make a RAG system cheap and fast in production, applied to the
orders pipeline (practice/rag_langchain_pinecone.py). Interviewers ask "your RAG
works — now make it 10x cheaper and 2x faster without hurting quality." These are
the answers, each runnable/measured.

============================================================================
THE LEVERS (in the order you should reach for them)
============================================================================
  1. EXACT CACHE          identical prompt seen before -> skip the LLM entirely.
                          Biggest, cheapest win for repeated queries.
  2. SEMANTIC CACHE       near-duplicate question (cosine sim >= T) -> reuse the
                          cached answer. Hand-rolled here over the orders.
  3. MODEL CASCADE/ROUTE  answer with the CHEAP model; escalate to the STRONG
                          model only when confidence is low. Pay for GPT-4-class
                          only on the hard minority.
  4. STREAMING            chain.stream() -> first token fast; cuts PERCEIVED
                          latency even when total time is unchanged.
  5. BATCHING             chain.batch() -> amortize overhead across many queries
                          (offline eval / bulk scoring).
  6. CONTEXT TRIMMING     fewer/cleaner retrieved tokens -> less to encode -> lower
                          cost AND latency (see file 01's budgeting).
  7. QUANTIZATION         shrink the MODEL itself (weights fp16->int4). This is your
                          AP Audit story: QLoRA fine-tune of Phi-3.5 = 4-bit
                          quantized base + trainable LoRA adapters, so a 3.8B model
                          fine-tunes and serves on one modest GPU.

============================================================================
QUANTIZATION — the concept, tied to your AP Audit QLoRA of Phi-3.5
============================================================================
Quantization stores weights in fewer bits (fp16 -> int8 -> int4/NF4). Memory and
bandwidth drop ~4x from fp16->int4, so a model that needed an A100 now fits a
consumer GPU, and inference is faster (less memory traffic). The tradeoff is a
small accuracy loss you must EVAL for (back to eval_suite — never ship a quantized
model without re-running faithfulness). QLoRA = load the base in 4-bit (NF4), FREEZE
it, and train tiny LoRA adapters on top — you fine-tune a 3.8B Phi-3.5 cheaply
because you never update the frozen 4-bit base. Say: "quantization is inference-time
cost; QLoRA uses it to make TRAINING cheap too, which is exactly what AP Audit did."

RUN
    python practice/llm_optimization/02_inference_cost_latency.py          # offline: hand-rolled caches + routing + cost math
    python practice/llm_optimization/02_inference_cost_latency.py --live   # real streaming + cache hit/miss timing

DEPS: langchain-openai for --live. Offline path is pure stdlib (hand-rolled
      embeddings for the semantic cache). No new deps.
"""

from __future__ import annotations

import hashlib
import math
import sys
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")
PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))

# Illustrative pricing ($/1M tokens) — gpt-4o-mini vs a strong model. Prices move;
# treat as relative magnitudes, not quotes. The POINT is the ~15-30x gap that makes
# routing pay off.
PRICE_PER_MTOK = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o":      {"in": 2.50, "out": 10.00},
}


def cost_usd(model: str, in_tok: int, out_tok: int) -> float:
    p = PRICE_PER_MTOK[model]
    return (in_tok * p["in"] + out_tok * p["out"]) / 1_000_000


# ===========================================================================
# 1) EXACT CACHE — normalize prompt -> hash -> memoize.
# ===========================================================================
class ExactCache:
    """Exact-match prompt cache with LRU eviction. Normalize (lowercase, collapse
    whitespace) so trivially-different-but-identical prompts still hit."""

    def __init__(self, capacity: int = 256):
        self._d: OrderedDict[str, str] = OrderedDict()
        self.capacity = capacity
        self.hits = self.misses = 0

    @staticmethod
    def _key(prompt: str) -> str:
        norm = " ".join(prompt.lower().split())
        return hashlib.sha256(norm.encode()).hexdigest()

    def get(self, prompt: str):
        k = self._key(prompt)
        if k in self._d:
            self._d.move_to_end(k)
            self.hits += 1
            return self._d[k]
        self.misses += 1
        return None

    def put(self, prompt: str, answer: str):
        k = self._key(prompt)
        self._d[k] = answer
        self._d.move_to_end(k)
        if len(self._d) > self.capacity:
            self._d.popitem(last=False)


# ===========================================================================
# 2) SEMANTIC CACHE — embed the query, reuse if cosine >= threshold.
# ===========================================================================
_STOP = {"what", "whats", "is", "the", "of", "a", "an", "to", "and", "in", "on",
         "for", "with", "s", "was", "are", "did", "do", "does", "me", "my"}


def hashed_embedding(text: str, dim: int = 64) -> list[float]:
    """Hand-rolled hashing embedding over CONTENT words — deterministic, no network,
    so the semantic cache is demonstrable offline. We drop stopwords so paraphrases
    ('what is the status of X' vs 'status of X') land close, which is exactly what a
    real sentence embedding does; in prod you swap in OpenAIEmbeddings (see
    demo_live) and the CACHE LOGIC is identical."""
    vec = [0.0] * dim
    import re
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _STOP:
            continue
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    n = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / n for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # both L2-normalized


class SemanticCache:
    """Reuse a cached answer when a new query is semantically near a stored one.
    Catches paraphrases the exact cache misses ('status of order 1003?' vs 'what's
    the state of order 1003'). Threshold is the precision/recall dial: too low and
    you serve a stale answer to a DIFFERENT question (a correctness bug), too high
    and you rarely hit. 0.9+ is typical for safety."""

    def __init__(self, embed_fn, threshold: float = 0.92):
        self.embed_fn = embed_fn
        self.threshold = threshold
        self._store: list[tuple[list[float], str, str]] = []  # (vec, query, answer)
        self.hits = self.misses = 0

    def get(self, query: str):
        qv = self.embed_fn(query)
        best_sim, best_ans = 0.0, None
        for vec, _q, ans in self._store:
            s = cosine(qv, vec)
            if s > best_sim:
                best_sim, best_ans = s, ans
        if best_ans is not None and best_sim >= self.threshold:
            self.hits += 1
            return best_ans, best_sim
        self.misses += 1
        return None, best_sim

    def put(self, query: str, answer: str):
        self._store.append((self.embed_fn(query), query, answer))


# ===========================================================================
# 3) MODEL CASCADE / ROUTING — cheap first, escalate on low confidence.
# ===========================================================================
@dataclass
class RouteResult:
    model_used: str
    answer: str
    escalated: bool


def confidence_of(answer: str) -> float:
    """Cheap, model-free confidence proxy: abstention or hedging -> low confidence.
    In prod, use the cheap model's logprobs, a self-rated score, or a fast
    faithfulness check. Low confidence is the ESCALATION trigger."""
    a = answer.lower()
    if "i don't know" in a or "not sure" in a or "cannot" in a:
        return 0.1
    hedges = sum(w in a for w in ("might", "maybe", "possibly", "unclear", "seems"))
    return max(0.2, 1.0 - 0.25 * hedges)


def cascade(query: str, cheap_fn, strong_fn, threshold: float = 0.5) -> RouteResult:
    """Answer with cheap model; if confidence < threshold, escalate to strong.
    Economics: if 80% of traffic is easy and answered by gpt-4o-mini, you pay
    gpt-4o prices on only 20% -> most of the quality at a fraction of the cost."""
    ans = cheap_fn(query)
    if confidence_of(ans) >= threshold:
        return RouteResult("gpt-4o-mini", ans, escalated=False)
    return RouteResult("gpt-4o", strong_fn(query), escalated=True)


# ===========================================================================
# OFFLINE DEMOS
# ===========================================================================
def demo_caches_offline() -> None:
    exact = ExactCache()
    sem = SemanticCache(hashed_embedding, threshold=0.9)

    def expensive_answer(q):  # stand-in for an LLM call
        time.sleep(0.01)
        return f"[answer to: {q}]"

    queries = [
        "What is the status of order 1003?",
        "What is the status of order 1003?",       # exact dup
        "what's the status of order 1003",          # paraphrase (semantic hit)
        "What happened with order 1011?",           # new
    ]
    print("  query -> cache outcome")
    for q in queries:
        if (a := exact.get(q)) is not None:
            print(f"    EXACT HIT   | {q}")
            continue
        a, sim = sem.get(q)
        if a is not None:
            print(f"    SEMANTIC HIT (sim={sim:.2f}) | {q}")
            exact.put(q, a)
            continue
        ans = expensive_answer(q)   # real call happens only here
        exact.put(q, ans)
        sem.put(q, ans)
        print(f"    MISS -> LLM call | {q}")
    print(f"  exact: {exact.hits} hits / {exact.misses} misses | "
          f"semantic: {sem.hits} hits / {sem.misses} misses")


def demo_cascade_offline() -> None:
    def cheap(q):
        # cheap model abstains on the hard/adversarial one
        return ("I don't know based on the order records."
                if "1099" in q else f"Order answer for: {q}")

    def strong(q):
        return f"[gpt-4o, deeper reasoning] answer for: {q}"

    total_cheap = total_routed = 0.0
    for q in ["status of order 1003?", "status of order 1005?", "details on order 1099?"]:
        r = cascade(q, cheap, strong, threshold=0.5)
        # rough token estimate for the cost illustration
        in_tok, out_tok = 400, 80
        total_cheap += cost_usd("gpt-4o-mini", in_tok, out_tok)
        total_routed += cost_usd(r.model_used, in_tok, out_tok)
        print(f"    {'ESCALATED->gpt-4o' if r.escalated else 'gpt-4o-mini      '} | {q}")
    all_strong = 3 * cost_usd("gpt-4o", 400, 80)
    print(f"  cost (3 queries): all-mini=${total_cheap:.5f}  "
          f"routed=${total_routed:.5f}  all-strong=${all_strong:.5f}")
    print("  -> routing pays strong-model price only on the 1 hard query.")


# ===========================================================================
# LIVE — real streaming + cache-hit timing over the orders pipeline.
# ===========================================================================
def demo_live() -> None:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from rag_langchain_pinecone import (
        CHAT_MODEL, build_rag_chain, get_embeddings, get_vectorstore, load_env,
    )
    load_env()
    vs = get_vectorstore(get_embeddings())
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    chain = build_rag_chain(retriever, ChatOpenAI(model=CHAT_MODEL, temperature=0))
    q = "What happened with order 1011?"

    # --- streaming: measure time-to-first-token vs total ---
    print("  STREAMING (time-to-first-token vs total):")
    t0 = time.time()
    first = None
    for chunk in chain.stream(q):
        if first is None:
            first = time.time() - t0
    total = time.time() - t0
    print(f"    first token @ {first:.2f}s, full answer @ {total:.2f}s "
          f"-> user sees output {total-first:.2f}s sooner than a blocking call")

    # --- exact cache over the LIVE chain ---
    print("  EXACT CACHE (cold vs warm latency):")
    cache = ExactCache()
    t0 = time.time(); a = chain.invoke(q); cold = time.time() - t0
    cache.put(q, a)
    t0 = time.time(); _ = cache.get(q); warm = time.time() - t0
    print(f"    cold (LLM) {cold:.2f}s -> warm (cache) {warm*1000:.1f}ms "
          f"({cold/max(warm,1e-6):.0f}x faster, $0 on the hit)")

    # --- real semantic cache with OpenAI embeddings ---
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    sem = SemanticCache(lambda t: _l2(emb.embed_query(t)), threshold=0.9)
    sem.put(q, a)
    hit, sim = sem.get("what happened to order 1011?")
    print(f"  SEMANTIC CACHE: paraphrase sim={sim:.3f} -> "
          f"{'HIT (reused answer, $0)' if hit else 'miss'}")


def _l2(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def main() -> None:
    if "--live" in sys.argv:
        print("[mode] LIVE\n=== streaming + caching over the orders chain ===")
        demo_live()
        return
    print("[mode] OFFLINE (hand-rolled caches + routing + cost math)\n")
    print("=== 1)+2) EXACT + SEMANTIC CACHE ===")
    demo_caches_offline()
    print("\n=== 3) MODEL CASCADE / ROUTING ===")
    demo_cascade_offline()
    print("\n=== 7) QUANTIZATION (concept) ===")
    print("  fp16->int4 ~= 4x less memory; QLoRA = frozen 4-bit base + LoRA adapters")
    print("  = your AP Audit fine-tune of Phi-3.5. Always re-eval faithfulness after "
          "quantizing (accuracy can dip).")
    print("\n  run --live for real streaming TTFT + cache-hit timing.")


if __name__ == "__main__":
    main()
