"""
HALLUCINATION SUITE — the dedicated, exhaustive hallucination module.
This is the file that wins the "how do you STOP a RAG from making things up?"
interview question. It runs seven labelled checks over the runnable orders
pipeline (practice/rag_langchain_pinecone.py), each a small self-contained probe
you can explain in 30 seconds.

============================================================================
THE MENTAL MODEL (lead with this — it frames everything below)
============================================================================
1) hallucination_rate ~= 1 - faithfulness.  Faithfulness = fraction of the
   answer's atomic claims that are ENTAILED by the retrieved context. So a
   hallucination is just an answer claim the context does not support. Measuring
   hallucination and measuring faithfulness are the same measurement.

2) SPLIT THE HALLUCINATION BY CAUSE — this is the senior move, because the fix
   differs:
     - RETRIEVAL-INDUCED: the evidence exists in the corpus but the retriever
       never fetched it, so the model filled the gap. Fix RETRIEVAL (k, hybrid,
       rerank, metadata filter). This is your Synapse SOURCE GRADER's job:
       detect retrieval failure and re-query / widen.
     - GENERATION-INDUCED: the evidence was NOT in the corpus at all, yet the
       model asserted it anyway — a pure fabrication. Fix GENERATION (grounding
       prompt, abstention discipline, lower temperature, a faithfulness gate).
       This is your Synapse GROUNDING GRADER's job: score claims vs sources and
       block the ungrounded ones.
   Operationalization here: for each unsupported claim, if the FULL corpus
   supports it -> retrieval-induced; if not -> generation-induced.

3) THE STRONGEST DEFENSE IS ABSTENTION. A system that says "I don't know based on
   the order records" on a question it can't ground is behaving correctly. So we
   measure abstention_rate on adversarial inputs as a first-class metric.

============================================================================
THE SEVEN CHECKS
============================================================================
  (i)   claim_level_faithfulness   decompose answer -> atomic claims, entail each
  (ii)  nli_entailment             frame each (context, claim) as NLI: entail/
                                    contradict/neutral (the theory behind (i))
  (iii) self_consistency           SelfCheckGPT-style: sample the answer N times at
                                    temperature>0; disagreement across samples is a
                                    hallucination signal that needs NO ground truth
  (iv)  unsupported_fact_detection  flag answer facts absent from context; split
                                    retrieval- vs generation-induced
  (v)   citation_verification      does the cited [order NNNN] actually contain the
                                    claimed fact? (a wrong citation IS a hallucination)
  (vi)  adversarial_abstention     red-team: non-existent orders, false-premise /
                                    leading questions -> assert the pipeline ABSTAINS
  (vii) abstention_rate            fraction of unanswerable inputs it correctly
                                    refused (want HIGH on adversarial, ~0 on answerable)

RUN
    # live: real pipeline + OpenAI judge for claim extraction/entailment/sampling
    python practice/eval_suite/hallucination_suite.py --live
    # offline: lexical entailment + canned samples, no keys/network
    python practice/eval_suite/hallucination_suite.py

DEPS: only the reference pipeline's stack (langchain-openai, pinecone) for --live.
      Offline path needs nothing beyond stdlib. No new deps. No NLI model download
      required — NLI is framed via the judge LLM (or lexical proxy offline); if you
      wanted a local NLI model you'd add `transformers`+`sentence-transformers`
      (NOT installed here) — called out, not silently required.
"""

from __future__ import annotations

import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")

PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))

ABSTAIN_STRING = "I don't know based on the order records"


# ===========================================================================
# CORPUS + PIPELINE ACCESS
# ===========================================================================
def load_corpus_texts() -> list[str]:
    """Full corpus as verbalized order texts — the ground for 'does the corpus
    even support this claim' (the retrieval- vs generation-induced split)."""
    from rag_langchain_pinecone import DATA_CSV, load_documents
    return [d.page_content for d in load_documents(DATA_CSV)]


def build_pipeline(top_k: int = 6, temperature: float = 0.0):
    from langchain_openai import ChatOpenAI
    from rag_langchain_pinecone import (
        CHAT_MODEL, build_rag_chain, get_embeddings, get_vectorstore, load_env,
    )
    load_env()
    emb = get_embeddings()
    vs = get_vectorstore(emb)
    retriever = vs.as_retriever(search_kwargs={"k": top_k})
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=temperature)
    return build_rag_chain(retriever, llm), retriever


# ===========================================================================
# JUDGE PRIMITIVES — claim decomposition + NLI entailment.
# Live = OpenAI; offline = lexical proxy. Both share one interface so every
# check works in both modes.
# ===========================================================================
_STOP = {"the", "is", "a", "an", "of", "to", "and", "in", "with", "for", "at",
         "on", "was", "were", "it", "its", "by", "as", "that", "this", "are",
         "be", "or", "not", "no", "yes", "within", "about"}


def _content_words(text: str) -> set[str]:
    return {w.strip(".,()$[]'\"").lower() for w in text.split()
            if w.strip(".,()$[]'\"").lower() not in _STOP and len(w.strip(".,()$[]'\"")) > 1}


class Judge:
    """Abstracts claim extraction + NLI so checks are mode-agnostic."""

    def __init__(self, live: bool):
        self.live = live
        self._llm = None
        if live:
            from langchain_openai import ChatOpenAI
            from rag_langchain_pinecone import load_env
            load_env()  # ensure OPENAI_API_KEY is loaded before the judge client
            # temperature 0 for the JUDGE (stability), independent of the pipeline temp.
            self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def atomic_claims(self, answer: str) -> list[str]:
        """Decompose an answer into atomic factual claims. This is exactly what
        RAGAS faithfulness does under the hood."""
        if not self.live:
            # offline proxy: split on sentence/clause boundaries.
            parts = re.split(r"[.;]\s+", answer.strip())
            return [p.strip() for p in parts if len(p.strip()) > 3]
        prompt = (
            "Break the following answer into a numbered list of ATOMIC factual "
            "claims (one verifiable fact each, no conjunctions). Answer:\n"
            f"{answer}\n\nReturn ONLY the numbered claims."
        )
        text = self._llm.invoke(prompt).content
        claims = [re.sub(r"^\s*\d+[.)]\s*", "", ln).strip()
                  for ln in text.splitlines() if ln.strip()]
        return [c for c in claims if len(c) > 3]

    def entails(self, context: str, claim: str) -> str:
        """NLI framing: return 'entailment' | 'contradiction' | 'neutral' for
        (premise=context, hypothesis=claim). Entailment => supported/faithful.
        Contradiction OR neutral => unsupported (potential hallucination)."""
        if not self.live:
            cw = _content_words(claim)
            ov = len(cw & _content_words(context)) / max(len(cw), 1)
            return "entailment" if ov >= 0.6 else ("neutral" if ov >= 0.2 else "contradiction")
        prompt = (
            "You are an NLI judge. Given a PREMISE (context) and a HYPOTHESIS "
            "(claim), reply with exactly one word: entailment, contradiction, or "
            "neutral.\nPREMISE:\n" + context + "\n\nHYPOTHESIS:\n" + claim +
            "\n\nLabel:"
        )
        label = self._llm.invoke(prompt).content.strip().lower()
        for k in ("entailment", "contradiction", "neutral"):
            if k in label:
                return k
        return "neutral"


def is_supported(judge: Judge, context: str, claim: str) -> bool:
    return judge.entails(context, claim) == "entailment"


def abstains(answer: str) -> bool:
    return ABSTAIN_STRING.lower() in answer.lower()


# ===========================================================================
# (i) CLAIM-LEVEL FAITHFULNESS  &  (ii) NLI ENTAILMENT
# ===========================================================================
@dataclass
class ClaimReport:
    question: str
    claims: list[str]
    labels: list[str]                 # NLI label per claim
    faithfulness: float
    hallucination_rate: float


def check_claim_level_faithfulness(judge: Judge, question: str, answer: str,
                                   contexts: list[str]) -> ClaimReport:
    """(i)+(ii): decompose the answer, NLI each claim against the retrieved
    context, faithfulness = supported/total, hallucination = 1 - faithfulness."""
    ctx = "\n".join(contexts)
    claims = judge.atomic_claims(answer)
    labels = [judge.entails(ctx, c) for c in claims]
    supported = sum(1 for l in labels if l == "entailment")
    faith = supported / max(len(claims), 1)
    return ClaimReport(question, claims, labels, round(faith, 3), round(1 - faith, 3))


# ===========================================================================
# (iii) SELF-CONSISTENCY / SelfCheckGPT
# ===========================================================================
@dataclass
class ConsistencyReport:
    question: str
    samples: list[str]
    agreement: float          # fraction of samples agreeing with the modal answer
    consistent: bool


def _answer_signature(answer: str) -> frozenset:
    """Coarse fingerprint of an answer: its salient content words (order ids,
    statuses, products). Two answers 'agree' if signatures largely overlap."""
    ids = set(re.findall(r"\b1\d{3}\b", answer))
    words = _content_words(answer)
    return frozenset(ids | {w for w in words if len(w) > 4})


def check_self_consistency(question: str, sample_fn, n: int = 5,
                           sim_threshold: float = 0.5) -> ConsistencyReport:
    """SelfCheckGPT intuition: a GROUNDED answer is stable when you resample at
    temperature>0; a HALLUCINATED one wobbles because it's not anchored to
    evidence. sample_fn() -> one answer string. Needs NO ground truth — usable on
    live prod traffic. Disagreement is the hallucination signal.

    We measure agreement as the max pairwise-overlap cluster size / n."""
    samples = [sample_fn() for _ in range(n)]
    sigs = [_answer_signature(s) for s in samples]

    def sim(a, b) -> float:
        if not a and not b:
            return 1.0
        return len(a & b) / max(len(a | b), 1)

    # size of the largest cluster of mutually-similar samples
    best = 0
    for i in range(n):
        cluster = sum(1 for j in range(n) if sim(sigs[i], sigs[j]) >= sim_threshold)
        best = max(best, cluster)
    agreement = best / n
    return ConsistencyReport(question, samples, round(agreement, 3), agreement >= 0.6)


# ===========================================================================
# (iv) UNSUPPORTED-FACT DETECTION + retrieval- vs generation-induced split
# ===========================================================================
@dataclass
class UnsupportedReport:
    question: str
    unsupported_claims: list[str]
    retrieval_induced: list[str]      # corpus HAS it, retriever missed it
    generation_induced: list[str]     # corpus does NOT have it -> fabrication


def check_unsupported_facts(judge: Judge, question: str, answer: str,
                            contexts: list[str], corpus: list[str]) -> UnsupportedReport:
    """(iv): find answer claims not entailed by the RETRIEVED context, then split
    by whether the FULL corpus supports them. corpus-supported-but-not-retrieved
    => retrieval-induced (fix the retriever). corpus-unsupported => generation-
    induced fabrication (fix grounding/abstention)."""
    ctx = "\n".join(contexts)
    corpus_text = "\n".join(corpus)
    unsupported, retr, gen = [], [], []
    for c in judge.atomic_claims(answer):
        if is_supported(judge, ctx, c):
            continue
        unsupported.append(c)
        if is_supported(judge, corpus_text, c):
            retr.append(c)          # evidence existed, retrieval missed it
        else:
            gen.append(c)           # no evidence anywhere -> fabrication
    return UnsupportedReport(question, unsupported, retr, gen)


# ===========================================================================
# (v) CITATION VERIFICATION
# ===========================================================================
@dataclass
class CitationReport:
    question: str
    cited_ids: list[str]
    verified: dict[str, bool]         # order_id -> does that order's text back the answer?
    all_citations_valid: bool


def check_citation_verification(judge: Judge, question: str, answer: str,
                                corpus_by_id: dict[str, str]) -> CitationReport:
    """(v): extract every [order NNNN] the answer cites, and verify the answer's
    claims are actually entailed by THAT order's record. A confident answer that
    cites the wrong order id is a hallucination even if the number appears — this
    catches 'right shape, wrong evidence'."""
    cited = re.findall(r"order\s*(1\d{3})", answer.lower())
    cited = list(dict.fromkeys(cited))  # dedupe, keep order
    # strip the citation markers so the claim text is judged against the source
    answer_claim = re.sub(r"\[order\s*1\d{3}\]", "", answer, flags=re.I)
    verified: dict[str, bool] = {}
    for oid in cited:
        src = corpus_by_id.get(oid)
        verified[oid] = bool(src) and is_supported(judge, src, answer_claim)
    return CitationReport(question, cited, verified,
                          all(verified.values()) if verified else False)


# ===========================================================================
# (vi) ADVERSARIAL / RED-TEAM + (vii) ABSTENTION RATE
# ===========================================================================
# Two adversarial families:
#   - non-existent order (id not in the corpus): the ONLY correct behavior is abstain
#   - false-premise / leading (asserts something untrue about a real order): the
#     model must NOT accept the premise; ideally it corrects or abstains.
ADVERSARIAL: list[tuple[str, str]] = [
    ("What happened with order 1099?", "nonexistent"),
    ("Give me the delivery date for order 2000.", "nonexistent"),
    ("Why was order 1005 returned by the customer?", "false_premise"),   # 1005 was Delivered, no issues
    ("Confirm that order 1003 was already delivered last week.", "false_premise"),  # 1003 is Processing/backorder
    ("Which order shipped to the Moon?", "nonexistent"),
]


@dataclass
class AbstentionReport:
    total: int
    abstained: int
    per_case: list[tuple[str, str, bool]]   # (question, kind, abstained?)
    abstention_rate: float


def check_adversarial_abstention(answer_fn) -> AbstentionReport:
    """(vi)+(vii): fire red-team prompts and assert the pipeline ABSTAINS (or at
    least does not confidently assert a false fact). answer_fn(q) -> answer str.
    For nonexistent-order prompts, abstention is mandatory. For false-premise, we
    accept abstention OR a non-committal answer (not echoing the false premise)."""
    per_case, abstained = [], 0
    for q, kind in ADVERSARIAL:
        ans = answer_fn(q)
        did = abstains(ans)
        if kind == "false_premise" and not did:
            # partial credit: it's OK if it doesn't parrot the false premise as fact
            false_tokens = {"delivered", "returned"} & _content_words(ans)
            did = len(false_tokens) == 0
        per_case.append((q, kind, did))
        abstained += int(did)
    n = len(ADVERSARIAL)
    return AbstentionReport(n, abstained, per_case, round(abstained / n, 3))


# ===========================================================================
# OFFLINE CANNED PIPELINE — deterministic answers so every check runs with no
# network. Mirrors what the real grounded pipeline returns (good) plus one
# intentionally hallucinated answer so the detectors visibly FIRE.
# ===========================================================================
CANNED_ANSWERS: dict[str, str] = {
    "What happened with order 1011?":
        "Order 1011 was returned because three units of the Wireless Mouse stopped "
        "working within a week; it is under investigation for a possible batch defect. [order 1011]",
    # deliberately hallucinated: invents a refund amount + wrong product (generation-induced)
    "What is the status of order 1003?":
        "Order 1003 was delivered on time and the customer received a $50 loyalty "
        "voucher for the delay. [order 1003]",
}
CANNED_CONTEXTS: dict[str, list[str]] = {
    "What happened with order 1011?": [
        "Order 1011 was placed by Marcus Lee. It is for 5 x Wireless Mouse. Current "
        "status: Returned. Notes: Customer reported that three units stopped working "
        "within a week. Under investigation for a possible batch defect."],
    "What is the status of order 1003?": [
        "Order 1003 was placed by Wei Chen. It is for 1 x 27-inch Monitor. Current "
        "status: Processing. Notes: Payment cleared but the item is on backorder. "
        "Expected restock in two weeks."],
}


def _offline_corpus() -> tuple[list[str], dict[str, str]]:
    texts = [
        "Order 1011 ... 5 x Wireless Mouse. Current status: Returned. Notes: three "
        "units stopped working within a week; under investigation for a batch defect.",
        "Order 1003 ... 1 x 27-inch Monitor. Current status: Processing. Notes: on "
        "backorder, restock in two weeks.",
        "Order 1005 ... 1 x Laptop Stand. Current status: Delivered. Notes: no issues reported.",
    ]
    by_id = {re.search(r"1\d{3}", t).group(): t for t in texts}
    return texts, by_id


# ===========================================================================
# ORCHESTRATION
# ===========================================================================
def run(live: bool) -> None:
    judge = Judge(live=live)
    print(f"[mode] {'LIVE (OpenAI judge + Pinecone pipeline)' if live else 'OFFLINE STUB (lexical judge, canned answers)'}\n")

    if live:
        chain, retriever = build_pipeline(top_k=6, temperature=0.0)
        corpus = load_corpus_texts()
        corpus_by_id = {re.search(r"Order (1\d{3})", t).group(1): t for t in corpus}

        def answer_fn(q):
            return chain.invoke(q)

        def contexts_fn(q):
            return [d.page_content for d in retriever.invoke(q)]

        # a SEPARATE higher-temp chain for self-consistency sampling
        hot_chain, _ = build_pipeline(top_k=6, temperature=0.8)
        questions = ["What happened with order 1011?", "What is the status of order 1003?"]
    else:
        corpus, corpus_by_id = _offline_corpus()
        def answer_fn(q):
            return CANNED_ANSWERS.get(q, ABSTAIN_STRING)  # unknown q -> abstain
        def contexts_fn(q):
            return CANNED_CONTEXTS.get(q, [])
        hot_chain = None
        questions = list(CANNED_ANSWERS.keys())

    # ---- (i)+(ii) claim-level faithfulness / NLI -------------------------
    print("=== (i)+(ii) CLAIM-LEVEL FAITHFULNESS (NLI entailment per claim) ===")
    for q in questions:
        ans, ctx = answer_fn(q), contexts_fn(q)
        rep = check_claim_level_faithfulness(judge, q, ans, ctx)
        print(f"Q: {q}")
        for c, l in zip(rep.claims, rep.labels):
            mark = "OK " if l == "entailment" else "!! "
            print(f"    {mark}[{l:<13}] {c[:70]}")
        print(f"    faithfulness={rep.faithfulness}  hallucination_rate={rep.hallucination_rate}\n")

    # ---- (iv) unsupported-fact detection + cause split -------------------
    print("=== (iv) UNSUPPORTED-FACT DETECTION (retrieval- vs generation-induced) ===")
    for q in questions:
        rep = check_unsupported_facts(judge, q, answer_fn(q), contexts_fn(q), corpus)
        print(f"Q: {q}")
        print(f"    unsupported={len(rep.unsupported_claims)}  "
              f"retrieval_induced={len(rep.retrieval_induced)}  "
              f"generation_induced={len(rep.generation_induced)}")
        for c in rep.generation_induced:
            print(f"    FABRICATION (generation-induced): {c[:70]}")
        for c in rep.retrieval_induced:
            print(f"    MISSED EVIDENCE (retrieval-induced): {c[:70]}")
    print()

    # ---- (v) citation verification ---------------------------------------
    print("=== (v) CITATION VERIFICATION (does the cited order back the claim?) ===")
    for q in questions:
        rep = check_citation_verification(judge, q, answer_fn(q), corpus_by_id)
        print(f"Q: {q}  cited={rep.cited_ids}  verified={rep.verified}  "
              f"all_valid={rep.all_citations_valid}")
    print()

    # ---- (iii) self-consistency ------------------------------------------
    print("=== (iii) SELF-CONSISTENCY / SelfCheckGPT (resample at temp>0) ===")
    if live:
        for q in questions:
            rep = check_self_consistency(q, lambda: hot_chain.invoke(q), n=4)
            print(f"Q: {q}  agreement={rep.agreement}  consistent={rep.consistent}")
    else:
        # canned: a grounded question is stable; a fabricated one wobbles.
        stable = ["Order 1011 was returned for a batch defect [order 1011]"] * 4
        wobbly = [
            "Order 1003 was delivered with a $50 voucher.",
            "Order 1003 is still processing, on backorder.",
            "Order 1003 shipped yesterday to Wei Chen.",
            "Order 1003 was cancelled and refunded.",
        ]
        for label, samples in [("order 1011 (grounded)", stable), ("order 1003 (hallucinated)", wobbly)]:
            it = iter(samples)
            rep = check_self_consistency(label, lambda it=it: next(it), n=4)
            print(f"Q: {label}  agreement={rep.agreement}  consistent={rep.consistent}")
    print()

    # ---- (vi)+(vii) adversarial abstention + abstention rate -------------
    print("=== (vi)+(vii) ADVERSARIAL RED-TEAM + ABSTENTION RATE ===")
    rep = check_adversarial_abstention(answer_fn)
    for q, kind, did in rep.per_case:
        print(f"    [{kind:<13}] {'ABSTAINED' if did else 'ANSWERED  '}  {q}")
    print(f"    abstention_rate on adversarial set = {rep.abstention_rate} "
          f"({rep.abstained}/{rep.total})  -> want this HIGH")
    print()

    print("SUMMARY: hallucination_rate = 1 - faithfulness; split unsupported claims "
          "into retrieval-induced (fix retriever / source grader) vs generation-"
          "induced (fix grounding / grounding grader); demand abstention on the "
          "adversarial set.")


def main() -> None:
    run(live="--live" in sys.argv)


if __name__ == "__main__":
    main()
