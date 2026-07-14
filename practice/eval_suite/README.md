# eval_suite — production RAG evaluation & hallucination testing

Exhaustive, CI-grade evaluation of the runnable reference pipeline
`practice/rag_langchain_pinecone.py`
(OpenAIEmbeddings `text-embedding-3-small` → Pinecone `customer-orders-rag` →
ChatOpenAI `gpt-4o-mini`, over 16 customer orders). Supersedes the intro
`practice/rag_eval_ragas_deepeval.py`.

Every file runs **live** (real OpenAI + Pinecone, `--live`) or **offline**
(lexical stub, no keys — clearly labelled, values illustrative not real judge
scores).

## Files → interview themes → your projects

| File | Interview theme | Maps to your work |
|---|---|---|
| `ragas_production.py` | "How do you eval a RAG system / localize a regression to retrieval vs generation?" | Synapse **grounding grader** = faithfulness; **source grader** = context_recall failure |
| `deepeval_production.py` | "Show me a CI quality gate for RAG" + faithfulness-vs-hallucination distinction | **Universal Document Ingestor** shipped DeepEval; GEval = encoding Synapse business rules |
| `hallucination_suite.py` | "How do you STOP a RAG from making things up?" (a recurring one for you) | grounding grader (generation-induced) + source grader (retrieval-induced) |

## The one mental model (lead with it)

Every RAG failure is a **retrieval** failure or a **generation** failure:

```
RETRIEVAL quality                 GENERATION quality
context_precision (ranking)       faithfulness   (grounded? → hallucination)
context_recall    (coverage)      answer_relevancy (on-topic?)
context_entity_recall             answer_correctness / answer_similarity
noise_sensitivity                 hallucination_rate ≈ 1 − faithfulness
```

Ground-truth needs: faithfulness + relevance + noise_sensitivity need **none**
(→ run online in prod, see `../llm_optimization/03_...`); precision, recall,
entity-recall, correctness, similarity need a **labelled gold set** (→ run in CI,
here).

## ragas_production.py

- Exhaustive metrics probed from the installed ragas (**0.4.3**): the four core
  + `context_entity_recall`, `answer_correctness`, `answer_similarity`,
  `noise_sensitivity`. Any metric absent in a future version is skipped with a note.
- **Pinned judge** (ChatOpenAI gpt-4o-mini @ temp 0 + OpenAIEmbeddings) so scores
  don't drift run-to-run — the detail that makes week-over-week comparison valid.
- Builds the eval dataset by **running the pipeline** over `ORDERS_GOLD`.
- **CI regression gate**: per-metric thresholds → PASS/FAIL → nonzero exit
  (`--gate`). `noise_sensitivity` is a *ceiling* (lower is better), not a floor.
- **Per-sample failure export** → `ragas_failures.json` (which sample failed which
  metric) for fast debugging.

```bash
python ragas_production.py            # offline lexical stub
python ragas_production.py --live         # real eval
python ragas_production.py --live --gate  # CI gate, exits nonzero on breach
```

## deepeval_production.py

- Metrics: Faithfulness, AnswerRelevancy, ContextualPrecision/Recall/Relevancy,
  **Hallucination**, **Toxicity**, **Bias**, and a custom **GEval**
  ("answer cites the correct order id").
- **Faithfulness vs Hallucination** distinction wired explicitly:
  `FaithfulnessMetric` reads `retrieval_context` (grounding vs what we retrieved);
  `HallucinationMetric` reads `context` (contradiction vs ground truth). Different
  field, different question — a common interview tell.
- **pytest-style CI** via `assert_test` (`test_orders_rag_quality`) + an
  `EvaluationDataset`; **synthetic data** (`Synthesizer`) noted as reference.
- Installed-version import paths verified: `deepeval.test_case` (**singular**),
  `deepeval.dataset`, `deepeval` (`assert_test`). deepeval **4.0.7**.

```bash
python deepeval_production.py          # offline stub
python deepeval_production.py --live       # real judges
deepeval test run deepeval_production.py   # pytest CI harness
```

## hallucination_suite.py — the priority module

Seven labelled checks, each a small runnable probe:

1. **claim-level faithfulness** — decompose answer → atomic claims, entail each.
2. **NLI entailment** — each (context, claim) as entailment/contradiction/neutral.
3. **self-consistency / SelfCheckGPT** — resample N× at temp>0; disagreement =
   hallucination signal (no ground truth needed).
4. **unsupported-fact detection** — flag ungrounded claims, then **split**
   retrieval-induced (corpus has it, retriever missed → fix retriever) vs
   generation-induced (corpus lacks it → fabrication → fix grounding/abstention).
5. **citation verification** — does the cited `[order NNNN]` actually back the claim?
6. **adversarial / red-team** — non-existent orders + false-premise questions;
   assert the pipeline **abstains**.
7. **abstention rate** — fraction of unanswerable inputs correctly refused.

Mental model: `hallucination_rate ≈ 1 − faithfulness`; the retrieval-vs-generation
split decides which fix to reach for. Offline stub includes one deliberately
hallucinated answer so the detectors visibly fire.

```bash
python hallucination_suite.py         # offline (lexical judge, canned answers)
python hallucination_suite.py --live      # OpenAI judge + real pipeline
```

## Dependencies

All already installed — **no new deps required**:
`ragas` 0.4.3, `deepeval` 4.0.7, `datasets`, `langchain-openai`, `langchain-pinecone`,
`pinecone`, `tiktoken`, `pydantic`. Metrics are resolved defensively so a version
bump degrades to a printed skip. A local NLI model (optional, **not** installed)
would need `transformers` + `sentence-transformers`; the suite frames NLI via the
judge LLM instead, so nothing extra is required.
