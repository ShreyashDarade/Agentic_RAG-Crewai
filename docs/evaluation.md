# Evaluation (Step 18)

"Better" is measured, not felt. This document says what can be measured today, what was measured, and what was not.

## The harness

`agentic-rag eval run|compare|check` (`src/agentic_rag/evaluation`, `src/agentic_rag/cli.py`):

* **Datasets** are BEIR-format directories (`corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`). Your own golden set uses the same layout;
  `python scripts/fetch_beir.py scifact` downloads the public one into `.dev/beir/` (licensed for non-commercial use, never committed).
* **Metrics** (definitions pinned in `evaluation/metrics.py`): nDCG@k (linear gain, log2(rank+1) discount), recall@k, precision@k,
  MRR@k, hit@k. Unjudged documents are not relevant. Per-query scores are saved in the run file so any comparison can be paired.
* **Statistics** (`evaluation/stats.py`): percentile bootstrap 95 % interval of the mean (10,000 resamples), paired sign-flip randomisation
  test (10,000 permutations; Smucker et al. 2007), Sakai's paired effect size, Holm correction when several configurations are tried.
  All seeded.
* **Regression gate** (`eval check`, exit code 1 on failure): fails on a significant loss (p < 0.05 and a negative mean change), on a
  95 % interval entirely below -0.005, or on an absolute floor. The thresholds are proposals to calibrate after a few baseline runs, not
  published rules.

```
python scripts/fetch_beir.py scifact
agentic-rag eval run --dataset .dev/beir/scifact --system bm25 --out run.json
agentic-rag eval compare docs/eval/scifact-bm25-k1=1.2-b=0.75.json run.json --metric ndcg@10
agentic-rag eval check --baseline docs/eval/scifact-bm25-k1=1.2-b=0.75.json --candidate run.json --floor 0.64
```

## Harness validation on a public benchmark

SciFact (BEIR; 5,183 documents, 300 test queries). The system here is this repository's own BM25 (`k1=1.2, b=0.75`, lower-cased `\w+`
tokens, no stemming or stop words) over title + abstract.

| | nDCG@10 | 95 % interval |
|---|---|---|
| This harness + this BM25 | **0.6617** | [0.617, 0.706] |
| BEIR paper, BM25 (Table 2) | 0.665 | — |

The difference is 0.003 against a tolerance of 0.02 that I chose in advance (BM25 implementations differ in tokenisation), so the metric
code, the loader and the runner reproduce a published number. This validates **BM25 retrieval and the metrics only**. The metric
implementations were checked against hand-computed values (`tests/unit/test_evaluation.py`), **not** against `pytrec_eval`.

## What was measured about this system

BM25 parameters on SciFact, each against the default with the paired test (nDCG@10; raw run files in `docs/eval/`):

| k1, b | nDCG@10 | change | 95 % interval of the change | effect size | p |
|---|---|---|---|---|---|
| 1.2, 0.75 (default) | 0.6617 | — | — | — | — |
| 0.9, 0.4 | 0.6601 | -0.0015 | [-0.0115, +0.0092] | 0.02 | 0.77 |
| 0.9, 0.75 | 0.6578 | -0.0038 | [-0.0108, +0.0030] | 0.06 | 0.28 |
| 1.2, 0.4 | 0.6610 | -0.0007 | [-0.0102, +0.0093] | 0.01 | 0.89 |
| 1.2, 0.9 | 0.6655 | +0.0039 | [-0.0033, +0.0113] | 0.06 | 0.31 |
| 1.5, 0.75 | 0.6647 | +0.0030 | [-0.0019, +0.0085] | 0.06 | 0.28 |
| 2.0, 0.75 | 0.6624 | +0.0007 | [-0.0097, +0.0109] | 0.01 | 0.90 |

Six configurations were tried, Holm-adjusted none is significant: **on SciFact BM25's parameters do not matter at this resolution**, so the
defaults stay and are recorded as "no evidence for a change", not as optimal. One dataset says nothing about your corpus.

## What was not measured (and why)

Everything that needs an embedding model or an LLM: dense retrieval, hybrid fusion against dense-only, `candidate_pool`, `rrf_k`,
chunk size and overlap, reranking, MMR, the HNSW parameters, `direct` against `crewai`, answer faithfulness, answer relevancy and
citation quality. **No API key or local model was available**, so each of these is `evidence = "unmeasured"` in `docs/defaults.toml`;
a test fails if a quality default is added without an entry. Features whose benefit is *unproven* are therefore not claimed to help:
reranking ships off, and there is no MMR or multi-query code (the evidence in `research.md` section 3 is that expansion often hurts strong
retrievers).

Implemented without a model: the deterministic citation check (every cited chunk must have been retrieved, enforced by the service on every
answer). Not implemented: LLM-judged faithfulness and answer relevancy (RAGAS-style), the judge-bias controls they require, and a
hand-labelled set for calibrating a judge. The harness is built so those slot in beside the retrieval metrics; they are not there.

## Gate in CI

`eval check` runs in CI only when a baseline run file is supplied; today the repository ships the SciFact BM25 baseline and no dense
baseline, so the gate protects the BM25 path (tokeniser, scoring) and nothing else.
