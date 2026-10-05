# Benchmark (Step 16)

Every number below comes from `scripts/bench.py` (raw JSON in `docs/bench/`, tables rendered by `scripts/bench_report.py`, which
reads only those files). Reproduce a row with one command:

```
python scripts/bench.py --docs 200 --queries 300 --seed 1 --stages     # add --profile for a cProfile of the event-loop thread
```

## What is measured, and what is not

* The **engine**: service, Milvus, BM25, rank fusion. Embeddings and the chat model are the in-memory fakes, so no network, no paid
  API and **no model latency**: a real deployment adds the embedding call (ingest, and one per query) and the generation call (query).
* **Milvus Lite** (embedded, single process, a local file), because standalone Milvus could not be started in this environment
  (image pull blocked). Lite serialises access, so **concurrency numbers here describe Lite, not Milvus standalone** and say nothing
  about how a real cluster scales. Run the same command with `AGENTIC_RAG_MILVUS_URI=http://host:19530` to measure standalone.
* One machine, a few runs per row; the first two rows differ by about 8 % on ingest and 3 % on search, so differences of that size are noise.
* Warm search = 300 queries drawn from the corpus vocabulary after a 50-query warm-up; ingest is cold (empty collection), 4 concurrent uploads.
* The first four rows were measured on the code as it stood at the time. Later rows were measured on this machine within one session so
  that "before" and "after" share conditions (the commit before review round 1 was checked out and run again for that).

## Results

Machine: 4 vCPU, Python 3.11.15, Linux-6.18.44-fc-v70-x86_64-with-glibc2.39; Milvus: milvus-lite (embedded, single process). Corpus: 200 synthetic documents, 4,688,201 bytes, seed 1.

| Run | Ingest docs/s | Ingest chunks/s | Search c=1 p50 / p95 ms | c=4 p50 / p95 ms | c=16 p50 / p95 ms | Search req/s (c=1 / 4 / 16) |
|---|---|---|---|---|---|---|
| baseline (run 1) | 13.17 | 263.3 | 35.02 / 61.89 | 138.84 / 223.91 | 581.38 / 630.43 | 25.7 / 26.6 / 27.9 |
| baseline (run 2) | 14.23 | 284.5 | 33.87 / 57.23 | 146.45 / 216.57 | 573.76 / 644.72 | 26.6 / 25.8 / 27.6 |
| A: cache the index verification | 17.77 | 355.4 | 33.66 / 52.36 | 140.93 / 214.64 | 581.55 / 683.14 | 27.2 / 26.2 / 26.9 |
| A+B: two-phase commit marker, 500-row batches | 16.57 | 331.4 | 35.26 / 51.58 | 145.54 / 234.18 | 603.41 / 694.12 | 26.0 / 25.5 / 26.0 |
| commit before review round 1, measured in the same session (run 1) | 16.87 | 337.4 | 35.93 / 80.27 | 144.07 / 222.52 | 612.58 / 668.98 | 25.1 / 25.6 / 26.1 |
| commit before review round 1, measured in the same session (run 2) | 17.13 | 342.7 | 35.43 / 64.67 | 153.43 / 233.87 | 615.25 / 713.06 | 25.4 / 24.4 / 26.0 |
| after review round 1, 8 store workers on Lite (run 1) | 16.02 | 320.4 | 37.69 / 58.7 | 160.7 / 255.84 | 684.29 / 1063.87 | 24.0 / 22.9 / 23.2 |
| after review round 1, 8 store workers on Lite (run 2) | 16.61 | 332.2 | 37.09 / 55.37 | 160.97 / 251.47 | 606.3 / 966.21 | 24.5 / 23.1 / 25.1 |
| after review round 1, 1 store worker on Lite: **current** (run 1) | 16.66 | 333.3 | 37.04 / 65.97 | 160.86 / 255.8 | 623.64 / 729.32 | 24.3 / 23.3 / 25.4 |
| after review round 1, 1 store worker on Lite: **current** (run 2) | 17.36 | 347.2 | 37.9 / 72.51 | 150.42 / 244.93 | 662.82 / 768.73 | 24.1 / 24.6 / 23.9 |
| after review round 1, 1 store worker on Lite: **current** (run 3, with stage timing) | 16.71 | 334.3 | 36.34 / 58.24 | 155.08 / 243.98 | 676.81 / 823.91 | 24.9 / 23.8 / 23.2 |
| experiment: store calls on the default executor (run 1) | 17.47 | 349.5 | 34.01 / 53.42 | 141.75 / 218.4 | 580.89 / 1123.16 | 26.6 / 26.4 / 26.3 |
| experiment: store calls on the default executor (run 2) | 17.74 | 354.9 | 33.62 / 48.17 | 138.77 / 214.89 | 642.21 / 1004.25 | 26.7 / 26.4 / 24.8 |
| experiment: lexical search on the default executor (run 1) | 16.51 | 330.1 | 34.47 / 48.44 | 149.78 / 232.47 | 593.01 / 1082.99 | 26.4 / 25.2 / 25.8 |
| experiment: lexical search on the default executor (run 2) | 16.64 | 332.8 | 36.93 / 70.42 | 149.73 / 229.29 | 571.35 / 1120.84 | 24.0 / 24.9 / 26.5 |

## Where the time goes (profile first, then change)

Each retrieval stage timed on its own on 4,000 indexed chunks (`--stages`, single caller, p50):

| Run | Indexed chunks | Query embedding (fake) | Dense search incl. embedding | BM25 | `service.search` end to end |
|---|---|---|---|---|---|
| after review round 1, 8 store workers on Lite (run 1) | 4000 | 0.007 ms | 31.879 ms | 3.329 ms | 37.961 ms |
| after review round 1, 8 store workers on Lite (run 2) | 4000 | 0.007 ms | 29.786 ms | 3.317 ms | 36.155 ms |
| after review round 1, 1 store worker on Lite: **current** (run 3, with stage timing) | 4000 | 0.007 ms | 31.021 ms | 3.491 ms | 37.957 ms |

About 82 % of an end-to-end search is the datastore call (query embedding is negligible with the fake) and BM25 is about 3.5 ms; the
application adds the rest. Throughput stays at about 23 to 26 requests/s whether 1, 4 or 16 clients are active while latency grows in
proportion to the client count: **the ceiling is the datastore (Lite's serialised access), not the application.** BM25's cost grows
with the number of chunks (0.67 ms at 1,200 chunks in an earlier ad-hoc timing, 3.3 to 3.5 ms at 4,000 here).

## Changes, before, after, kept or reverted

| Change | Before | After | Verdict |
|---|---|---|---|
| A. Remember that the index was already verified for this embedding model and size (saves `describe_collection` + a row query on every ingest) | 13.2 / 14.2 docs/s | 17.8 docs/s (+30 %) | **kept** |
| B. Write chunk 0 (the document record and commit marker) as its own second upsert; upsert in 500-row batches | 17.8 docs/s | 16.6 docs/s (-7 %, inside noise) | **kept**: it costs one extra round trip but makes the marker mean what the code says it means, and keeps each gRPC message under Milvus's 64 MB limit |
| C. Review round 1: bounded store/lexical thread pools, auth middleware, timeouts, ingest order write → sweep → lexical → commit | search c=1 p50 35.4 / 35.9 ms; c=16 p95 669 / 713 ms | with 8 store workers on Lite: c=1 p50 37.1 / 37.7 ms (+5 %); **c=16 p95 966 / 1,064 ms (+45 %)** | see next row |
| D. A store worker count of 1 when the engine is embedded (Milvus Lite runs one call at a time) | c=16 p95 966 / 1,064 ms | c=16 p95 729 / 769 ms | **kept**: the tail came from eight threads contending for Lite's lock; one worker serves requests in arrival order. Standalone Milvus keeps eight workers. |

What remains after D, against the same-session "before" runs (three "current" runs): c=1 p50 36.3 to 37.9 ms against 35.4 / 35.9 ms
(+1 to +5 %); c=16 p95 729 to 824 ms against 669 / 713 ms (+2 to +23 %); ingest within noise. Putting the store calls back on the
default executor (an experiment, kept in `docs/bench/experiment-*`) brought the median to 33.6 to 34.0 ms, at or below the "before" runs,
but not the c=16 tail; putting back only the lexical-search pool changed neither clearly. **The cause of the remaining few-percent median
cost was not isolated.** The pools buy failure containment (a hung store strands a bounded set of threads and no longer starves parsing
and BM25), so they are kept; a re-measurement against standalone Milvus is listed under next steps.

Not tried because the profile says they cannot help here: more worker threads or a connection pool (Lite is single-process),
caching query embeddings (embedding is 0.01 ms with the fake and a network call with the real one: measure with a real provider first),
and lowering `candidate_pool` (a retrieval-quality trade-off that needs the evaluation harness, Step 18).

## Load shedding and limits (verified by tests, not by this benchmark)

`/v1` accepts at most `AGENTIC_RAG_MAX_INFLIGHT_REQUESTS` (default 64) concurrent *authenticated* requests; the next ones get `503` with an
integer `Retry-After` (`tests/api/test_operability.py`, `tests/api/test_hardening.py`). Whether 64 is the right number for a given
deployment is **unmeasured**.
