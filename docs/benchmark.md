# Benchmark (Step 16)

Every number below comes from `scripts/bench.py` (raw JSON in `docs/bench/`, tables rendered by `scripts/bench_report.py`).
Reproduce a row with one command:

```
python scripts/bench.py --docs 200 --queries 300 --seed 1          # add --profile for a cProfile of the event-loop thread
```

## What is measured, and what is not

* The **engine**: service, Milvus, BM25, rank fusion. Embeddings and the chat model are the in-memory fakes, so no network, no paid
  API and **no model latency**: a real deployment adds the embedding call (ingest, and one per query) and the generation call (query).
* **Milvus Lite** (embedded, single process, a local file), because standalone Milvus could not be started in this environment
  (image pull blocked). Lite serialises access, so **concurrency numbers here describe Lite, not Milvus standalone** and say nothing
  about how a real cluster scales. Run the same command with `AGENTIC_RAG_MILVUS_URI=http://host:19530` to measure standalone.
* One machine, one run per row; run-to-run noise on the two baseline rows is about 8 % (ingest) and 3 % (search).
* Warm search = 300 queries drawn from the corpus vocabulary after a 50-query warm-up; ingest is cold (empty collection), 4 concurrent uploads.

## Results

Machine: 4 vCPU, Python 3.11.15, Linux-6.18.44-fc-v70-x86_64-with-glibc2.39; Milvus: milvus-lite (embedded, single process). Corpus: 200 synthetic documents, 4,688,201 bytes, seed 1.

| Run | Ingest docs/s | Ingest chunks/s | Search c=1 p50 / p95 ms | c=4 p50 / p95 ms | c=16 p50 / p95 ms | Search req/s (c=1 / 4 / 16) |
|---|---|---|---|---|---|---|
| baseline (run 1) | 13.17 | 263.3 | 35.02 / 61.89 | 138.84 / 223.91 | 581.38 / 630.43 | 25.7 / 26.6 / 27.9 |
| baseline (run 2) | 14.23 | 284.5 | 33.87 / 57.23 | 146.45 / 216.57 | 573.76 / 644.72 | 26.6 / 25.8 / 27.6 |
| A: cache the index verification | 17.77 | 355.4 | 33.66 / 52.36 | 140.93 / 214.64 | 581.55 / 683.14 | 27.2 / 26.2 / 26.9 |
| A+B: two-phase commit marker, 500-row batches | 16.57 | 331.4 | 35.26 / 51.58 | 145.54 / 234.18 | 603.41 / 694.12 | 26.0 / 25.5 / 26.0 |

## Where the time goes (profile first, then change)

A direct timing of each retrieval stage on 1,200 indexed chunks (single caller, p50): Milvus Lite search **10.7 ms**, BM25 **0.67 ms**,
query embedding (fake) **0.01 ms**; end-to-end search p50 at one client is 12.9 ms. About 83 % of a search is the datastore call and the
application adds roughly 2 ms. Throughput stays at about 26 requests/s whether 1, 4 or 16 clients are active while latency grows in
proportion to the client count: **the ceiling is the datastore (Lite's serialised access), not the application.** On 4,000 chunks a
search costs about 34 ms, so Lite's cost grows with the collection.

## Changes, before, after, kept or reverted

| Change | Before | After | Verdict |
|---|---|---|---|
| A. Remember that the index was already verified for this embedding model and size (saves `describe_collection` + a row query on every ingest) | 13.2 / 14.2 docs/s | 17.8 docs/s (+30 %) | **kept** |
| B. Write chunk 0 (the document record and commit marker) as its own second upsert; upsert in 500-row batches | 17.8 docs/s | 16.6 docs/s (-7 %, inside noise) | **kept**: it costs one extra round trip but makes the marker mean what the code says it means, and keeps each gRPC message under Milvus's 64 MB limit |

Not tried because the profile says they cannot help here: more worker threads or a connection pool (Lite is single-process),
caching query embeddings (embedding is 0.01 ms with the fake and a network call with the real one: measure with a real provider first),
and lowering `candidate_pool` (a retrieval-quality trade-off that needs the evaluation harness, Step 18).

## Load shedding and limits (verified by tests, not by this benchmark)

`/v1` accepts at most `AGENTIC_RAG_MAX_INFLIGHT_REQUESTS` (default 64) concurrent requests; the next ones get `503` with an integer
`Retry-After` (`tests/api/test_operability.py`). Whether 64 is the right number for a given deployment is **unmeasured**.
