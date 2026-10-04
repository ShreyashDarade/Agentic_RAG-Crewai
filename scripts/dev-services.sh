#!/usr/bin/env bash
# Start the real services the tests and the app need, and print the environment to use them.
#   scripts/dev-services.sh up      start (standalone via docker if reachable, else Milvus Lite)
#   scripts/dev-services.sh env     print `export` lines
#   scripts/dev-services.sh down    stop what `up` started
# Sessions restart: run `up` again; it is idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."
DEV_DIR=".dev"; mkdir -p "$DEV_DIR"
compose=(docker compose -f deploy/milvus-standalone.yml -p agentic-rag-dev)

have_docker() { command -v docker >/dev/null && docker info >/dev/null 2>&1; }

cmd="${1:-up}"
case "$cmd" in
  up)
    if [[ -n "${AGENTIC_RAG_MILVUS_URI:-}" ]]; then
      echo "using AGENTIC_RAG_MILVUS_URI from the environment" >&2
    elif have_docker && "${compose[@]}" up -d --wait >&2 2>"$DEV_DIR/compose.err"; then
      echo "http://127.0.0.1:19530" > "$DEV_DIR/milvus.uri"; echo "milvus: standalone (docker)" >&2
    else
      [[ -s "$DEV_DIR/compose.err" ]] && echo "standalone unavailable: $(tail -1 "$DEV_DIR/compose.err")" >&2
      echo "$PWD/$DEV_DIR/milvus-lite.db" > "$DEV_DIR/milvus.uri"; echo "milvus: Milvus Lite (embedded)" >&2
    fi ;;
  env)
    uri="${AGENTIC_RAG_MILVUS_URI:-$(cat "$DEV_DIR/milvus.uri" 2>/dev/null || true)}"
    [[ -n "$uri" ]] || { echo "run: scripts/dev-services.sh up" >&2; exit 1; }
    echo "export AGENTIC_RAG_MILVUS_URI=$uri" ;;
  down)
    have_docker && "${compose[@]}" down -v >&2 || true
    rm -f "$DEV_DIR/milvus.uri" ;;
  *) echo "usage: $0 up|env|down" >&2; exit 2 ;;
esac
