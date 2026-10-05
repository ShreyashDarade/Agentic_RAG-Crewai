"""The pool that keeps a hung dependency from starving the process (review A1)."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from agentic_rag.blocking import BlockingPool
from agentic_rag.errors import VectorStoreUnavailable


def _pool(workers: int = 2, backlog: int = 1) -> BlockingPool:
    return BlockingPool("test", workers=workers, backlog=backlog, saturated=lambda: VectorStoreUnavailable("full"))


async def test_runs_callables_and_propagates_results_and_exceptions() -> None:
    pool = _pool()
    assert await pool.run(lambda a, b=0: a + b, 1, b=2) == 3
    with pytest.raises(KeyError):
        await pool.run(lambda: {}["x"])
    assert pool.active == 0
    pool.close()


async def test_a_cancelled_await_does_not_free_the_slot_until_the_thread_returns() -> None:
    pool = _pool(workers=1, backlog=0)
    release = threading.Event()
    started = threading.Event()

    def hang() -> None:
        started.set()
        release.wait(10)

    task = asyncio.create_task(pool.run(hang))
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pool.active == 1  # the thread is still blocked: the slot is still taken
    with pytest.raises(VectorStoreUnavailable):  # fail fast instead of queueing behind it
        await pool.run(lambda: 1)
    release.set()
    for _ in range(100):
        if pool.active == 0:
            break
        await asyncio.sleep(0.01)
    assert pool.active == 0
    assert await pool.run(lambda: 7) == 7
    pool.close()


async def test_a_call_cancelled_while_queued_never_runs_and_frees_its_slot() -> None:
    pool = _pool(workers=1, backlog=2)
    release = threading.Event()
    ran: list[str] = []
    blocker = asyncio.create_task(pool.run(release.wait, 10))
    await asyncio.sleep(0.05)
    queued = asyncio.create_task(pool.run(lambda: ran.append("queued")))
    await asyncio.sleep(0.05)
    assert pool.active == 2
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert pool.active == 1
    release.set()
    await blocker
    await asyncio.sleep(0.05)
    assert ran == []
    pool.close()


async def test_a_hung_pool_does_not_stop_other_pools_or_the_default_executor() -> None:
    hung = _pool(workers=2, backlog=0)
    other = _pool(workers=1, backlog=0)
    release = threading.Event()
    tasks = [asyncio.create_task(hung.run(release.wait, 10)) for _ in range(2)]
    await asyncio.sleep(0.05)
    started = time.monotonic()
    assert await other.run(lambda: "fine") == "fine"
    assert await asyncio.to_thread(lambda: "also fine") == "also fine"
    assert time.monotonic() - started < 1
    release.set()
    await asyncio.gather(*tasks)
    hung.close()
    other.close()


async def test_workers_are_daemon_threads_so_a_stuck_call_cannot_block_process_exit() -> None:
    pool = _pool(workers=1, backlog=0)
    release = threading.Event()
    task = asyncio.create_task(pool.run(release.wait, 10))
    await asyncio.sleep(0.05)
    worker = next(t for t in threading.enumerate() if t.name == "test-0")
    assert worker.daemon
    release.set()
    await task
    pool.close()


async def test_a_closed_pool_refuses_work() -> None:
    pool = _pool()
    pool.close()
    with pytest.raises(VectorStoreUnavailable):
        await pool.run(lambda: 1)


async def test_an_abandoned_pool_does_not_leak_threads() -> None:
    import gc

    before = threading.active_count()
    for _ in range(20):
        pool = _pool(workers=2, backlog=0)
        await pool.run(lambda: 1)
        del pool
    gc.collect()
    for _ in range(100):
        if threading.active_count() <= before + 1:
            break
        await asyncio.sleep(0.02)
    assert threading.active_count() <= before + 1
