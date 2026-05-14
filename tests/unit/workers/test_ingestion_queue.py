"""Tests for the IngestionQueue bounded-concurrency system.

Tests cover:
- Single task submission and execution
- Concurrency limiting (semaphore enforcement)
- Task ordering (waiting tasks run after slot freed)
- Error isolation (one failure doesn't block others)
- Configurable concurrency limit
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.workers.ingestion_queue import IngestionQueue


# ── Helpers ──────────────────────────────────────────────────────


def _mock_deps() -> dict:
    """Return a dict of mock dependencies for submit()."""
    return {
        "embedding_service": AsyncMock(),
        "vector_store": AsyncMock(),
        "llm_provider": AsyncMock(),
    }


# ── Tests ────────────────────────────────────────────────────────


class TestIngestionQueueInit:
    """Initialization and configuration."""

    def test_default_concurrency(self) -> None:
        queue = IngestionQueue()
        assert queue.max_concurrent == 2

    def test_custom_concurrency(self) -> None:
        queue = IngestionQueue(max_concurrent=5)
        assert queue.max_concurrent == 5

    def test_single_concurrency(self) -> None:
        queue = IngestionQueue(max_concurrent=1)
        assert queue.max_concurrent == 1


class TestIngestionQueueSubmit:
    """Task submission and execution."""

    @pytest.mark.asyncio
    async def test_single_task_executes(self) -> None:
        """A single submitted task should run the ingestion worker."""
        queue = IngestionQueue(max_concurrent=2)

        with patch(
            "app.workers.ingestion_worker.run_ingestion_task",
            new_callable=AsyncMock,
        ) as mock_run:
            await queue.submit(
                document_id="doc-1",
                file_content=b"hello",
                tenant_id="t-1",
                **_mock_deps(),
            )
            # Let the background task run
            await asyncio.sleep(0.1)

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args
            assert call_kwargs.kwargs["document_id"] == "doc-1"
            assert call_kwargs.kwargs["tenant_id"] == "t-1"
            assert call_kwargs.kwargs["file_content"] == b"hello"

    @pytest.mark.asyncio
    async def test_submit_returns_immediately(self) -> None:
        """submit() should return immediately, not block until ingestion finishes."""
        slow_event = asyncio.Event()

        async def slow_task(**kwargs: object) -> None:
            await slow_event.wait()

        queue = IngestionQueue(max_concurrent=1)

        with patch(
            "app.workers.ingestion_worker.run_ingestion_task",
            side_effect=slow_task,
        ):
            # submit() should return instantly
            await queue.submit(
                document_id="doc-1",
                file_content=b"data",
                tenant_id="t-1",
                **_mock_deps(),
            )
            # If we get here, submit didn't block — test passes
            slow_event.set()
            await asyncio.sleep(0.1)


class TestIngestionQueueConcurrency:
    """Semaphore-bounded concurrency enforcement."""

    @pytest.mark.asyncio
    async def test_limits_concurrent_tasks(self) -> None:
        """Only max_concurrent tasks should run simultaneously."""
        max_concurrent = 2
        queue = IngestionQueue(max_concurrent=max_concurrent)
        active_count = 0
        max_observed = 0
        task_started = asyncio.Event()
        gate = asyncio.Event()

        async def tracked_task(**kwargs: object) -> None:
            nonlocal active_count, max_observed
            active_count += 1
            max_observed = max(max_observed, active_count)
            task_started.set()
            await gate.wait()
            active_count -= 1

        with patch(
            "app.workers.ingestion_worker.run_ingestion_task",
            side_effect=tracked_task,
        ):
            # Submit 4 tasks with max_concurrent=2
            for i in range(4):
                await queue.submit(
                    document_id=f"doc-{i}",
                    file_content=b"data",
                    tenant_id="t-1",
                    **_mock_deps(),
                )

            # Wait for first batch to start
            await asyncio.sleep(0.2)

            # At most 2 should be running
            assert active_count <= max_concurrent

            # Release all
            gate.set()
            await asyncio.sleep(0.3)

            # All should have completed
            assert active_count == 0
            # Never exceeded the limit
            assert max_observed <= max_concurrent

    @pytest.mark.asyncio
    async def test_queued_tasks_run_after_slot_freed(self) -> None:
        """Tasks waiting on the semaphore should run after a slot is freed."""
        queue = IngestionQueue(max_concurrent=1)
        execution_order: list[str] = []
        gates: dict[str, asyncio.Event] = {
            "doc-0": asyncio.Event(),
            "doc-1": asyncio.Event(),
        }

        async def ordered_task(**kwargs: object) -> None:
            doc_id = kwargs["document_id"]
            execution_order.append(f"start-{doc_id}")
            await gates[doc_id].wait()
            execution_order.append(f"end-{doc_id}")

        with patch(
            "app.workers.ingestion_worker.run_ingestion_task",
            side_effect=ordered_task,
        ):
            # Submit 2 tasks with max_concurrent=1
            await queue.submit(
                document_id="doc-0",
                file_content=b"first",
                tenant_id="t-1",
                **_mock_deps(),
            )
            await queue.submit(
                document_id="doc-1",
                file_content=b"second",
                tenant_id="t-1",
                **_mock_deps(),
            )

            await asyncio.sleep(0.1)
            # doc-0 should be running, doc-1 waiting
            assert "start-doc-0" in execution_order
            assert "start-doc-1" not in execution_order

            # Release doc-0
            gates["doc-0"].set()
            await asyncio.sleep(0.2)

            # doc-1 should now be running
            assert "end-doc-0" in execution_order
            assert "start-doc-1" in execution_order

            # Release doc-1
            gates["doc-1"].set()
            await asyncio.sleep(0.1)
            assert "end-doc-1" in execution_order


class TestIngestionQueueErrorIsolation:
    """Errors in one task should not block subsequent tasks."""

    @pytest.mark.asyncio
    async def test_failed_task_releases_slot(self) -> None:
        """A task that raises an exception should still release its semaphore slot."""
        queue = IngestionQueue(max_concurrent=1)
        results: list[str] = []

        call_count = 0

        async def failing_then_succeeding(**kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                results.append("failed")
                raise RuntimeError("Simulated failure")
            results.append("succeeded")

        with patch(
            "app.workers.ingestion_worker.run_ingestion_task",
            side_effect=failing_then_succeeding,
        ):
            await queue.submit(
                document_id="doc-fail",
                file_content=b"bad",
                tenant_id="t-1",
                **_mock_deps(),
            )
            await queue.submit(
                document_id="doc-ok",
                file_content=b"good",
                tenant_id="t-1",
                **_mock_deps(),
            )

            # Wait for both to complete
            await asyncio.sleep(0.5)

            # The second task should have run despite the first failing
            assert "succeeded" in results

    @pytest.mark.asyncio
    async def test_multiple_failures_dont_deadlock(self) -> None:
        """Multiple consecutive failures should not exhaust the semaphore."""
        queue = IngestionQueue(max_concurrent=1)
        completed = []

        call_count = 0

        async def all_fail_then_succeed(**kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError(f"Failure {call_count}")
            completed.append(kwargs["document_id"])

        with patch(
            "app.workers.ingestion_worker.run_ingestion_task",
            side_effect=all_fail_then_succeed,
        ):
            for i in range(3):
                await queue.submit(
                    document_id=f"doc-{i}",
                    file_content=b"data",
                    tenant_id="t-1",
                    **_mock_deps(),
                )

            await asyncio.sleep(0.5)

            # Third task should have succeeded
            assert "doc-2" in completed
