# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute.  All rights reserved.

import threading
import time
from typing import Any

import pytest
from rclpy.task import Future

from synchros2.futures import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    FIRST_EXCEPTION,
    unwrap_future,
    wait_for_future,
)
from synchros2.scope import ROSAwareScope


def test_wait_for_cancelled_future(ros: ROSAwareScope) -> None:
    """Cancelled futures should not hang."""
    future = Future()
    future.cancel()
    result = wait_for_future(future, context=ros.context)
    assert future in result.done


def test_wait_for_single_future(ros: ROSAwareScope) -> None:
    """Wait for a single future to complete."""
    future = Future()

    def complete_later() -> None:
        time.sleep(0.05)
        future.set_result(42)

    threading.Thread(target=complete_later, daemon=True).start()

    result = wait_for_future(future, timeout_sec=1.0, context=ros.context)
    assert result
    assert future in result.done
    assert future.result() == 42


def test_wait_for_all_futures(ros: ROSAwareScope) -> None:
    """Wait for all futures to complete."""
    futures = [Future() for _ in range(3)]

    def complete_all() -> None:
        for i, f in enumerate(futures):
            time.sleep(0.02)
            f.set_result(i)

    threading.Thread(target=complete_all, daemon=True).start()

    result = wait_for_future(futures, timeout_sec=1.0, context=ros.context, return_when=ALL_COMPLETED)
    assert result
    assert len(result.done) == 3
    assert len(result.not_done) == 0


def test_wait_for_first_future(ros: ROSAwareScope) -> None:
    """Return as soon as first future completes."""
    futures = [Future() for _ in range(3)]

    def complete_first() -> None:
        time.sleep(0.02)
        futures[0].set_result(0)

    threading.Thread(target=complete_first, daemon=True).start()

    result = wait_for_future(futures, timeout_sec=1.0, context=ros.context, return_when=FIRST_COMPLETED)
    assert result
    assert len(result.done) >= 1
    assert futures[0] in result.done


def test_wait_for_first_exception(ros: ROSAwareScope) -> None:
    """Return when first exception occurs."""
    futures = [Future() for _ in range(3)]

    def set_exception() -> None:
        time.sleep(0.02)
        futures[1].set_exception(RuntimeError("oops"))

    threading.Thread(target=set_exception, daemon=True).start()

    result = wait_for_future(futures, timeout_sec=1.0, context=ros.context, return_when=FIRST_EXCEPTION)
    assert result
    assert futures[1] in result.done
    assert futures[1].exception() is not None


def test_wait_timeout(ros: ROSAwareScope) -> None:
    """Timeout when futures don't complete in time."""
    futures = [Future() for _ in range(2)]
    futures[0].set_result(0)
    # futures[1] never completes

    result = wait_for_future(futures, timeout_sec=0.05, context=ros.context, return_when=ALL_COMPLETED)
    assert not result
    assert len(result.done) == 1
    assert len(result.not_done) == 1

    futures[1].set_result(1)  # cleanup


def test_unwrap_single_future(ros: ROSAwareScope) -> None:
    """Unwrap a single future's result."""
    future = Future()
    future.set_result(42)

    result = unwrap_future(future, context=ros.context)
    assert result == 42


def test_unwrap_single_future_timeout(ros: ROSAwareScope) -> None:
    """Unwrapping should timeout if future doesn't complete."""
    future = Future()

    with pytest.raises(ValueError):
        unwrap_future(future, timeout_sec=0.05, context=ros.context)

    future.set_result(0)  # cleanup


def test_unwrap_multiple_futures_as_completed(ros: ROSAwareScope) -> None:
    """Unwrap multiple futures as they complete (via callback chain)."""
    futures = [Future(), Future(), Future()]

    # Chain completions: f0 completes, then f1, then f2
    def chain_completions() -> None:
        time.sleep(0.02)
        futures[0].set_result(10)

        def complete_f1(_: Any) -> None:
            time.sleep(0.02)
            futures[1].set_result(20)

        def complete_f2(_: Any) -> None:
            time.sleep(0.02)
            futures[2].set_result(30)

        futures[0].add_done_callback(complete_f1)
        futures[1].add_done_callback(complete_f2)

    threading.Thread(target=chain_completions, daemon=True).start()

    results = list(unwrap_future(futures, timeout_sec=1.0, context=ros.context))
    assert set(results) == {10, 20, 30}


def test_unwrap_strict_mode_preserves_order(ros: ROSAwareScope) -> None:
    """Strict unwrapping yields results in original order."""
    futures = [Future(), Future(), Future()]

    # Complete out of order: f2, then f0, then f1
    def complete_out_of_order() -> None:
        time.sleep(0.02)
        futures[2].set_result(30)

        def complete_f0(_: Any) -> None:
            time.sleep(0.02)
            futures[0].set_result(10)

        def complete_f1(_: Any) -> None:
            time.sleep(0.02)
            futures[1].set_result(20)

        futures[2].add_done_callback(complete_f0)
        futures[0].add_done_callback(complete_f1)

    threading.Thread(target=complete_out_of_order, daemon=True).start()

    results = list(unwrap_future(futures, timeout_sec=1.0, context=ros.context, strict=True))
    assert results == [10, 20, 30]  # In order despite out-of-order completion


def test_unwrap_propagates_exceptions(ros: ROSAwareScope) -> None:
    """Exceptions from futures should propagate through unwrap."""
    futures = [Future(), Future()]
    futures[0].set_result(42)
    futures[1].set_exception(RuntimeError("boom"))

    results = []
    with pytest.raises(RuntimeError, match="boom"):
        for result in unwrap_future(futures, strict=True, context=ros.context):
            results.append(result)

    assert 42 in results


def test_unwrap_timeout_with_multiple_futures(ros: ROSAwareScope) -> None:
    """Unwrap should timeout if not all futures complete."""
    futures = [Future(), Future()]
    futures[0].set_result(42)
    # futures[1] never completes

    with pytest.raises(ValueError):
        list(unwrap_future(futures, timeout_sec=0.05, context=ros.context))

    futures[1].set_result(0)  # cleanup


def test_unwrap_empty_list(ros: ROSAwareScope) -> None:
    """Unwrapping empty list should work."""
    results = list(unwrap_future([], context=ros.context))
    assert results == []


def test_wait_for_empty_list(ros: ROSAwareScope) -> None:
    """Waiting for empty list should succeed immediately."""
    result = wait_for_future([], context=ros.context)
    assert result
