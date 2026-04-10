# Copyright (c) 2023 Robotics and AI Institute LLC dba RAI Institute.  All rights reserved.
import functools
import logging
import threading
import time
from typing import Generator, List

import pytest
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from synchros2.executors import AutoScalingMultiThreadedExecutor, AutoScalingThreadPool, background, foreground
from synchros2.futures import unwrap_future, wait_for_future


@pytest.fixture
def ros_context(domain_id: int) -> Generator[Context, None, None]:
    """A fixture yielding a managed rclpy.context.Context instance."""
    context = Context()
    rclpy.init(context=context, domain_id=domain_id)
    try:
        yield context
    finally:
        context.try_shutdown()


@pytest.fixture
def ros_node(ros_context: Context) -> Generator[Node, None, None]:
    """A fixture yielding a managed rclpy.node.Node instance."""
    node = rclpy.create_node("pytest_node", context=ros_context)
    try:
        yield node
    finally:
        node.destroy_node()


def test_autoscaling_thread_pool() -> None:
    """Asserts that the autoscaling thread pool scales and de-scales on demand."""
    with AutoScalingThreadPool(max_idle_time=2.0, logger=logging.root) as pool:
        assert len(pool.workers) == 0
        assert not bool(pool.working)

        events = [threading.Event() for _ in range(10)]
        results = pool.map(lambda i, e: e.wait() and i, range(10), events)
        assert len(pool.workers) == len(events)
        assert bool(pool.working)
        assert not pool.capped

        for e in events:
            e.set()
        assert list(results) == list(range(10))

        def predicate() -> bool:
            return len(pool.workers) == 0

        with pool.scaling_event:
            assert pool.scaling_event.wait_for(predicate, timeout=10)
        assert not bool(pool.working)


def test_autoscaling_thread_pool_checks_arguments() -> None:
    """Asserts that the autoscaling thread pool checks for valid arguments on construction."""
    with pytest.raises(ValueError):
        AutoScalingThreadPool(min_workers=-1)

    with pytest.raises(ValueError):
        AutoScalingThreadPool(max_workers=0)

    with pytest.raises(ValueError):
        AutoScalingThreadPool(max_idle_time=-1)

    with pytest.raises(ValueError):
        AutoScalingThreadPool(submission_quota=0)

    with pytest.raises(ValueError):
        AutoScalingThreadPool(submission_patience=-1)


def test_autoscaling_thread_pool_when_shutdown() -> None:
    """Asserts that the autoscaling thread no longer accepts work when shutdown."""
    pool = AutoScalingThreadPool()

    pool.shutdown()

    with pytest.raises(RuntimeError):
        pool.submit(lambda: None)


def test_autoscaling_thread_pool_with_limits() -> None:
    """Asserts that the autoscaling thread pool enforces the user-defined range on the number of workers."""
    with AutoScalingThreadPool(min_workers=2, max_workers=5, max_idle_time=0.1, logger=logging.root) as pool:
        assert len(pool.workers) == 2
        assert bool(not pool.working)

        events = [threading.Event() for _ in range(10)]
        results = pool.map(lambda i, e: e.wait() and i, range(10), events)
        assert len(pool.workers) == 5
        assert bool(pool.working)

        for e in events:
            e.set()
        assert list(results) == list(range(10))

        def predicate() -> bool:
            return len(pool.workers) == 2

        with pool.scaling_event:
            assert pool.scaling_event.wait_for(predicate, timeout=10)
        assert not pool.working


def test_autoscaling_thread_pool_with_quota() -> None:
    """Asserts that the autoscaling thread pool respects submission quotas."""
    with AutoScalingThreadPool(
        submission_quota=5,
        submission_patience=1.0,
        max_idle_time=2.0,
        logger=logging.root,
    ) as pool:
        assert len(pool.workers) == 0
        assert not bool(pool.working)

        events = [threading.Event() for _ in range(10)]
        results = pool.map(lambda i, e: e.wait() and i, range(10), events)
        assert len(pool.workers) == 5
        assert bool(pool.working)
        assert pool.capped

        for e in events[:5]:
            e.set()
            assert len(pool.workers) == 5

        for e in events[5:]:
            e.set()
            assert len(pool.workers) == 5

        assert list(results) == list(range(10))
        assert len(pool.workers) == 5

        def predicate() -> bool:
            return len(pool.workers) == 0

        with pool.scaling_event:
            assert pool.scaling_event.wait_for(predicate, timeout=10)
        assert not pool.working


def test_autoscaling_executor(ros_context: Context, ros_node: Node) -> None:
    """Asserts that the autoscaling multithreaded executor scales to attend a
    synchronous service call from a "one-shot" task callback, serviced by
    the same executor.
    """

    def dummy_server_callback(_: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        response.success = True
        return response

    ros_node.create_service(
        Trigger,
        "/dummy/trigger",
        dummy_server_callback,
        callback_group=MutuallyExclusiveCallbackGroup(),
    )

    client = ros_node.create_client(Trigger, "/dummy/trigger", callback_group=MutuallyExclusiveCallbackGroup())

    with foreground(AutoScalingMultiThreadedExecutor(context=ros_context, logger=logging.root)) as executor:
        assert len(executor.default_thread_pool.workers) == 0
        assert not executor.default_thread_pool.working
        executor.add_node(ros_node)
        try:
            future = executor.create_task(
                lambda: unwrap_future(
                    client.call_async(Trigger.Request()),
                    timeout_sec=2.0,
                    context=ros_context,
                ),
            )
            executor.spin_until_future_complete(future, timeout_sec=5)
            response = future.result()
            assert response.success
            assert executor.default_thread_pool.wait(timeout=10)
            assert not executor.default_thread_pool.working
        finally:
            executor.remove_node(ros_node)


def test_autoscaling_executor_with_callback_group_affinity(ros_context: Context, ros_node: Node) -> None:
    """Asserts that the autoscaling multithreaded executor handles callback group affinity properly"""

    with background(AutoScalingMultiThreadedExecutor(context=ros_context, logger=logging.root)) as executor:
        executor.add_node(ros_node)

        def slow_thread_tracker(threads: List[threading.Thread]) -> None:
            threads.append(threading.current_thread())
            time.sleep(0.2)

        timer_period_sec = 0.1

        default_callback_group = ReentrantCallbackGroup()
        default_thread_pool_threads: List[threading.Thread] = []
        timer_callback = functools.partial(slow_thread_tracker, default_thread_pool_threads)
        ros_node.create_timer(timer_period_sec, timer_callback, default_callback_group)

        callback_group = ReentrantCallbackGroup()
        thread_pool = executor.add_static_thread_pool(num_threads=1)
        executor.bind(callback_group, thread_pool)

        static_thread_pool_threads: List[threading.Thread] = []
        timer_callback = functools.partial(slow_thread_tracker, static_thread_pool_threads)
        ros_node.create_timer(timer_period_sec, timer_callback, callback_group)

        time.sleep(1.0)

    assert len(static_thread_pool_threads) > 0
    assert len(default_thread_pool_threads) > len(static_thread_pool_threads)
    assert all(thread is static_thread_pool_threads[0] for thread in static_thread_pool_threads[1:])
    assert not any(thread is static_thread_pool_threads[0] for thread in default_thread_pool_threads)


def test_background_executor(ros_context: Context) -> None:
    """Asserts that an executor can be safely pushed to a background thread."""
    with background(SingleThreadedExecutor(context=ros_context)) as executor:
        with pytest.raises(RuntimeError):
            executor.spin()

        def deferred() -> bool:
            time.sleep(1.0)
            return True

        future = executor.create_task(deferred)

        with pytest.raises(RuntimeError):
            executor.spin_until_future_complete(future)

        assert wait_for_future(future, timeout_sec=10.0, context=ros_context)
        assert future.result()


def test_autoscaling_executor_with_timers_thread_pool(ros_context: Context, ros_node: Node) -> None:
    """Asserts that the autoscaling multithreaded executor routes timer callbacks to the
    dedicated timers thread pool and leaves non-timer work to the default thread pool.
    """
    with background(
        AutoScalingMultiThreadedExecutor(
            context=ros_context,
            num_threads_for_timers=1,
            logger=logging.root,
        ),
    ) as executor:
        assert executor.timers_thread_pool is not None
        executor.add_node(ros_node)

        timer_threads: List[threading.Thread] = []
        task_threads: List[threading.Thread] = []

        def timer_callback() -> None:
            timer_threads.append(threading.current_thread())

        ros_node.create_timer(0.05, timer_callback, ReentrantCallbackGroup())

        def task_callback() -> None:
            task_threads.append(threading.current_thread())
            time.sleep(0.05)

        for _ in range(5):
            executor.create_task(task_callback)

        time.sleep(1.0)

    assert len(timer_threads) > 0
    assert len(task_threads) > 0
    # All timer callbacks must run on the same single timers-pool thread
    assert all(t is timer_threads[0] for t in timer_threads[1:])
    # Task callbacks must never have run on the timers-pool thread
    assert not any(t is timer_threads[0] for t in task_threads)


@pytest.mark.filterwarnings("ignore")
def test_background_executor_shows_errors(ros_context: Context, ros_node: Node) -> None:
    """Asserts that an background executor does not swallow callback exceptions."""
    with background(
        AutoScalingMultiThreadedExecutor(
            context=ros_context,
            logger=logging.root,
        ),
    ) as executor:
        executor.add_node(ros_node)

        def callback() -> None:
            raise RuntimeError("failed timer")

        ros_node.create_timer(0.1, callback)

        with pytest.warns(RuntimeWarning):
            time.sleep(1.0)
