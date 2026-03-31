# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute.  All rights reserved.

import threading
from typing import Any, Generator

import pytest
import rclpy
from rcl_interfaces.srv import GetParameters
from rclpy.context import Context
from std_srvs.srv import Trigger

from synchros2.executors import AutoScalingMultiThreadedExecutor
from synchros2.node import Node
from synchros2.scope import ROSAwareScope


@pytest.fixture
def ros_context(domain_id: int) -> Generator[Context, None, None]:
    """A fixture yielding a managed rclpy.context.Context instance."""
    context = Context()
    rclpy.init(context=context, domain_id=domain_id)
    try:
        yield context
    finally:
        context.try_shutdown()


def test_node_destruction_during_execution(ros_context: Context) -> None:
    """Asserts that node destructionthe autoscaling multithreaded executor scales to attend a
    synchronous service call from a "one-shot" timer callback, serviced by
    the same executor.
    """

    def dummy_server_callback(_: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        response.success = True
        return response

    node = Node("pytest_node", context=ros_context)
    node.create_service(Trigger, "/dummy/trigger", dummy_server_callback)
    client = node.create_client(Trigger, "/dummy/trigger")

    executor = AutoScalingMultiThreadedExecutor(max_threads=1, context=ros_context)
    executor.add_node(node)

    barrier = threading.Barrier(2)
    try:
        # First smoke test the executor with a service invocation
        future = client.call_async(Trigger.Request())
        executor.spin_until_future_complete(future, timeout_sec=5.0)
        assert future.done() and future.result().success
        # Then block its sole worker thread
        executor.create_task(lambda: barrier.wait())
        while barrier.n_waiting == 0:
            executor.spin_once(0.1)
        # Then queue node destruction
        executor.create_task(lambda: node.destroy_node())
        assert not node.destruction_requested  # still queued
        # Then queue another service invocation
        future = client.call_async(Trigger.Request())
        # Unblock worker thread in executor
        barrier.wait()
        # Check that node destruction prevents call future completion
        executor.spin_until_future_complete(future, timeout_sec=2.0)
        assert bool(node.destruction_requested)
        assert executor.default_thread_pool.wait(timeout=5.0)
        assert not future.done()  # future response will never be resolved
    finally:
        barrier.reset()
        executor.remove_node(node)
        executor.shutdown()


def test_node_post_initialization(ros: ROSAwareScope) -> None:
    """Asserts that Node post initialization is honored and that it affords blocking calls."""

    class ComplexProxyNode(Node):
        def __init__(self, *args: Any, remote_node_name: str, **kwargs: Any) -> None:
            super().__init__(f"proxy_node_for_{remote_node_name}", *args, **kwargs)
            self._get_remote_node_parameters_client = self.create_client(
                GetParameters,
                f"{remote_node_name}/get_parameters",
            )

        def __post_init__(self) -> None:
            response = self._get_remote_node_parameters_client.call(
                GetParameters.Request(names=["verbose"]),
            )
            self.verbose = response.values[0].bool_value

    assert ros.node is not None
    ros.node.declare_parameter("verbose", True)
    with ros.managed(ComplexProxyNode, remote_node_name=ros.node.get_name()) as node:
        assert node.verbose
