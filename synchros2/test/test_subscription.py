# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute.  All rights reserved.
import itertools
import time
from typing import Any, Iterator, cast

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from std_msgs.msg import Int8, String

from synchros2.futures import wait_for_future
from synchros2.node import Node
from synchros2.scope import ROSAwareScope
from synchros2.subscription import Subscription, wait_for_message, wait_for_messages
from synchros2.utilities import ensure

DEFAULT_QOS_PROFILE = QoSProfile(
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_ALL,
    depth=1,
)


def test_wait_for_message(ros: ROSAwareScope) -> None:
    """Asserts that waiting for a message works as expected."""
    assert ros.node is not None
    pub = ros.node.create_publisher(Int8, "test", DEFAULT_QOS_PROFILE)
    assert not wait_for_message(Int8, "test", node=ros.node, timeout_sec=0.5)

    def deferred_publish() -> None:
        time.sleep(0.5)
        pub.publish(Int8(data=1))

    assert ros.executor is not None
    ros.executor.create_task(deferred_publish)
    message = wait_for_message(Int8, "test", node=ros.node, timeout_sec=5.0)
    assert message is not None
    assert message.data == 1


def test_subscription_matching_publishers(ros: ROSAwareScope) -> None:
    """Asserts that checking for publisher matching on a subscription works as expected."""
    assert ros.node is not None
    sequence = Subscription(Int8, "sequence", DEFAULT_QOS_PROFILE, node=ros.node)
    assert sequence.matched_publishers == 0
    future = sequence.publisher_matches(1)
    assert not future.done()
    future.cancel()

    ros.node.create_publisher(Int8, "sequence", DEFAULT_QOS_PROFILE)
    assert wait_for_future(sequence.publisher_matches(1), timeout_sec=5.0)
    assert sequence.matched_publishers == 1


def test_subscription_future_wait(ros: ROSAwareScope) -> None:
    """Asserts that waiting for a subscription update works as expected."""
    assert ros.node is not None
    pub = ros.node.create_publisher(Int8, "sequence", DEFAULT_QOS_PROFILE)
    sequence = Subscription(Int8, "sequence", DEFAULT_QOS_PROFILE, node=ros.node)
    assert wait_for_future(sequence.publisher_matches(1), timeout_sec=5.0)
    assert sequence.matched_publishers == 1

    pub.publish(Int8(data=1))

    assert wait_for_future(sequence.update, timeout_sec=5.0)
    assert cast(Int8, ensure(sequence.latest)).data == 1


def test_subscription_matching_future_wait(ros: ROSAwareScope) -> None:
    """Asserts that waiting for a matching subscription update works as expected."""
    assert ros.node is not None
    pub = ros.node.create_publisher(Int8, "sequence", DEFAULT_QOS_PROFILE)
    sequence = Subscription(Int8, "sequence", DEFAULT_QOS_PROFILE, node=ros.node)
    assert wait_for_future(sequence.publisher_matches(1), timeout_sec=5.0)
    assert sequence.matched_publishers == 1

    def deferred_publish() -> None:
        time.sleep(0.5)
        for num in range(5):
            pub.publish(Int8(data=num))

    assert ros.executor is not None
    ros.executor.create_task(deferred_publish)

    future = sequence.matching_update(lambda message: message.data == 3)
    assert wait_for_future(future, timeout_sec=5.0)
    message = future.result()
    assert message.data == 3


def test_subscription_iteration(ros: ROSAwareScope) -> None:
    """Asserts that iterating over subscription messages works as expected."""
    assert ros.node is not None

    pub = ros.node.create_publisher(
        Int8,
        "sequence",
        DEFAULT_QOS_PROFILE,
    )
    sequence = Subscription(
        Int8,
        "sequence",
        DEFAULT_QOS_PROFILE,
        history_length=3,
        node=ros.node,
    )
    assert wait_for_future(sequence.publisher_matches(1), timeout_sec=5.0)
    assert sequence.matched_publishers == 1

    expected_sequence_numbers = [1, 10, 100]

    def deferred_publish() -> None:
        time.sleep(0.5)
        for num in expected_sequence_numbers:
            pub.publish(Int8(data=num))

    assert ros.executor is not None
    ros.executor.create_task(deferred_publish)
    message_stream: Iterator[Int8] = sequence.stream(timeout_sec=5.0)

    streamed_numbers = [msg.data for msg in itertools.islice(message_stream, 3)]
    assert expected_sequence_numbers == streamed_numbers

    historic_numbers = [msg.data for msg in sequence.history]
    assert expected_sequence_numbers == historic_numbers


def test_deferred_start_subscription(ros: ROSAwareScope) -> None:
    """Asserts that deferred subscription start works as expected."""
    assert ros.node is not None
    pub = ros.node.create_publisher(Int8, "sequence", DEFAULT_QOS_PROFILE)
    sequence = Subscription(
        Int8,
        "sequence",
        DEFAULT_QOS_PROFILE,
        node=ros.node,
        autostart=False,
    )
    assert wait_for_future(sequence.publisher_matches(1), timeout_sec=5.0)
    assert sequence.matched_publishers == 1

    pub.publish(Int8(data=1))

    future = sequence.update
    assert not future.done()
    sequence.start()

    assert wait_for_future(future, timeout_sec=5.0)
    assert cast(Int8, ensure(sequence.latest)).data == 1


def test_subscription_cancelation(ros: ROSAwareScope) -> None:
    """Asserts that cancelling a subscription works as expected."""
    assert ros.node is not None
    pub = ros.node.create_publisher(Int8, "sequence", DEFAULT_QOS_PROFILE)
    sequence = Subscription(Int8, "sequence", DEFAULT_QOS_PROFILE, node=ros.node)
    assert wait_for_future(sequence.publisher_matches(1), timeout_sec=5.0)
    assert sequence.matched_publishers == 1

    pub.publish(Int8(data=1))

    assert wait_for_future(sequence.update, timeout_sec=5.0)

    def deferred_cancellation() -> None:
        time.sleep(0.5)
        sequence.unsubscribe()

    assert ros.executor is not None
    ros.executor.create_task(deferred_cancellation)

    streamed_numbers = [msg.data for msg in sequence.stream()]
    assert len(streamed_numbers) == 1
    assert streamed_numbers[0] == 1

    pub.publish(Int8(data=10))

    assert wait_for_future(sequence.update, timeout_sec=5.0)
    assert sequence.update.cancelled()

    historic_numbers = [msg.data for msg in sequence.history]
    assert len(historic_numbers) == 1
    assert historic_numbers[0] == 1

    assert cast(Int8, ensure(sequence.latest)).data == 1


def test_wait_for_messages(ros: ROSAwareScope) -> None:
    """Asserts that waiting for multiple synchronized messages works as expected."""

    class TestNode(Node):
        def __init__(self, name: str, **kwargs: Any) -> None:
            super().__init__(name, **kwargs)
            self.pub = self.create_publisher(String, "~/test", 10)
            self.timer = self.create_timer(0.5, self.do_publish)

        def do_publish(self) -> None:
            self.pub.publish(String(data=f"hello from {self.get_name()}"))

    ros.load(TestNode, "foo")
    ros.load(TestNode, "bar")

    messages = wait_for_messages(
        ["foo/test", "bar/test"],
        [String, String],
        timeout_sec=10.0,
        allow_headerless=True,
        delay=0.5,
        node=ros.node,
    )
    assert messages is None or messages == (
        String(data="hello from foo"),
        String(data="hello from bar"),
    )

    messages = wait_for_messages(
        ["bar/test", "foo/test"],
        [String, String],
        allow_headerless=True,
        delay=0.5,
        timeout_sec=10.0,
        node=ros.node,
    )
    assert messages is None or messages == (
        String(data="hello from bar"),
        String(data="hello from foo"),
    )

    # Test the case where the topic doesn't exist - timeout expected
    messages = wait_for_messages(
        ["/test", "foo/test"],
        [String, String],
        allow_headerless=True,
        timeout_sec=1.0,
        node=ros.node,
    )
    assert messages is None
