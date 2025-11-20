# Copyright (c) 2023 Robotics and AI Institute LLC dba RAI Institute.  All rights reserved.

import logging
import threading
from typing import List

from rcl_interfaces.msg import Log
from rclpy.clock import ROSClock
from rclpy.time import Time

from synchros2.futures import unwrap_future
from synchros2.logging import LoggingSeverity, logs_to_ros
from synchros2.scope import ROSAwareScope
from synchros2.subscription import Subscription


def test_memoizing_logger(verbose_ros: ROSAwareScope) -> None:
    messages: List[str] = []
    cv = threading.Condition()

    def callback(message: Log) -> None:
        nonlocal messages, cv
        with cv:
            messages.append(message.msg)
            cv.notify()

    assert verbose_ros.node is not None
    verbose_ros.node.create_subscription(Log, "/rosout", callback, 10)

    logger = verbose_ros.node.get_logger()
    logger.set_level(LoggingSeverity.INFO)

    assert not logger.debug("Debug message should not be logged")

    for i in range(2):
        is_first_iteration = i == 0
        did_log_once = logger.error(
            f"Error message should have been logged only if {i} == 0",
            once=True,
        )
        assert did_log_once == is_first_iteration
        did_skip_first_log = logger.error(
            f"Error message should have been logged only if {i} != 0",
            skip_first=True,
        )
        assert did_skip_first_log != is_first_iteration

    assert logger.warning("Warning message always logged", once=True)
    assert logger.warning("Warning message always logged", once=True)

    fake_clock = ROSClock()
    fake_clock._set_ros_time_is_active(True)

    num_throttled_logs = 2
    num_attempts_per_sec = 5
    for i in range(num_attempts_per_sec * num_throttled_logs):
        fake_clock.set_ros_time_override(Time(seconds=float(i) / num_attempts_per_sec))
        assert logger.info(
            "Info message should be throttled",
            throttle_duration_sec=1.0,
            throttle_time_source_type=fake_clock,
        ) == (i % num_attempts_per_sec == 0)

    expected_messages = [
        "Error message should have been logged only if 0 == 0",
        "Error message should have been logged only if 1 != 0",
        "Warning message always logged",
        "Warning message always logged",
    ] + ["Info message should be throttled"] * num_throttled_logs

    def all_messages_arrived() -> bool:
        return len(messages) == len(expected_messages)

    with cv:
        assert cv.wait_for(all_messages_arrived, timeout=5.0)
    assert messages == expected_messages


def test_log_forwarding(verbose_ros: ROSAwareScope) -> None:
    assert verbose_ros.node is not None
    rosout = Subscription(Log, "/rosout", 10, node=verbose_ros.node)
    assert unwrap_future(rosout.publisher_matches(1), timeout_sec=5.0) > 0

    with logs_to_ros(verbose_ros.node):
        logger = logging.getLogger("my_logger")
        logger.setLevel(logging.INFO)
        logger.propagate = True  # ensure propagation is enabled
        for i in range(5):
            logger.info("test #%d", i)
        log = unwrap_future(rosout.latest_update, timeout_sec=5.0)
        expected_level = Log.INFO
        if isinstance(expected_level, bytes):
            # NOTE(hidmic): log levels are of bytestring type in earlier distros
            expected_level = int.from_bytes(expected_level, byteorder="little")
        assert log.level == expected_level
        assert log.name == verbose_ros.node.get_logger().name
        assert log.msg.startswith("(logging.my_logger) test")


def test_two_tiered_log_forwarding(verbose_ros: ROSAwareScope) -> None:
    assert verbose_ros.node is not None
    rosout = Subscription(Log, "/rosout", 10, node=verbose_ros.node)
    assert unwrap_future(rosout.publisher_matches(1), timeout_sec=5.0) > 0

    with logs_to_ros(verbose_ros.node):
        logger = logging.getLogger()
        logger.propagate = True  # ensure propagation is enabled
        with logs_to_ros(verbose_ros.node, "my_verbose_logger", logging.DEBUG, propagate=False):
            verbose_logger = logging.getLogger("my_verbose_logger")
            for i in range(5):
                verbose_logger.info("test #%d", i)
            expected_level = Log.INFO
            if isinstance(expected_level, bytes):
                # NOTE(hidmic): log levels are of bytestring type in earlier distros
                expected_level = int.from_bytes(expected_level, byteorder="little")
            for log in rosout.stream(duration_sec=1.0):
                assert log.level == expected_level
                ros_logger = verbose_ros.node.get_logger()
                assert log.name == ros_logger.get_child("my_verbose_logger").name
                assert log.msg.startswith("(logging.my_verbose_logger) test")
