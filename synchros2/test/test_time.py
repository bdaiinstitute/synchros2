# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute.  All rights reserved.
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Generator

import pytest
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.context import Context
from rclpy.duration import Duration
from rclpy.exceptions import ROSInterruptException
from rclpy.time import Time

from synchros2.time import SteadyRate, as_proper_duration, as_proper_time


@pytest.fixture
def ros_context(domain_id: int) -> Generator[Context, None, None]:
    context = Context()
    rclpy.init(context=context, domain_id=domain_id)
    try:
        yield context
    finally:
        context.try_shutdown()


@pytest.fixture
def steady_clock() -> Clock:
    return Clock(clock_type=ClockType.STEADY_TIME)


def test_as_proper_time_from_int() -> None:
    t = as_proper_time(5)
    assert isinstance(t, Time)
    assert t.nanoseconds == 5_000_000_000


def test_as_proper_time_from_float() -> None:
    t = as_proper_time(1.5)
    assert isinstance(t, Time)
    assert t.nanoseconds == 1_500_000_000


def test_as_proper_time_from_datetime() -> None:
    dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    t = as_proper_time(dt)
    assert isinstance(t, Time)
    assert t.nanoseconds == int(dt.timestamp() * 1e9)


def test_as_proper_time_from_time() -> None:
    original = Time(seconds=42)
    t = as_proper_time(original)
    assert t is original


def test_as_proper_time_raises_on_invalid_type() -> None:
    with pytest.raises(ValueError):
        as_proper_time("not a time")  # type: ignore[arg-type]


def test_as_proper_duration_from_int() -> None:
    d = as_proper_duration(3)
    assert isinstance(d, Duration)
    assert d.nanoseconds == 3_000_000_000


def test_as_proper_duration_from_float() -> None:
    d = as_proper_duration(0.5)
    assert isinstance(d, Duration)
    assert d.nanoseconds == 500_000_000


def test_as_proper_duration_from_timedelta() -> None:
    td = timedelta(seconds=2, milliseconds=500)
    d = as_proper_duration(td)
    assert isinstance(d, Duration)
    assert d.nanoseconds == 2_500_000_000


def test_as_proper_duration_from_duration() -> None:
    original = Duration(seconds=7)
    d = as_proper_duration(original)
    assert d is original


def test_as_proper_duration_raises_on_invalid_type() -> None:
    with pytest.raises(ValueError):
        as_proper_duration("not a duration")  # type: ignore[arg-type]


def test_steady_rate_fires_at_expected_frequency(ros_context: Context, steady_clock: Clock) -> None:
    """SteadyRate sleep() fires at approximately the requested frequency."""
    frequency = 10.0  # Hz
    rate = SteadyRate(frequency, steady_clock, context=ros_context)

    iterations = 10
    start = time.monotonic()
    for _ in range(iterations):
        rate.sleep()
    elapsed = time.monotonic() - start

    expected = iterations / frequency
    # Allow ±5 % tolerance for CI timing jitter
    assert abs(elapsed - expected) / expected < 0.05, f"elapsed={elapsed:.3f}s, expected≈{expected:.3f}s"


def test_steady_rate_raises_on_context_shutdown(ros_context: Context, steady_clock: Clock) -> None:
    """SteadyRate.sleep() raises ROSInterruptException when the context is shut down."""
    rate = SteadyRate(0.1, steady_clock, context=ros_context)  # very slow

    exceptions: list = []

    def sleeper() -> None:
        try:
            rate.sleep()
        except ROSInterruptException as e:
            exceptions.append(e)

    worker = threading.Thread(target=sleeper)
    worker.start()
    time.sleep(0.05)  # let the thread enter sleep_until
    ros_context.try_shutdown()
    worker.join(timeout=2.0)

    assert not worker.is_alive(), "sleeper thread did not unblock after context shutdown"
    assert len(exceptions) == 1
    assert isinstance(exceptions[0], ROSInterruptException)


def test_steady_rate_raises_on_destroy(ros_context: Context, steady_clock: Clock) -> None:
    """SteadyRate.sleep() raises RuntimeError when called after destroy()."""
    rate = SteadyRate(10.0, steady_clock, context=ros_context)
    rate.destroy()

    with pytest.raises(RuntimeError):
        rate.sleep()
