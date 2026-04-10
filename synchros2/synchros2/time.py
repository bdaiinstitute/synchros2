# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute.  All rights reserved.

import threading
from datetime import datetime, timedelta
from typing import Optional, Union

from rclpy.context import Context
from rclpy.duration import Duration
from rclpy.exceptions import ROSInterruptException
from rclpy.time import Time
from rclpy.utilities import get_default_context


def as_proper_time(time: Union[int, float, datetime, Time]) -> Time:
    """Convert `time` to a proper, standardized `Time` object.

    For conversion, the following rules apply:
    - if an `int` or a `float` is provided, it is assumed to be a timestamp expressed in seconds.
    - if a `datetime` is provided, its UTC `datetime.timestamp()` is used.
    - if a `Duration` is provided, it is returned as is.

    Args:
        time: the time to be converted.

    Returns:
        an standardized `Time` object in ROS time, representing the given `time`.

    Raises:
        ValueError: if the given `time` is not of a supported type.
    """
    if isinstance(time, (int, float)):
        return Time(seconds=time)
    if isinstance(time, datetime):
        return Time(seconds=time.timestamp())
    if not isinstance(time, Time):
        raise ValueError(f"unsupported time type: {time}")
    return time


def as_proper_duration(duration: Union[int, float, timedelta, Duration]) -> Duration:
    """Convert `duration` to a proper, standardized `Duration` object.

    For conversion, the following rules apply:
    - if an `int` or a `float` is provided, it is assumed to be a duration expressed in seconds.
    - if a `timedelta` is provided, `timedelta.total_seconds()` are used.
    - if a `Duration` is provided, it is returned as is.

    Args:
        duration: the duration to be converted.

    Returns:
        an standardized `Duration` object representing the given `duration`.

    Raises:
        ValueError: if the given `duration` is not of a supported type.
    """
    if isinstance(duration, (int, float)):
        return Duration(seconds=duration)
    if isinstance(duration, timedelta):
        return Duration(seconds=duration.total_seconds())
    if not isinstance(duration, Duration):
        raise ValueError(f"unsupported duration type: {duration}")
    return duration


class SteadyRate:
    """An rclpy.Rate equivalent that uses clock functionality directly, without timer overhead."""

    def __init__(self, frequency: float, clock: Time, *, context: Optional[Context] = None) -> None:
        self._clock = clock
        if context is None:
            context = get_default_context()
        self._context = context
        self._period = as_proper_duration(1.0 / frequency)
        self._deadline = self._clock.now() + self._period

        self._lock = threading.Lock()
        self._num_sleepers = 0

        self._is_shutdown = False
        self._is_destroyed = False
        self._context.on_shutdown(self._on_shutdown)

    def _on_shutdown(self) -> None:
        self._is_shutdown = True
        self.destroy()

    def destroy(self) -> None:
        """Destroy the rate."""
        self._is_destroyed = True

    def _presleep(self) -> None:
        if self._is_shutdown:
            raise ROSInterruptException()
        if self._is_destroyed:
            raise RuntimeError("MonotonicRate cannot sleep because it has been destroyed")
        with self._lock:
            self._num_sleepers += 1

    def _postsleep(self) -> None:
        with self._lock:
            self._num_sleepers -= 1
            if self._num_sleepers == 0:
                now = self._clock.now()
                next_deadline = self._deadline + self._period
                if now < self._deadline or now > next_deadline:
                    next_deadline = now + self._period
                self._deadline = next_deadline
        if self._is_shutdown:
            self.destroy()
            raise ROSInterruptException()

    def sleep(self) -> None:
        """Block until the current period is over."""
        self._presleep()
        try:
            self._clock.sleep_until(self._deadline, context=self._context)
        finally:
            self._postsleep()
