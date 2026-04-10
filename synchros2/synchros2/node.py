# Copyright (c) 2023 Robotics and AI Institute LLC dba RAI Institute.  All rights reserved.
import contextlib
import functools
from typing import Any, Callable, Iterable, Optional, Type

from rclpy.callback_groups import CallbackGroup
from rclpy.clock import Clock
from rclpy.exceptions import InvalidHandle
from rclpy.node import Node as BaseNode
from rclpy.waitable import Waitable

from synchros2.callback_groups import NonReentrantCallbackGroup
from synchros2.logging import MemoizingRcutilsLogger
from synchros2.time import SteadyRate


def suppressed(exception: Type[BaseException], func: Callable) -> Callable:
    """Suppress the given `exception` type from `func` invocations"""

    @functools.wraps(func)
    def __wrapper(*args: Any, **kwargs: Any) -> Any:
        with contextlib.suppress(exception):
            return func(*args, **kwargs)

    return __wrapper


class Node(BaseNode):
    """An rclpy.node.Node subclass that:

    * changes the default callback group to be non-reentrant
    * wraps its logger with a memoizing one for improved efficiency
    """

    def __init__(self, *args: Any, default_callback_group: Optional[CallbackGroup] = None, **kwargs: Any) -> None:
        """Initializes the node.

        Args:
            args: positional arguments for a ros Node
            default_callback_group: optional callback group to use as default
                for all subsequently created entities, such as subscriptions
                and clients.
            kwargs: keyword arguments for a ros Node

        See rclpy.node.Node documentation for further reference on available arguments.
        """
        if default_callback_group is None:
            default_callback_group = NonReentrantCallbackGroup()
        self._default_callback_group_override = default_callback_group
        self._destruction_requested = False
        super().__init__(*args, **kwargs)
        self._logger: MemoizingRcutilsLogger = MemoizingRcutilsLogger(self._logger)

    @property
    def default_callback_group(self) -> CallbackGroup:
        """Get the default callback group."""
        # NOTE(hidmic): this overrides the hardcoded default group in rclpy.node.Node implementation
        return self._default_callback_group_override

    def create_rate(
        self,
        frequency: float,
        clock: Optional[Clock] = None,
    ) -> SteadyRate:
        """Create a Rate object.

        :param frequency: The frequency the Rate runs at (Hz).
        :param clock: The clock the Rate gets time from.
        """
        if clock is None:
            clock = self.get_clock()
        return SteadyRate(frequency, clock, context=self._context)

    def destroy_rate(self, rate: SteadyRate) -> bool:
        """Destroy a Rate object created by the node.

        :return: ``True`` if successful, ``False`` otherwise.
        """
        rate.destroy()
        return True

    @property
    def waitables(self) -> Iterable[Waitable]:
        """Get patched node waitables.

        Workaround for https://github.com/ros2/rclpy/issues/1284.
        """
        for waitable in super().waitables:
            if not getattr(waitable, "__patched__", False):
                waitable.add_to_wait_set = suppressed(InvalidHandle, waitable.add_to_wait_set)
                waitable.is_ready = suppressed(IndexError, waitable.is_ready)
                waitable.__patched__ = True
            yield waitable

    @property
    def destruction_requested(self) -> bool:
        """Checks whether destruction was requested or not."""
        return self._destruction_requested

    def destroy_node(self) -> None:
        """Overrides node destruction API."""
        self._destruction_requested = True
        super().destroy_node()
