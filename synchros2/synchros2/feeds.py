# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute.  All rights reserved.

from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
    cast,
    overload,
)

import tf2_ros
from rclpy.node import Node

import synchros2.scope as scope
from synchros2.filters import (
    Adapter,
    ApproximateTimeSynchronizer,
    ExactTimeSynchronizer,
    Filter,
    TimeSynchronizerBase,
    TransformFilter,
    Tunnel,
)
from synchros2.futures import FutureLike
from synchros2.utilities import Tape

MessageT = TypeVar("MessageT")


class MessageFeed(Generic[MessageT]):
    """An ergonomic wrapper for generic message filters."""

    def __init__(
        self,
        link: Filter,
        *,
        history_length: Optional[int] = None,
        node: Optional[Node] = None,
        trace: bool = False,
    ) -> None:
        """Initializes the message feed.

        Args:
            link: Wrapped message filter, connecting this message feed with its source.
            history_length: optional historic data size, defaults to 1
            node: optional node for lifetime control, defaults to the current process node.
            trace: Whether to log when messages are received by the message feed
        """
        if node is None:
            node = scope.ensure_node()
        if history_length is None:
            history_length = 1
        self._link = link
        self._tape: Tape[MessageT] = Tape(history_length)
        if trace:
            self._logger = node.get_logger()
            self._link.registerCallback(self._callback_trace)
        self._link.registerCallback(self._callback)
        node.context.on_shutdown(self.close)

    def _callback_trace(self, *msgs: Any) -> None:
        self._logger.debug(f"{self.__class__.__qualname__} received {len(msgs)} messages")

    def _callback(self, *msgs: Any) -> None:
        # NOTE: use variadic generics when they become available
        self._tape.write(cast(MessageT, msgs if len(msgs) > 1 else msgs[0]))

    @property
    def link(self) -> Filter:
        """Gets the underlying message connection."""
        return self._link

    @property
    def history(self) -> List[MessageT]:
        """Gets the entire history of messages received so far."""
        return self._tape.content(greedy=True)

    @property
    def latest(self) -> Optional[MessageT]:
        """Gets the latest message received, if any."""
        return self._tape.head

    @property
    def latest_update(self) -> FutureLike[MessageT]:
        """Gets the future to the latest message, which may not have been received yet."""
        return self._tape.latest_write

    @property
    def update(self) -> FutureLike[MessageT]:
        """Gets the future to the next message yet to be received."""
        return self._tape.future_write

    def matching_update(
        self,
        matching_predicate: Callable[[MessageT], bool],
    ) -> FutureLike[MessageT]:
        """Gets a future to the next matching message yet to be received.

        Args:
            matching_predicate: a boolean predicate to match incoming messages.

        Returns:
            a future.
        """
        return self._tape.future_matching_write(matching_predicate)

    def recall(self, callback: Callable[[MessageT], None]) -> Tunnel:
        """Adds a callback for message recalling.

        Returns:
            the underlying connection, which can be closed to stop future callbacks.
        """
        tunnel = Tunnel(self.link)
        tunnel.registerCallback(callback)
        return tunnel

    @overload
    def stream(
        self,
        *,
        forward_only: bool = False,
        expunge: bool = False,
        buffer_size: Optional[int] = None,
        duration_sec: Optional[float] = None,
        timeout_sec: Optional[float] = None,
    ) -> Generator[MessageT, None, None]:
        """Overload for plain streaming."""

    @overload
    def stream(
        self,
        *,
        greedy: Literal[True],
        forward_only: bool = False,
        expunge: bool = False,
        buffer_size: Optional[int] = None,
        duration_sec: Optional[float] = None,
        timeout_sec: Optional[float] = None,
    ) -> Generator[List[MessageT], None, None]:
        """Overload for greedy, batched streaming."""

    def stream(
        self,
        *,
        greedy: bool = False,
        forward_only: bool = False,
        expunge: bool = False,
        buffer_size: Optional[int] = None,
        duration_sec: Optional[float] = None,
        timeout_sec: Optional[float] = None,
    ) -> Generator[Union[MessageT, List[MessageT]], None, None]:
        """Iterates over messages as they come.

        Iteration stops when the given timeout expires or when the associated context
        is shutdown. Note that iterating over the message stream is a blocking operation.

        Args:
            greedy: if true, greedily batch messages as it becomes available.
            forward_only: whether to ignore previosuly received messages.
            expunge: if true, wipe out the message history after reading
            if it applies (i.e. non-forward only streams).
            buffer_size: optional maximum size for the incoming messages buffer.
            If none is provided, the buffer will be grow unbounded.
            duration_sec: optional duration, in seconds, to collect messages from the feed.
            timeout_sec: optional timeout, in seconds, for a new message to be received.

        Returns:
            a lazy iterator over messages, one message at a time or in batches if greedy.

        Raises:
            TimeoutError: if streaming times out waiting for a new message.
        """
        if greedy:
            # use boolean literals to help mypy
            return self._tape.content(
                follow=True,
                greedy=True,
                expunge=expunge,
                forward_only=forward_only,
                buffer_size=buffer_size,
                duration_sec=duration_sec,
                timeout_sec=timeout_sec,
            )
        return self._tape.content(
            follow=True,
            expunge=expunge,
            forward_only=forward_only,
            buffer_size=buffer_size,
            duration_sec=duration_sec,
            timeout_sec=timeout_sec,
        )

    def start(self) -> None:
        """Start the message feed."""
        self._link.start()

    def flush(self) -> None:
        """Flush the message feed."""
        self._tape.flush()

    def stop(self) -> None:
        """Stop the message feed."""
        self._link.stop()
        self._tape.close()

    close = stop


class AdaptedMessageFeed(MessageFeed[MessageT]):
    """A message feed decorator to simplify adapter patterns."""

    def __init__(
        self,
        feed: MessageFeed,
        fn: Callable[..., MessageT],
        *,
        autostart: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the message feed.

        Args:
            feed: the upstream (ie. decorated) message feed.
            fn: message adapting callable.
            kwargs: all other keyword arguments are forwarded
            for `MessageFeed` initialization.
            autostart: whether to start feeding messages immediately or not.
            kwargs: key word arguments to pass to the message feed
        """
        super().__init__(Adapter(feed.link, fn, autostart=autostart), **kwargs)
        self._feed = feed

    @property
    def feed(self) -> MessageFeed:
        """Gets the upstream message feed."""
        return self._feed

    def stop(self) -> None:
        """Stop this message feed and the upstream one as well."""
        self._feed.stop()
        super().stop()


class FramedMessageFeed(MessageFeed[MessageT]):
    """A message feed decorator, incorporating transforms using a `TransformFilter` instance."""

    def __init__(
        self,
        feed: MessageFeed,
        target_frame_id: str,
        *,
        tolerance_sec: float = 1.0,
        tf_buffer: Optional[tf2_ros.Buffer] = None,
        history_length: Optional[int] = None,
        node: Optional[Node] = None,
        autostart: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the message feed.

        Args:
            feed: the upstream message feed.
            target_frame_id: the target frame ID for transforms.
            tf_buffer: optional buffer of transforms to look up. If none is provided
            a transforms' buffer and a listener will be instantiated.
            tolerance_sec: optional tolerance, in seconds, to wait for late transforms.
            history_length: optional historic data size, defaults to 1.
            node: optional node for the underlying native subscription, defaults to
            the current process node.
            autostart: whether to start feeding messages immediately or not.
            kwargs: key word arguments to pass to the message feed
        """
        if node is None:
            node = scope.ensure_node()
        if tf_buffer is None:
            tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(tf_buffer, node)
        super().__init__(
            TransformFilter(
                feed.link,
                target_frame_id,
                tf_buffer,
                tolerance_sec,
                node.get_logger(),
                autostart=autostart,
            ),
            history_length=history_length,
            node=node,
            **kwargs,
        )
        self._feed = feed

    @property
    def feed(self) -> MessageFeed[MessageT]:
        """Gets the upstream message feed."""
        return self._feed

    def stop(self) -> None:
        """Stop this message feed and the upstream one as well."""
        self._feed.stop()
        super().stop()


class SynchronizedMessageFeedBase(MessageFeed):
    """A base class for message feeds' aggregators."""

    def __init__(
        self,
        time_synchronizer_filter: TimeSynchronizerBase,
        *feeds: MessageFeed,
        history_length: Optional[int] = None,
        node: Optional[Node] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the message feed.

        Args:
            time_synchronizer_filter: This should be one of `synchros2.filters.ApproximateTimeSynchronizer` or
                `synchros2.filters.ExactTimeSynchronizer`
            feeds: upstream message feeds to be synchronized.
            history_length: optional historic data size, defaults to 1.
            node: optional node for the underlying native subscription, defaults to
            the current process node.
            kwargs: key word arguments to pass to the message feed
        """
        super().__init__(time_synchronizer_filter, history_length=history_length, node=node, **kwargs)
        self._feeds = feeds

    @property
    def feeds(self) -> Iterable[MessageFeed]:
        """Gets all aggregated message feeds."""
        return self._feeds

    def stop(self) -> None:
        """Stop this message feed and all upstream ones as well."""
        for feed in self._feeds:
            feed.stop()
        super().stop()


class SynchronizedMessageFeed(SynchronizedMessageFeedBase):
    """A message feeds' aggregator using a `message_filters.ApproximateTimeSynchronizer` instance."""

    def __init__(
        self,
        *feeds: MessageFeed,
        queue_size: int = 10,
        delay: float = 0.2,
        allow_headerless: bool = False,
        history_length: Optional[int] = None,
        node: Optional[Node] = None,
        autostart: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the message feed.

        Args:
            feeds: upstream message feeds to be synchronized.
            queue_size: the message queue size for synchronization.
            delay: the maximum delay, in seconds, between messages for synchronization to succeed.
            allow_headerless: whether it's OK for there to be no header in the messages (in which
            case, the ROS time of arrival will be used).
            history_length: optional historic data size, defaults to 1.
            node: optional node for the underlying native subscription, defaults to
            the current process node.
            autostart: whether to start feeding messages immediately or not.
            kwargs: key word arguments to pass to the message feed
        """
        super().__init__(
            ApproximateTimeSynchronizer(
                [f.link for f in feeds],
                queue_size,
                delay,
                allow_headerless=allow_headerless,
                autostart=autostart,
            ),
            *feeds,
            history_length=history_length,
            node=node,
            **kwargs,
        )


class ExactSynchronizedMessageFeed(SynchronizedMessageFeedBase):
    """A message feeds' aggregator using a `message_filters.TimeSynchronizer` instance."""

    def __init__(
        self,
        *feeds: MessageFeed,
        queue_size: int = 10,
        history_length: Optional[int] = None,
        node: Optional[Node] = None,
        autostart: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the message feed.

        Args:
            feeds: upstream message feeds to be synchronized.
            queue_size: the message queue size for synchronization.
            history_length: optional historic data size, defaults to 1.
            node: optional node for the underlying native subscription, defaults to
            the current process node.
            autostart: whether to start feeding messages immediately or not.
            kwargs: key word arguments to pass to the message feed
        """
        super().__init__(
            ExactTimeSynchronizer(
                [f.link for f in feeds],
                queue_size,
                autostart=autostart,
            ),
            *feeds,
            history_length=history_length,
            node=node,
            **kwargs,
        )
        self._feeds = feeds
