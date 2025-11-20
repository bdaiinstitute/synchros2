# Copyright (c) 2023 Robotics and AI Institute LLC dba RAI Institute.  All rights reserved.

import collections
import contextlib
import functools
import inspect
import queue
import threading
import time
import warnings
import weakref
from collections.abc import Mapping, MutableSet
from typing import Any, Callable, Generator, Generic, List, Literal, Optional, Tuple, TypeVar, Union, overload

import rclpy.clock
import rclpy.duration
import rclpy.time
from rclpy.task import Future

from synchros2.futures import FutureLike

T = TypeVar("T")
U = TypeVar("U")


def namespace_with(*args: Optional[str]) -> str:
    """Puts together a ROS 2 like namespace from its constitutive parts."""
    sanitized_args = list(filter(None, args))
    if not sanitized_args:
        raise ValueError("nothing to namespace")
    sanitized_args = [arg.rstrip("/") for arg in sanitized_args]
    namespace = sanitized_args[0]
    for arg in sanitized_args[1:]:
        if not arg.startswith("/"):
            namespace += "/" + arg
        else:
            namespace = arg
    return namespace


def either_or(obj: Any, name: str, default: Any) -> Any:
    """Gets either an object attribute's value or a default value.

    Unlike `getattr`, callable attributes are applied as getters on `obj`.
    """
    if not hasattr(obj, name):
        return default
    value_or_getter = getattr(obj, name)
    if callable(value_or_getter):
        return value_or_getter(obj)
    return value_or_getter


def fqn(obj: Any) -> Optional[str]:
    """Computes the fully qualified name of a given object, if any."""
    if not hasattr(obj, "__qualname__"):
        return None
    name = obj.__qualname__
    if not hasattr(obj, "__module__"):
        return name
    module = obj.__module__
    return f"{module}.{name}"


def bind_to_thread(callable_: Callable, thread: threading.Thread) -> Callable:
    """Binds a callable to a thread, so it can only be invoked from that thread."""

    @functools.wraps(callable_)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        if threading.current_thread() is not thread:
            raise RuntimeError(f"{fqn(callable_)}() is bound to a different thread")
        return callable_(*args, **kwargs)

    return _wrapper


class Tape(Generic[T]):
    """A thread-safe data tape that can be written and iterated safely."""

    class Stream(Generic[U]):
        """A synchronized data stream."""

        def __init__(self, max_size: Optional[int] = None, label: Optional[str] = None) -> None:
            """Initializes the stream.

            Args:
                max_size: optional maximum stream size. Must be a positive number.
                label: optional label for the stream (useful in debug and error messages).
            """
            assert max_size is None or max_size > 0
            self._queue: queue.Queue = queue.Queue(max_size or 0)
            self._label = label

        @property
        def label(self) -> Optional[str]:
            """Get stream label."""
            return self._label

        def write(self, data: U) -> bool:
            """Write data to the stream.

            Returns:
                True if the write operation succeeded, False if the
                stream has grown to its specified maximum size and
                the write operation cannot be carried out.
            """
            try:
                self._queue.put_nowait(data)
            except queue.Full:
                return False
            return True

        def try_read(self) -> Optional[U]:
            """Try to read data from the stream.

            Returns:
                data if the read is successful, and ``None``
                if there is nothing to be read or the stream
                is interrupted.
            """
            try:
                data = self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                return None
            return data

        def read(self, timeout_sec: Optional[float] = None) -> Optional[U]:
            """Read data from the stream.

            Args:
                timeout_sec: optional read timeout, in seconds.

            Returns:
                data if the read is successful, and ``None``
                if the stream is interrupted.

            Raises:
                TImeoutError if the read times out.
            """
            try:
                data = self._queue.get(timeout=timeout_sec)
            except queue.Empty as e:
                raise TimeoutError() from e
            self._queue.task_done()
            return data

        def interrupt(self) -> None:
            """Interrupt the stream and wake up the reader."""
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(None)

        @property
        def consumed(self) -> bool:
            """Check if all stream data has been consumed."""
            return self._queue.empty()

    def __init__(self, max_length: Optional[int] = None) -> None:
        """Initializes the data tape.

        Args:
            max_length: optional maximum tape length.
        """
        self._lock = threading.RLock()
        self._streams: MutableSet[Tape.Stream[T]] = weakref.WeakSet()
        self._content: Optional[collections.deque] = None
        if max_length is None or max_length > 0:
            self._content = collections.deque(maxlen=max_length)
        self._future_write: Optional[Future] = None
        self._future_matching_writes: List[Tuple[Callable[[T], bool], Future]] = []
        self._write_callbacks: List[Callable[[T], None]] = []
        self._closed = False

    def flush(self) -> None:
        """Flushes the data tape content, if any."""
        with self._lock:
            if self._content is not None:
                self._content.clear()

    @property
    def latest_write(self) -> FutureLike[T]:
        """Gets the future to the latest data written or to be written."""
        with self._lock:
            if self._content is None:
                raise RuntimeError("zero-length tape only writes forward")
            if len(self._content) > 0:
                data = self._content[-1]
                latest_write = Future()
                latest_write.set_result(data)
                return latest_write
            if self._future_write is None:
                self._future_write = Future()
                if self._closed:
                    self._future_write.cancel()
            return self._future_write

    @property
    def future_write(self) -> FutureLike[T]:
        """Gets the future to the next data yet to be written."""
        with self._lock:
            if self._future_write is None:
                self._future_write = Future()
                if self._closed:
                    self._future_write.cancel()
            return self._future_write

    def future_matching_write(self, matching_predicate: Callable[[T], bool]) -> FutureLike[T]:
        """Gets a future to the next matching data yet to be written.

        Args:
            matching_predicate: a boolean predicate to match written data.

        Returns:
            a future.
        """
        with self._lock:
            future_write = Future()
            if not self._closed:
                self._future_matching_writes.append(
                    (
                        matching_predicate,
                        future_write,
                    ),
                )
            else:
                future_write.cancel()
            return future_write

    def add_write_callback(self, callback: Callable[[T], None], forward_only: bool = False) -> None:
        """Adds a callback to be invoked on each write.

        Args:
            callback: a callable taking the written data as its only argument.
            forward_only: if true, ignore existing content and call back on future writes.
        """
        with self._lock:
            if not forward_only and self._content is not None:
                for data in self._content:
                    callback(data)
            self._write_callbacks.append(callback)

    def write(self, data: T) -> bool:
        """Write the data tape."""
        with self._lock:
            if self._closed:
                return False
            for callback in self._write_callbacks:
                callback(data)
            for stream in self._streams:
                if not stream.write(data):
                    message = "Stream is filled up, dropping message"
                    if stream.label:
                        message = f"{stream.label}: {message}"
                    warnings.warn(message, RuntimeWarning, stacklevel=1)
            if self._content is not None:
                self._content.append(data)
            if self._future_write is not None:
                self._future_write.set_result(data)
                self._future_write = None
            for item in list(self._future_matching_writes):
                matching_predicate, future_write = item
                if matching_predicate(data):
                    future_write.set_result(data)
                    self._future_matching_writes.remove(item)
            return True

    @property
    def head(self) -> Optional[T]:
        """Returns the data tape head, if any."""
        with self._lock:
            if self._content is None:
                return None
            if len(self._content) == 0:
                return None
            return self._content[-1]

    @overload
    def content(
        self,
        *,
        follow: bool = ...,
        forward_only: bool = ...,
        expunge: bool = ...,
        buffer_size: Optional[int] = ...,
        duration_sec: Optional[float] = ...,
        timeout_sec: Optional[float] = ...,
        label: Optional[str] = ...,
    ) -> Generator[T, None, None]:
        """Overload for non-greedy iteration."""

    @overload
    def content(
        self,
        *,
        greedy: Literal[True],
        follow: Literal[True],
        forward_only: bool = ...,
        expunge: bool = ...,
        buffer_size: Optional[int] = ...,
        duration_sec: Optional[float] = ...,
        timeout_sec: Optional[float] = ...,
        label: Optional[str] = ...,
    ) -> Generator[List[T], None, None]:
        """Overload for greedy batched iteration."""

    @overload
    def content(
        self,
        *,
        greedy: Literal[True],
        expunge: bool = ...,
        buffer_size: Optional[int] = ...,
        duration_sec: Optional[float] = ...,
        timeout_sec: Optional[float] = ...,
        label: Optional[str] = ...,
    ) -> List[T]:
        """Overload for greedy full reads."""

    def content(
        self,
        *,
        greedy: bool = False,
        follow: bool = False,
        forward_only: bool = False,
        expunge: bool = False,
        buffer_size: Optional[int] = None,
        duration_sec: Optional[float] = None,
        timeout_sec: Optional[float] = None,
        label: Optional[str] = None,
    ) -> Union[Generator[Union[T, List[T]], None, None], List[T]]:
        """Iterate over the data tape.

        When following the data tape, iteration stops when the given timeout
        expires and when the data tape is closed.

        Args:
            greedy: if true, greedily batch content as it becomes available.
            follow: whether to follow the data tape as it gets written or not.
            forward_only: if true, ignore existing content and only look ahead
            when following the data tape.
            expunge: if true, wipe out existing content in the data tape after
            reading if it applies (i.e. non-forward only iterations).
            buffer_size: optional buffer size when following the data tape.
            If none is provided, the buffer will grow as necessary.
            duration_sec: optional duration, in seconds, to gather content from the data tape.
            timeout_sec: optional timeout, in seconds, when following the data tape.
            label: optional label to qualify logs and warnings.

        Returns:
            a lazy iterator over the data tape, one item at a time or in batches if greedy.

        Raises:
            TimeoutError: if iteration times out waiting for new data.
        """
        # Here we split the generator in two, so that setup code is executed eagerly.
        with self._lock:
            content: Optional[collections.deque] = None
            if not forward_only and self._content is not None:
                content = self._content.copy()
            if self._content is not None and expunge:
                self._content.clear()
            stream: Optional[Tape.Stream] = None
            if follow and not self._closed:
                stream = Tape.Stream(buffer_size, label)
                self._streams.add(stream)

        if content is not None and stream is None and greedy:
            return list(content)

        if timeout_sec is None:
            timeout_sec = duration_sec

        def _generator() -> Generator[Union[T, List[T]], None, None]:
            nonlocal content, stream
            try:
                if content is not None:
                    if greedy:
                        yield list(content)
                    else:
                        yield from content

                if stream is not None:
                    start_time = time.time()
                    while not self._closed:
                        try:
                            feedback = stream.read(timeout_sec)
                            if feedback is None:
                                continue
                            if greedy:
                                batch = [feedback]
                                while True:
                                    feedback = stream.try_read()
                                    if feedback is None:
                                        break
                                    batch.append(feedback)
                                yield batch
                            else:
                                yield feedback
                        except TimeoutError:
                            if duration_sec is None:
                                raise
                            elapsed_time = time.time() - start_time
                            if elapsed_time >= duration_sec:
                                break
                            continue

                    last_batch: List[T] = []
                    while not stream.consumed:
                        feedback = stream.try_read()
                        if feedback is not None:
                            last_batch.append(feedback)
                    if not greedy:
                        yield from last_batch
                    else:
                        yield last_batch
            finally:
                if stream is not None:
                    with self._lock:
                        self._streams.remove(stream)

        return _generator()

    def close(self) -> None:
        """Close the data tape.

        This will interrupt all following content iterators.
        """
        with self._lock:
            if not self._closed:
                self._closed = True
                for stream in self._streams:
                    stream.interrupt()
                if self._future_write is not None:
                    self._future_write.cancel()
                for _, future_write in self._future_matching_writes:
                    future_write.cancel()


def synchronized(
    func: Optional[Callable] = None,
    lock: Optional[threading.Lock] = None,
) -> Callable:
    """Wraps `func` to synchronize invocations, optionally taking a user defined `lock`.

    This function can be used as a decorator, like:

    .. code-block:: python

        @synchronized
        def my_function(...):
            ...

    or:

    .. code-block:: python

        @synchronized(lock=my_lock)
        def my_function(...):
            ...
    """
    if lock is None:
        lock = threading.Lock()
    assert lock is not None

    def _decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def __wrapper(*args: Any, **kwargs: Any) -> Any:
            with lock:  # type: ignore
                return func(*args, **kwargs)

        return __wrapper

    if func is None:
        return _decorator
    return _decorator(func)


def functional_decorator(base_decorator: Callable) -> Callable:
    """Wraps a decorating callable to be usable as a Python decorator for functions.

    As an example, consider the following decorator example:

    .. code-block:: python

        @functional_decorator
        def my_decorator(func, some_flag=None):
            ...

    This decorator can then be used like this:

    .. code-block:: python

        @my_decorator
        def my_function(*args):
            ...

    and also like this:

    .. code-block:: python

        @my_decorator(some_flag=True)
        def my_function(*args):
            ...
    """

    @functools.wraps(base_decorator)
    def _wrapper(func: Optional[Callable] = None, **kwargs: Any) -> Callable:
        def _bound_decorator(func: Callable) -> Callable:
            return base_decorator(func, **kwargs)

        if func is None:
            return _bound_decorator
        return _bound_decorator(func)

    return _wrapper


@functional_decorator
def throttle(
    func: Callable,
    min_period: rclpy.duration.Duration,
    time_source: Optional[rclpy.clock.Clock] = None,
    fill_value: Any = None,
) -> Callable:
    """Decorates a callable to throttle invocations.

    Args:
        func: callable to be decorated.
        min_period: minimum time between consecutive invocations.
        time_source: optional time source to measure time against.
        If none is provided, the system clock will be used.
        fill_value: optional value to return for throttled invocations.

    Returns:
        decorated callable.
    """
    safe_time_source = rclpy.clock.Clock() if time_source is None else time_source
    time_of_last_call: Optional[rclpy.time.Time] = None

    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal time_of_last_call
        return_value = fill_value
        current_time = safe_time_source.now()
        if time_of_last_call is None or current_time - time_of_last_call >= min_period:
            return_value = func(*args, **kwargs)
            time_of_last_call = current_time
        return return_value

    return _wrapper


@functional_decorator
def skip(func: Callable, num_times: int, fill_value: Any = None) -> Callable:
    """Decorates a callable to skip the first few invocations a prescribed number of times.

    Args:
        func: callable to be decorated.
        num_times: number of times to skip the invocation.
        fill_value: optional value to return for skipped invocations.

    Returns:
        decorated callable.
    """
    num_skipped_calls = 0

    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal num_skipped_calls
        if num_skipped_calls < num_times:
            num_skipped_calls += 1
            return fill_value
        return func(*args, **kwargs)

    return _wrapper


@functional_decorator
def cap(func: Callable, num_times: int, fill_value: Any = None) -> Callable:
    """Decorates a callable to cap invocations to a prescribed number of times.

    Args:
        func: callable to be decorated.
        num_times: maximum number of times the callable may be invoked.
        fill_value: optional value to return once invocations reach their cap.

    Returns:
        decorated callable.
    """
    num_calls = 0

    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal num_calls
        if num_calls < num_times:
            num_calls += 1
            return func(*args, **kwargs)
        return fill_value

    return _wrapper


def take_kwargs(func: Callable, kwargs: Mapping) -> Tuple[Mapping, Mapping]:
    """Take keyword arguments given a callable's signature.

    Args:
        func: callable to take keyword arguments for.
        kwargs: mapping to take keyword arguments from.

    Returns:
        a tuple of taken and dropped keyword arguments.
    """
    signature = inspect.signature(func)
    taken, dropped = {}, {}
    for name, value in kwargs.items():
        if name in signature.parameters:
            taken[name] = value
        else:
            dropped[name] = value
    return taken, dropped


def localized_error_message(user_message: Optional[str] = None) -> str:
    """Returns an error message with source location information."""
    this_frame = inspect.currentframe()
    assert this_frame is not None
    inner_frame = this_frame.f_back
    assert inner_frame is not None
    outer_frame = inner_frame.f_back
    assert outer_frame is not None
    traceback = inspect.getframeinfo(outer_frame)
    message = f"{traceback.filename}:{traceback.lineno}: "
    if traceback.code_context is not None:
        message += "".join(traceback.code_context).strip("\n ") + " failed"
    else:
        traceback = inspect.getframeinfo(inner_frame)
        message += f"{traceback.function}() failed"
    if user_message is not None:
        message += ": " + user_message
    return message


def ensure(value: Union[T, None]) -> T:
    """Ensures `value` is not None or fails trying."""
    if value is None:
        raise ValueError(localized_error_message())
    return value
