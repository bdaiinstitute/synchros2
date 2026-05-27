# Copyright (c) 2023 Robotics and AI Institute LLC dba RAI Institute.  All rights reserved.
import threading
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, FIRST_EXCEPTION
from typing import (
    Any,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    NamedTuple,
    Optional,
    Protocol,
    Set,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)

from rclpy.clock import Clock
from rclpy.context import Context
from rclpy.duration import Duration
from rclpy.utilities import get_default_context

from synchros2.clock import wait_for

T = TypeVar("T", covariant=True)


@runtime_checkable
class FutureLike(Awaitable[T], Protocol[T]):
    """A future-like awaitable object.

    Matches `rclpy.task.Future` and `concurrent.futures.Future` protocols.
    """

    def result(self) -> T:
        """Get future result (may block)."""
        ...

    def exception(self) -> Optional[Exception]:
        """Get future exception, if any."""
        ...

    def done(self) -> bool:
        """Check if future is ready."""
        ...

    def add_done_callback(self, func: Callable[["FutureLike[T]"], None]) -> None:
        """Add a callback to be scheduled as soon as the future is ready."""
        ...

    def cancel(self) -> None:
        """Cancel future."""
        ...

    def cancelled(self) -> bool:
        """Check if future was cancelled."""
        ...


@runtime_checkable
class FutureConvertible(Awaitable[T], Protocol[T]):
    """An awaitable that is convertible to a future-like object."""

    def as_future(self) -> FutureLike[T]:
        """Get future-like view."""
        ...


AnyFuture = Union[FutureLike, FutureConvertible]


def as_proper_future(instance: AnyFuture) -> FutureLike:
    """Return `instance` as a proper future-like object."""
    if isinstance(instance, FutureConvertible):
        return instance.as_future()
    return instance


class WaitResult(NamedTuple):
    """Result of waiting for multiple futures.

    A named tuple with 'done' and 'not_done' sets of futures.
    """

    ok: bool
    done: Set[FutureLike]
    not_done: Set[FutureLike]

    def __bool__(self) -> bool:
        """Equivalent to result.ok."""
        return self.ok


@overload
def wait_for_future(
    future: AnyFuture,
    timeout_sec: Optional[float] = None,
    *,
    clock: Optional[Clock] = None,
    context: Optional[Context] = None,
) -> WaitResult:
    ...


@overload
def wait_for_future(
    future: Iterable[AnyFuture],
    timeout_sec: Optional[float] = None,
    *,
    return_when: str = ALL_COMPLETED,
    clock: Optional[Clock] = None,
    context: Optional[Context] = None,
) -> WaitResult:
    ...


def wait_for_future(
    future: Union[AnyFuture, Iterable[AnyFuture]],
    timeout_sec: Optional[float] = None,
    *,
    clock: Optional[Clock] = None,
    context: Optional[Context] = None,
    return_when: str = ALL_COMPLETED,
) -> WaitResult:
    """Block while waiting for future(s) to become done.

    Args:
        future: A single future or an iterable of futures to wait on
        timeout_sec: An optional timeout for how long to wait
        clock: An optional clock to use for timeout waits,
            defaults to the clock of the current scope if any, otherwise the system clock
        context: Current context (will use the default if none is given)
        return_when: One of FIRST_COMPLETED, FIRST_EXCEPTION, or ALL_COMPLETED.
            Only applies when waiting for multiple futures. Defaults to ALL_COMPLETED.

    Returns:
        A result object indicating which futures are done and which are not,
        and whether the wait was successful (i.e. not timed out).

    Examples:
        Single future:
            >>> result = wait_for_future(my_future, timeout_sec=5.0)
            >>> if result:
            ...     value = my_future.result()

        Multiple futures:
            >>> result = wait_for_future([f1, f2, f3], return_when=FIRST_COMPLETED)
            >>> for future in result.done:
            ...     print(future.result())
    """
    if return_when not in {FIRST_COMPLETED, FIRST_EXCEPTION, ALL_COMPLETED}:
        raise ValueError(f"Invalid return_when value: {return_when}")

    if context is None:
        context = get_default_context()

    if clock is None:
        import synchros2.scope

        clock = synchros2.scope.clock()

    done_futures: Set[FutureLike] = set()
    if not isinstance(future, (FutureConvertible, FutureLike)):
        pending_futures = {as_proper_future(f) for f in future}
    else:
        pending_futures = {as_proper_future(future)}

    if not pending_futures:
        return WaitResult(ok=True, done=set(), not_done=set())

    lock = threading.Lock()
    event = threading.Event()

    def _done_callback(future: FutureLike) -> None:
        with lock:
            if future in pending_futures:
                pending_futures.remove(future)
                done_futures.add(future)

        should_return = False
        if return_when == FIRST_COMPLETED:
            should_return = True
        elif return_when == FIRST_EXCEPTION:
            exception_occurred = future.exception() is not None
            should_return = exception_occurred or not pending_futures
        elif return_when == ALL_COMPLETED:
            should_return = not pending_futures

        if should_return:
            event.set()

    context.on_shutdown(event.set)
    for future in list(pending_futures):
        future.add_done_callback(_done_callback)
        if future.cancelled():
            _done_callback(future)

    if not event.is_set():
        wait_for(event, clock=clock, timeout_sec=timeout_sec)

    with lock:
        return WaitResult(ok=event.is_set(), done=done_futures.copy(), not_done=pending_futures.copy())


@overload
def unwrap_future(
    future: AnyFuture,
    timeout_sec: Optional[float] = None,
    *,
    clock: Optional[Clock] = None,
    context: Optional[Context] = None,
) -> Any:
    ...


@overload
def unwrap_future(
    future: Iterable[AnyFuture],
    timeout_sec: Optional[float] = None,
    *,
    clock: Optional[Clock] = None,
    context: Optional[Context] = None,
    strict: bool = False,
) -> Iterator[Any]:
    ...


def unwrap_future(
    future: Union[AnyFuture, Iterable[AnyFuture]],
    timeout_sec: Optional[float] = None,
    *,
    clock: Optional[Clock] = None,
    context: Optional[Context] = None,
    strict: bool = False,
) -> Union[Any, Iterator[Any]]:
    """Fetch future result(s) when done.

    For a single future, blocks until the future is done and returns its result.
    For multiple futures, returns a generator that yields results as futures complete
    (like concurrent.futures.as_completed).

    Note: This function may block and may raise if a future raises or it times out
    waiting. See wait_for_future() documentation for further reference on arguments.

    Args:
        future: A single future or an iterable of futures
        timeout_sec: An optional timeout for how long to wait
        clock: An optional clock to use for timeout waits
        context: Current context (will use the default if none is given)
        strict: If True, yield results in order regardless of completion order.
            If False (default), yield results as they complete.
            Irrelevant when a single future is provided.

    Returns:
        the result(s) of the future(s) when they are done.

    Raises:
        ValueError: If timeout occurs before future(s) complete

    Examples:
        Single future:
            >>> result = unwrap_future(my_future, timeout_sec=5.0)

        Multiple futures (non-strict, as completed):
            >>> for result in unwrap_future([f1, f2, f3], timeout_sec=10.0):
            ...     process(result)

        Multiple futures (strict, in order):
            >>> for result in unwrap_future([f1, f2, f3], timeout_sec=10.0, strict=True):
            ...     process(result)
    """
    if context is None:
        context = get_default_context()

    if clock is None:
        import synchros2.scope

        clock = synchros2.scope.clock()

    if isinstance(future, (FutureConvertible, FutureLike)):
        proper_future = as_proper_future(future)
        if not wait_for_future(proper_future, timeout_sec, clock=clock, context=context):
            raise ValueError("cannot unwrap future that is not done")
        return proper_future.result()

    def _result_generator() -> Any:
        nonlocal future
        future = cast(Iterable[AnyFuture], future)
        pending_futures = [as_proper_future(f) for f in future]
        if not pending_futures:
            return

        deadline = None
        if timeout_sec is not None:
            assert clock is not None
            deadline = clock.now() + Duration(seconds=timeout_sec)

        if strict:
            for future in pending_futures:
                remaining_timeout_sec = None
                if deadline is not None:
                    assert clock is not None
                    remaining_duration = deadline - clock.now()
                    if remaining_duration.nanoseconds <= 0:
                        raise ValueError("timeout waiting for futures")
                    remaining_timeout_sec = remaining_duration.nanoseconds / 1e9

                if not wait_for_future(future, timeout_sec=remaining_timeout_sec, clock=clock, context=context):
                    raise ValueError("timeout waiting for futures")
                yield future.result()
            return

        while pending_futures:
            remaining_timeout_sec = None
            if deadline is not None:
                assert clock is not None
                remaining_duration = deadline - clock.now()
                if remaining_duration.nanoseconds <= 0:
                    raise ValueError("timeout waiting for futures")
                remaining_timeout_sec = remaining_duration.nanoseconds / 1e9

            result = wait_for_future(
                pending_futures,
                timeout_sec=remaining_timeout_sec,
                clock=clock,
                context=context,
                return_when=FIRST_COMPLETED,
            )

            if not result:
                raise ValueError("timeout waiting for futures")

            for future in result.done:
                if future in pending_futures:
                    pending_futures.remove(future)
                    yield future.result()

    return _result_generator()


wait_and_return_result = unwrap_future
"""Fetch future result when it is done.

Alias for unwrap_future()."""
