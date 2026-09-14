"""Cooperative cancellation shared by API workers and LangGraph nodes."""
from concurrent.futures import CancelledError
from contextvars import ContextVar
from functools import wraps
from threading import Event

cancel_event: ContextVar[Event | None] = ContextVar("aquabio_cancel_event", default=None)


def check_cancelled() -> None:
    event = cancel_event.get()
    if event is not None and event.is_set():
        raise CancelledError("Task cancelled")


def cancellable(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        check_cancelled()
        return function(*args, **kwargs)
    return wrapped
