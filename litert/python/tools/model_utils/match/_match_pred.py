"""Match predicate functions."""

from typing import Any, Callable, TypeVar, overload
from . import context


ANY = context.ANY


@overload
def match_pred(pred: Callable[[], bool] | bool):
  """Raises NoMatchException if the predicate is False or returns False."""
  ...


T = TypeVar('T')


@overload
def match_pred(value: T, pred: Callable[[T], bool]):
  """Matches a predicate against a value."""
  ...


def match_pred(value, pred: Callable[[Any], bool] | bool = None):
  if pred is None:
    pred = value
    if callable(pred):
      pred = pred()
  else:
    pred = pred(value)

  if not pred:
    raise context.NoMatchError()
