from typing import Callable, overload
from . import context

ANY = context.ANY
NoMatchException = context.NoMatchException


@overload
def MatchPred(pred: Callable[[], bool] | bool):
  ...


def MatchPred(value, pred: Callable[[], bool] | bool = None):

  if pred is None:
    pred = value
    if callable(pred):
      pred = pred()
  else:
    pred = pred(value)

  if not pred:
    raise NoMatchException("MatchCheck assertion failed")
