from typing import Callable, Sequence
from xdsl.ir.core import *
from xdsl.irdl import *
from .. import core
from . import context
from . import predicate

ANY = context.ANY
NoMatchException = context.NoMatchException


def MatchOp(
    op_or_name: str | core.MlirOpBase,
    operands: Sequence[SSAValue | core.MlirOpBase] | None = None,
    results: Sequence[SSAValue | core.MlirOpBase] | None = None,
    preds: (
        Sequence[Callable[[core.MlirOpBase], bool] | predicate.Predicate] | None
    ) = None,
):
  """Finds an operation that matches the given criteria.

  This function searches for an operation that matches the specified operation
  name or type, operands, results, and predicates. It iterates through a set of
  candidate operations, filtering them based on the provided criteria.

  Args:
    op_or_name: The name of the operation or an core.MlirOpBase instance.
    operands: A sequence of expected operands. Use `ANY` to match any operand.
    results: A sequence of expected results. Use `ANY` to match any result.
    preds: A sequence of predicates to apply to the operation.

  Returns:
    The first operation that matches all the criteria.

  Raises:
    NoMatchException: If no matching operation is found.
  """
  if preds is None:
    preds = []

  candidates = set()
  if isinstance(op_or_name, IRDLOperation):
    candidates.add(op_or_name)
    name = op_or_name.name
  else:
    name = op_or_name

  if operands is not None:
    operands = [SSAValue.get(v) for v in operands]
  if results is not None:
    results = [SSAValue.get(v) for v in results]

  for x in results or []:
    if isinstance(x, SSAValue):
      candidates.add(x.owner)
  for x in operands or []:
    if isinstance(x, SSAValue):
      for use in x.uses:
        candidates.add(use.operation)
  for op in candidates:
    if not isinstance(op, IRDLOperation):
      continue
    if op.name != name:
      continue
    matched = True
    if matched and operands is not None:
      if len(operands) < len(op.operands):
        continue
      for expect, target in zip(operands, op.operands):
        if expect == ANY:
          continue
        if expect != target:
          matched = False
          break
    if matched and results is not None:
      if len(results) < len(op.results):
        continue
      for expect, target in zip(results, op.results):
        if expect == ANY:
          continue
        if expect != target:
          matched = False
          break
    if not matched:
      continue

    for pred in preds:
      if not pred(op):
        matched = False
        break
    if not matched:
      continue
    return op
  raise NoMatchException()
