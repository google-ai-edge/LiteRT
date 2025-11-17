# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Match operation functions."""

from typing import Callable, Sequence, Type, TypeVar, overload
from xdsl import irdl
from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.match import context
from litert.python.tools.model_utils.match import predicate

ANY = context.ANY
NoMatchError = context.NoMatchError
SSAValue = irdl.SSAValue

OpT = TypeVar('OpT', bound=core.MlirOpBase)


@overload
def match_op(
    op_or_name: Type[OpT],  # Input is the class itself, like tfl.AddOp
    operands: Sequence[SSAValue | core.MlirOpBase] | None = None,
    results: Sequence[SSAValue | core.MlirOpBase] | None = None,
    preds: (
        Sequence[Callable[[core.MlirOpBase], bool] | predicate.Predicate] | None
    ) = None,
) -> OpT:  # Return type is an instance of that specific class
  ...


@overload
def match_op(
    op_or_name: str,  # Input is the operation name string, like "tfl.add"
    operands: Sequence[SSAValue | core.MlirOpBase] | None = None,
    results: Sequence[SSAValue | core.MlirOpBase] | None = None,
    preds: (
        Sequence[Callable[[core.MlirOpBase], bool] | predicate.Predicate] | None
    ) = None,
) -> core.MlirOpBase:  # Return type is the base MlirOpBase class
  ...


def match_op(
    op_or_name: str | Type[OpT],
    operands: Sequence[SSAValue | core.MlirOpBase] | None = None,
    results: Sequence[SSAValue | core.MlirOpBase] | None = None,
    preds: (
        Sequence[Callable[[core.MlirOpBase], bool] | predicate.Predicate] | None
    ) = None,
) -> core.MlirOpBase | OpT:
  """Finds an operation that matches the given criteria.

  This function searches for an operation that matches the specified operation
  name or type, operands, results, and predicates. It iterates through a set of
  candidate operations, filtering them based on the provided criteria.

  Args:
    op_or_name: The name (str) or the class (Type[MlirOpBase]) of the operation.
    operands: A sequence of expected operands. Use `ANY` to match any operand.
    results: A sequence of expected results. Use `ANY` to match any result.
    preds: A sequence of predicates to apply to the operation.

  Returns:
    The first operation that matches all the criteria. The specific return type
    depends on the type of `op_or_name` (see overloads).

  Raises:
    NoMatchError: If no matching operation is found.
  """
  if preds is None:
    preds = []

  candidates = set()
  if hasattr(op_or_name, 'name'):
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
    if not isinstance(op, irdl.IRDLOperation):
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
  raise NoMatchError()
