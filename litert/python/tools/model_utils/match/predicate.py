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
"""Predicate functions for common pattern matching."""

import abc
import functools
from typing import Any, Callable, TypeVar, overload
from xdsl import irdl
from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

SSAValue = irdl.SSAValue


class Predicate(abc.ABC):
  """Base class for predicates."""

  def __init__(self):
    self._predicate = None

  @abc.abstractmethod
  def __call__(self, x) -> bool:
    raise NotImplementedError()

  @classmethod
  def Wraps(
      cls,
      build_pred: Callable[..., Callable[[], bool]],
  ) -> type["Predicate"]:
    """Wraps a predicate function into a Predicate class."""
    T = TypeVar("T")

    # pylint: disable=invalid-name
    @functools.wraps(build_pred)
    def __init__(self: Predicate, *args: Any, **kwargs: Any):
      """Initializes the predicate."""
      self._predicate: Callable[[T], bool] = build_pred(*args, **kwargs)

    def __call__(self: Predicate, x: T) -> bool:
      return self._predicate(x)

    # pylint: enable=invalid-name

    pred_cls = type(
        getattr(build_pred, "__name__"),
        (Predicate,),
        {
            "__init__": __init__,
            "__call__": __call__,
            "__doc__": build_pred.__doc__,
        },
    )
    return pred_cls

  def __and__(self, other: "Predicate"):
    return PredicateAnd(self, other)

  def __or__(self, other: "Predicate"):
    return PredicateOr(self, other)


@Predicate.Wraps
def PredicateAnd(pred1: Predicate, pred2: Predicate) -> Predicate:
  """Checks if an operation satisfies both of the given predicates.

  Args:
    pred1: The first predicate.
    pred2: The second predicate.

  Returns:
    True if the operation satisfies both `pred1` and `pred2`, False otherwise.
  """

  def Pred(x):
    return pred1(x) and pred2(x)

  return Pred


@Predicate.Wraps
def PredicateOr(pred1: Predicate, pred2: Predicate) -> Predicate:
  """Checks if an operation satisfies at least one of the given predicates.

  Args:
    pred1: The first predicate.
    pred2: The second predicate.

  Returns:
    True if the operation satisfies either `pred1` or `pred2`, False otherwise.
  """

  def Pred(op: core.MlirOpBase):
    return pred1(op) or pred2(op)

  return Pred


@Predicate.Wraps
def OperandPred(
    i: int, pred: Callable[[SSAValue], bool]
) -> Callable[[core.MlirOpBase], bool]:
  """Checks if the i-th operand of an operation satisfies a given condition.

  Args:
    i: The index of the operand to check.
    pred: A predicate function that takes an SSAValue and returns a boolean.

  Returns:
    True if the i-th operand exists and satisfies the given predicate, False
    otherwise.
  """

  def Pred(op: core.MlirOpBase):
    if i >= len(op.operands):
      return False
    return pred(op.operands[i])

  return Pred


@Predicate.Wraps
def OperandShapePred(
    i: int, shape_pred: Callable[[list[int]], bool]
) -> Callable[[core.MlirOpBase], bool]:
  """Checks if the shape of the i-th operand of an operation satisfies a given condition.

  Args:
    i: The index of the operand to check.
    shape_pred: A predicate function that takes a shape (list of integers) and
      returns a boolean.

  Returns:
    True if the i-th operand is a RankedTensorType and its shape satisfies the
    given predicate, False otherwise.
  """

  def Pred(x: SSAValue):
    if not isinstance(x.type, mlir.RankedTensorType):
      return False
    return shape_pred(x.type.shape)

  return OperandPred(i, Pred)


@Predicate.Wraps
def OperandHasRank(i: int, rank: int) -> Callable[[core.MlirOpBase], bool]:
  """Checks if the i-th operand of an operation has the given rank.

  Args:
    i: The index of the operand to check.
    rank: The expected rank.

  Returns:
    True if the i-th operand is a RankedTensorType and has the given rank, False
    otherwise.
  """
  return OperandShapePred(i, lambda s: len(s) == rank)


@Predicate.Wraps
def OperandHasShape(
    i: int, shape: list[int]
) -> Callable[[core.MlirOpBase], bool]:
  """Checks if the i-th operand of an operation has the given shape.

  Args:
    i: The index of the operand to check.
    shape: The expected shape.

  Returns:
    True if the i-th operand is a RankedTensorType and has the given shape,
    False otherwise.
  """
  return OperandShapePred(i, lambda s: s == shape)


@overload
def ResultPred(
    pred: Callable[[SSAValue], bool],
) -> Callable[[core.MlirOpBase], bool]:
  ...


@Predicate.Wraps
def ResultPred(
    i: int = 0, pred: Callable[[SSAValue], bool] = None
) -> Callable[[core.MlirOpBase], bool]:
  """Checks if the i-th result of an operation satisfies a given condition.

  Args:
    i: The index of the result to check. Defaults to 0.
    pred: A predicate function that takes an SSAValue and returns a boolean.

  Returns:
    True if the i-th result exists and satisfies the given predicate, False
    otherwise.
  """
  if pred is None:
    return ResultPred(0, i)  # type: ignore

  def Pred(op: core.MlirOpBase):
    if i >= len(op.results):
      return False
    return pred(op.results[i])

  return Pred


@overload
def ResultShapePred(
    shape_pred: Callable[[list[int]], bool],
) -> Callable[[core.MlirOpBase], bool]:
  ...


@Predicate.Wraps
def ResultShapePred(
    i: int = 0, shape_pred: Callable[[list[int]], bool] = None
) -> Callable[[core.MlirOpBase], bool]:
  """Checks if the shape of the i-th result of an operation satisfies a given condition.

  Args:
    i: The index of the result to check. Defaults to 0.
    shape_pred: A predicate function that takes a shape (list of integers) and
      returns a boolean.

  Returns:
    True if the i-th result is a RankedTensorType and its shape satisfies the
    given predicate, False otherwise.
  """
  if shape_pred is None:
    return ResultShapePred(0, i)  # type: ignore

  def Pred(r: SSAValue):
    if not isinstance(r.type, mlir.RankedTensorType):
      return False
    return shape_pred(r.type.shape)

  return ResultPred(i, Pred)


@overload
def ResultHasRank(rank: int) -> Callable[[core.MlirOpBase], bool]:
  ...


@Predicate.Wraps
def ResultHasRank(
    i: int = 0, rank: int = -1
) -> Callable[[core.MlirOpBase], bool]:
  """Checks if the i-th result of an operation has the given rank.

  Args:
    i: The index of the result to check. Defaults to 0.
    rank: The expected rank.

  Returns:
    True if the i-th result is a RankedTensorType and has the given rank, False
    otherwise.
  """
  if rank == -1:
    return ResultHasRank(0, i)  # type: ignore
  return ResultShapePred(i, lambda s: len(s) == rank)


@overload
def ResultHasShape(shape: list[int]) -> Callable[[core.MlirOpBase], bool]:
  ...


@Predicate.Wraps
def ResultHasShape(
    i: int = 0, shape: list[int] = None  # type: ignore
) -> Callable[[core.MlirOpBase], bool]:
  """Checks if the i-th result of an operation has the given shape.

  Args:
    i: The index of the result to check. Defaults to 0.
    shape: The expected shape.

  Returns:
    True if the i-th result is a RankedTensorType and has the given shape, False
    otherwise.
  """
  if shape is None:
    return ResultHasShape(0, i)  # type: ignore
  return ResultShapePred(i, lambda s: s == shape)


@overload
def ResultHasOneUse() -> Callable[[core.MlirOpBase], bool]:
  ...


@Predicate.Wraps
def ResultHasOneUse(i: int = 0) -> Callable[[core.MlirOpBase], bool]:
  """Checks if the i-th result of an operation has exactly one use.

  Args:
    i: The index of the result to check. Defaults to 0.

  Returns:
    True if the i-th result has exactly one use, False otherwise.
  """
  return ResultPred(i, lambda x: len(x.uses) == 1)


@Predicate.Wraps
def ShapePred(shape_pred: Callable[[list[int]], bool]):
  def Pred(x: SSAValue):
    if not isinstance(x.type, mlir.RankedTensorType):
      return False
    return shape_pred(x.type.shape)

  return Pred


@Predicate.Wraps
def HasRank(rank: int):
  return ShapePred(lambda s: len(s) == rank)
