import abc
import functools
from typing import Callable, TypeVar
from xdsl.ir.core import *
from xdsl.irdl import *
from .. import core
from ..dialect import mlir


class Predicate(abc.ABC):

  @abc.abstractmethod
  def __call__(self, x) -> bool:
    raise NotImplementedError()

  @staticmethod
  def wraps(
      build_pred: Callable[..., Callable[[], bool]],
  ) -> type["Predicate"]:
    T = TypeVar("T")

    @functools.wraps(build_pred)
    def __init__(self, *args: Any, **kwargs: Any):
      self._pred: Callable[[T], bool] = build_pred(*args, **kwargs)

    def __call__(self, x: T) -> bool:
      return self._pred(x)

    cls = type(
        build_pred.__name__,
        (Predicate,),
        {
            "__init__": __init__,
            "__call__": __call__,
            "__doc__": build_pred.__doc__,
        },
    )
    return cls

  def __and__(self, other: "Predicate"):
    return PredicateAnd(self, other)

  def __or__(self, other: "Predicate"):
    return PredicateOr(self, other)


@Predicate.wraps
def PredicateAnd(pred1: Predicate, pred2: Predicate) -> Predicate:
  """Checks if an operation satisfies both of the given predicates.

  Args:
    pred1: The first predicate.
    pred2: The second predicate.

  Returns:
    True if the operation satisfies both `pred1` and `pred2`, False otherwise.
  """

  def _pred(x):
    return pred1(x) and pred2(x)

  return _pred


@Predicate.wraps
def PredicateOr(pred1: Predicate, pred2: Predicate) -> Predicate:
  """Checks if an operation satisfies at least one of the given predicates.

  Args:
    pred1: The first predicate.
    pred2: The second predicate.

  Returns:
    True if the operation satisfies either `pred1` or `pred2`, False otherwise.
  """

  def _pred(op: core.MlirOpBase):
    return pred1(op) or pred2(op)

  return _pred


@Predicate.wraps
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

  def _pred(op: core.MlirOpBase):
    if i >= len(op.operands):
      return False
    return pred(op.operands[i])

  return _pred


@Predicate.wraps
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

  def _pred(x: SSAValue):
    if not isinstance(x.type, mlir.RankedTensorType):
      return False
    return shape_pred(x.type.shape)

  return OperandPred(i, _pred)


@Predicate.wraps
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


@Predicate.wraps
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


@Predicate.wraps
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

  def _pred(op: core.MlirOpBase):
    if i >= len(op.results):
      return False
    return pred(op.results[i])

  return _pred


@overload
def ResultShapePred(
    shape_pred: Callable[[list[int]], bool],
) -> Callable[[core.MlirOpBase], bool]:
  ...


@Predicate.wraps
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

  def _pred(r: SSAValue):
    if not isinstance(r.type, mlir.RankedTensorType):
      return False
    return shape_pred(r.type.shape)

  return ResultPred(i, _pred)


@overload
def ResultHasRank(rank: int) -> Callable[[core.MlirOpBase], bool]:
  ...


@Predicate.wraps
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


@Predicate.wraps
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


@Predicate.wraps
def ResultHasOneUse(i: int = 0) -> Callable[[core.MlirOpBase], bool]:
  """Checks if the i-th result of an operation has exactly one use.

  Args:
    i: The index of the result to check. Defaults to 0.

  Returns:
    True if the i-th result has exactly one use, False otherwise.
  """
  return ResultPred(i, lambda x: len(x.uses) == 1)


@Predicate.wraps
def ShapePred(shape_pred: Callable[[list[int]], bool]):
  def _pred(x: SSAValue):
    if not isinstance(x.type, mlir.RankedTensorType):
      return False
    return shape_pred(x.type.shape)

  return _pred


@Predicate.wraps
def HasRank(rank: int):
  return ShapePred(lambda s: len(s) == rank)
