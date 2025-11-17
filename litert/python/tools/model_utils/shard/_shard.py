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
"""Operation sharding rules and algorithm."""
import copy
from typing import Callable

import numpy as np

from litert.python.tools import model_utils as mu
from litert.python.tools.model_utils.dialect import mlir
from litert.python.tools.model_utils.dialect import tfl


def split(x: mlir.SSARankedTensorValue, dim: int, parts: int):
  mu.match.pred(isinstance(x.type, mlir.RankedTensorType))
  shape = x.type.shape
  if dim < 0:
    dim += len(shape)

  mu.match.pred(0 <= dim < len(shape))
  mu.match.pred(shape[dim] % parts == 0)

  return tfl.split(dim, x, parts)


def concat(xs: list[mlir.SSARankedTensorValue], dim: int):
  for x in xs:
    mu.match.pred(isinstance(x.type, mlir.RankedTensorType))
  if dim < 0:
    x = xs[0]
    dim += len(x.type.shape)
  return tfl.concatenation(xs, dim)


_default_sharders = {}


def register_sharder(op: mu.core.MlirOpBase | str):
  def reg(sharder: Callable[[mu.core.MlirOpBase, int, int], None]):
    nonlocal op
    if not isinstance(op, str):
      op = op.name
    _default_sharders[op] = sharder
    return sharder

  return reg


class BottomUpSharder:

  def __init__(
      self,
      sharders=None,
      *,
      max_shard_depth: int | None = None,
      sharding_allowlist: (
          set[mu.core.MlirOpBase | mlir.SSARankedTensorValue] | None
      ) = None,
      sharding_denylist: (
          set[mu.core.MlirOpBase | mlir.SSARankedTensorValue] | None
      ) = None,
  ):
    """Initializes the BottomUpSharder.

    Args:
        sharders: A dictionary mapping operation names (strings) to sharding
          functions. If None, defaults to `self.default_sharders()`.
        max_shard_depth: The maximum depth for recursive sharding. Sharding will
          stop if the depth exceeds this limit.
        sharding_allowlist: If provided, only operations/values in this set will
          be considered for sharding.
        sharding_denylist: If provided, only operations/values in this set will
          never be sharded.
    """

    if sharders is None:
      sharders = self.default_sharders()

    self.sharders = sharders

    self._shard_depth = 0
    self.max_shard_depth = max_shard_depth or 100000

    if sharding_allowlist is not None:
      sharding_allowlist = {
          x if isinstance(x, mu.core.MlirOpBase) else x.owner
          for x in sharding_allowlist
      }
    if sharding_denylist is not None:
      sharding_denylist = {
          x if isinstance(x, mu.core.MlirOpBase) else x.owner
          for x in sharding_denylist
      }

    self.sharding_allowlist = sharding_allowlist
    self.sharding_denylist = sharding_denylist

  @classmethod
  def default_sharders(cls):
    return copy.copy(_default_sharders)

  def split(self, x: mlir.SSARankedTensorValue, dim: int, parts: int):
    return split(x, dim, parts)

  def concat(self, xs: list[mlir.SSARankedTensorValue], dim: int):
    return concat(xs, dim)

  def shard(
      self,
      x: mlir.SSARankedTensorValue | mu.core.MlirOpBase,
      dim: int,
      parts: int,
  ) -> list[mlir.SSARankedTensorValue]:
    """The internal entry function to shard an op."""
    self._shard_depth += 1

    try:
      if parts <= 1:
        return [mlir.SSARankedTensorValue.get(x)]

      op = None
      if isinstance(x, mu.core.MlirOpBase):
        op = x
      elif isinstance(x.owner, mu.core.MlirOpBase):
        op = x.owner

      ok_to_shard = op is not None

      if self.max_shard_depth is not None:
        ok_to_shard = ok_to_shard and self._shard_depth <= self.max_shard_depth
      if self.sharding_allowlist is not None:
        ok_to_shard = ok_to_shard and op in self.sharding_allowlist
      if self.sharding_denylist is not None:
        ok_to_shard = ok_to_shard and op not in self.sharding_denylist

      if ok_to_shard:
        sharder = self.sharders.get(op.name)
        if sharder is not None:
          with mu.MatchingContext():
            with mu.core.OpBuildingContext(op):
              if len(op.results) == 1:
                # Auto-normalize the dimension.
                if dim < 0:
                  dim += op.results[0].type.rank
                assert 0 <= dim < op.results[0].type.rank
              results = sharder(self, op, dim, parts)
              if isinstance(results, (list, tuple)):
                return results

      # Reasons to get here:
      # 1. x does not have an owner op (e.g. a func.func argument)
      # 2. Sharder does not exist for the owner op.
      # 3. Sharding failed
      # 4. Sharding is not allowed for the owner op.
      if self._shard_depth == 1:
        return [mlir.SSARankedTensorValue.get(x)]

      # Try to split the tensor along the dimension.
      x = mlir.SSARankedTensorValue.get(x)
      return split(x, dim, parts)

    finally:
      self._shard_depth -= 1

  @register_sharder("tfl.no_value")
  def _shard_no_value(self, op, dim: int, parts: int):
    return [op.results[0]] * parts

  @register_sharder("tfl.add")
  @register_sharder("tfl.mul")
  @register_sharder("tfl.sub")
  def _shard_broadcastable_binary(
      self, op: mu.core.MlirOpBase, dim: int, parts: int
  ):
    o = op.results[0].type.shape

    mu.match.pred(-len(o) <= dim < len(o))
    mu.match.pred(o[dim] % parts == 0)

    xs = []
    ys = []
    for av, bv in [
        (op.operands[0], op.operands[1]),
        (op.operands[1], op.operands[0]),
    ]:
      a = av.type.shape
      b = bv.type.shape

      if len(a) != len(o):
        continue

      if len(b) == len(o) and a[dim] == b[dim] == o[dim]:
        xs = self.shard(av, dim, parts)
        ys = self.shard(bv, dim, parts)
      elif (
          not b
          or np.prod(b) == 1
          or (len(o) == len(b) and b[dim] == 1)
          or (len(o) > len(b) and dim < len(o) - len(b))
      ):
        # The dim on B is 1 (to be broadcated). No need to shard B.
        xs = self.shard(av, dim, parts)
        ys = [bv] * parts
      elif len(o) > len(b) and dim >= len(o) - len(b):
        # B would be broadcasted to A. Shard B with shifted dimension.
        xs = self.shard(av, dim, parts)
        ys = self.shard(bv, dim - len(o) + len(b), parts)

      if xs and ys:
        break

    if not (xs and ys):
      mu.match.fail()

    assert len(xs) == len(ys)

    new_shape = list(op.results[0].type.shape)
    new_shape[dim] = new_shape[dim] // parts

    sharded_results = []
    for x, y in zip(xs, ys):
      new_result_type = op.results[0].type.clone(shape=new_shape)
      new_op = type(op).build(
          operands=[x, y],
          result_types=[new_result_type],
          attributes=op.attributes,
          regions=op.regions,
      )
      new_op.name = op.name  # In case it's a generic MlirOp
      sharded_results.append(new_op.results[0])

    return sharded_results

  @register_sharder("tfl.broadcast_to")
  def _shard_broadcast_to(self, op: tfl.BroadcastToOp, dim: int, parts: int):
    ishape = op.input.type.shape
    oshape = op.output.type.shape

    mu.match.pred(
        len(ishape) == len(oshape)
        and dim < len(oshape)
        and oshape[dim] % parts == 0
    )

    input = op.input
    new_shape = list(shape)
    new_shape[dim] = new_shape[dim] // parts

    if oshape[dim] == 1:
      return [tfl.broadcast_to(input, new_shape) for _ in range(parts)]
    else:
      sharded_inputs = self.shard(input, dim, parts)
      return [tfl.broadcast_to(x, new_shape) for x in sharded_inputs]

  @register_sharder("tfl.conv_2d")
  def _shard_conv2d(self, op: tfl.Conv2DOp, dim: int, parts: int):
    out = op.output
    shape = out.type.shape

    # Only supports sharding along the channel dimension.
    mu.match.pred(dim == len(shape) - 1)
    # If the output is used by another op, do not shard but add a split after.
    mu.match.pred(len(out.uses) == 1)
    mu.match.pred(out.type.shape[dim] % parts == 0)

    filters = self.shard(op.filter, 0, parts)
    bias = self.shard(op.bias, 0, parts)

    shape = list(out.type.shape)
    shape[dim] = shape[dim] // parts
    result_type = out.type.clone(shape=shape)

    sharded_results = [
        tfl.conv_2d(
            input=op.input,
            filter=f,
            bias=b,
            result_type=result_type,
            **op.attributes,
        )
        for f, b in zip(filters, bias)
    ]
    return sharded_results
