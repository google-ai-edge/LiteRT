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
"""Utilities for working with ModelUtils graphs."""

import collections
import heapq
from typing import cast
import xdsl
from xdsl import irdl
from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import func
from litert.python.tools.model_utils.dialect import mlir

SSAValue = irdl.SSAValue
Block = xdsl.ir.Block
BlockArgument = xdsl.ir.BlockArgument


def topological_sort(block: Block) -> None:
  """Sorts the operations in the block topologically."""
  ops = list(block.ops)

  indices = {op: i for i, op in enumerate(ops)}
  incnt = collections.Counter({op: len(op.operands) for op in ops})

  h = []
  for op in ops:
    for operand in op.operands:
      if isinstance(operand, BlockArgument):
        incnt[op] -= 1
    if incnt[op] == 0:
      heapq.heappush(h, (indices[op], op))

  new_ops = []
  while h:
    _, op = heapq.heappop(h)
    new_ops.append(op)
    for result in op.results:
      result = cast(SSAValue, result)
      for use in result.uses:
        incnt[use.operation] -= 1
        if incnt[use.operation] == 0:
          heapq.heappush(h, (indices[use.operation], use.operation))

  if len(new_ops) != len(ops):
    raise ValueError(
        "Failed to sort the operations in the block. Maybe some operands'"
        " owners are not in the block or there are cycles."
    )

  # Detach the ops from the block and reinsert them in the correct order.
  # TODO(cnchan): Improve efficiency by avoiding detaching ops first.
  for op in new_ops[1:]:
    op.detach()
  for prev, curr in zip(new_ops, new_ops[1:]):
    block.insert_op_after(curr, prev)


def inline_call_like_op(
    call_like_op: core.MlirOpBase,
    impl: Block | func.FuncOp,
    *,
    erase_impl: bool = False,
) -> None:
  """Replaces a call-like operation with the implementation.

  Args:
    call_like_op: The call-like operation to inline.
    impl: The implementation for the call-like operation.
    erase_impl: Whether to remove the impl (applicable when impl is a
      func.FuncOp).
  """
  if not isinstance(impl, (Block, func.FuncOp)):
    raise ValueError(
        f"The implementation must be a Block or a func.FuncOp. Got {type(impl)}"
    )

  input_impl = impl
  if isinstance(impl, func.FuncOp):
    impl = impl.body.block

  mapping = {}

  if len(call_like_op.operands) != len(impl.args):
    raise ValueError(
        "Expect number of operands to be the same. Got"
        f" {len(call_like_op.operands)} vs {len(impl.args)}"
    )

  for x, y in zip(call_like_op.operands, impl.args):
    mapping[y] = x

  for impl_op in impl.ops:
    if isinstance(impl_op, func.ReturnOp):
      if len(impl_op.operands) != len(call_like_op.results):
        raise ValueError(
            "Expect number of results to be the same. Got"
            f" {len(call_like_op.results)} vs {len(impl_op.operands)}"
        )
      for x, y in zip(call_like_op.results, impl_op.operands):
        x.replace_by(mapping[y])
      break

    with core.OpBuildingContext(call_like_op):
      # TODO(cnchan): Create new op with dedicated op classes.
      new_op = mlir.MlirOp(
          name=impl_op.name,
          operands=[mapping[x] for x in impl_op.operands],
          result_types=[r.type for r in impl_op.results],
          attributes=impl_op.attributes,
          location=impl_op.location,
      )
    for x, y in zip(new_op.results, impl_op.results):
      mapping[y] = x

  call_like_op.detach()
  call_like_op.erase()
  if erase_impl and isinstance(input_impl, func.FuncOp):
    input_impl.detach()
    input_impl.erase()
