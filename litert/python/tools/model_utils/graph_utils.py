import collections
import heapq
from typing import cast
import xdsl.ir.core

SSAValue = xdsl.ir.core.SSAValue
Block = xdsl.ir.core.Block
BlockArgument = xdsl.ir.core.BlockArgument


def topological_sort(block: xdsl.ir.core.Block) -> None:
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
  # TODO: Improve efficiency by avoiding detaching ops first.
  for op in new_ops[1:]:
    op.detach()
  for prev, curr in zip(new_ops, new_ops[1:]):
    block.insert_op_after(curr, prev)
