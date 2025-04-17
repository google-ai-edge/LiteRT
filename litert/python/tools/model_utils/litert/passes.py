"""Passes and utilities for use by litert backends."""

# pytype: disable=not-callable

import copy

from litert.python.tools import model_utils as mu
from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import tfl

################################################################################
# Utilities
################################################################################

# These will move to core litert lib.


def call_pass(name: str, input_model: bytes) -> bytes:
  original_module, ctx = mu.read_flatbuffer(content=input_model)
  module = copy.deepcopy(original_module)

  passes = {p.name: p for p in [ExampleRewrites, QualcommRewrites]}
  if name not in passes:
    return input_model

  with ctx:
    passes[name]()(module)
    # Add verify when bug is fixed with xdsl.
    module.cleanup()
    return mu.write_flatbuffer(module)


################################################################################
# Example Backend
################################################################################

# These will move to example vendor litert lib.


class ExampleRewrites(core.RewritePatternPassBase):
  """A simple pass that rewrites tfl.mul to tfl.add based on a pattern."""

  name = "example-rewrites"


@ExampleRewrites.register_rewrite_pattern(tfl.MulOp)
def mul_to_add(op: tfl.MulOp, rewriter) -> None:
  """A pattern that replaces tfl.mul with tfl.add.

  Args:
    op: The tfl.mul op.
    rewriter: The rewriter to use.
  """

  with mu.MatchingContext():
    mu.match.pred(op.name == "tfl.mul")
    mu.match.pred(op.fused_activation_function == "NONE")

    lhs, rhs = op.operands
    out = op.results[0]

    with core.OpBuildingContext(anchor=rewriter):
      new_out = tfl.add(lhs, rhs)
      out.replace_by(new_out)

      rewriter.erase_op(op)


################################################################################
# Qualcomm Backend.
################################################################################

# These will move to qualcomm vendor litert lib.


class QualcommRewrites(core.RewritePatternPassBase):
  """A simple pass that rewrites tfl.mul to tfl.add based on a pattern."""

  name = "qualcomm-rewrites"


@QualcommRewrites.register_rewrite_pattern(tfl.FullyConnectedOp)
def fully_connected_to_conv2d(op: tfl.FullyConnectedOp, rewriter) -> None:
  # TODO(lukeboyer): Implement this.
  pass
