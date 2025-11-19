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
"""Context for building operations with associated locations and tracking."""

from litert.python.mlir import ir
import xdsl
import xdsl.irdl
import xdsl.pattern_rewriter
import xdsl.rewriter
from . import dialect_base

__all__ = ["OpBuildingContext"]

InsertPoint = xdsl.rewriter.InsertPoint
PatternRewriter = xdsl.pattern_rewriter.PatternRewriter
MlirOpBase = dialect_base.MlirOpBase

_ctx_stack: list["OpBuildingContext"] = []


class OpBuildingContext:
  """A context manager for building operations with associated locations and tracking.

  This class provides a context where newly created operations automatically
  have their location set to a specified value. It also keeps track of the
  operations created within the context and inserts them before/after the
  matched op.
  """

  def __init__(
      self,
      anchor: ir.Location | MlirOpBase | PatternRewriter,
      *,
      no_insert: bool = False,
      insert_before: bool = True,
      insert_after: bool = False,
      lazy_insert: bool = False,
  ):
    self._rewriter = None
    if isinstance(anchor, PatternRewriter):
      self._rewriter = anchor
    elif isinstance(anchor, MlirOpBase):
      self._rewriter = PatternRewriter(anchor)

    self._anchor_op = None
    if isinstance(anchor, MlirOpBase):
      self._anchor_op = anchor
    elif isinstance(anchor, PatternRewriter):
      self._anchor_op = anchor.current_operation

    self.location = None
    if isinstance(anchor, ir.Location):
      self.location = anchor
    elif hasattr(self._anchor_op, "location"):
      self.location = self._anchor_op.location

    if self._rewriter is None:
      # Disable auto insertion if the anchor is not a PatternRewriter.
      no_insert = True

    self.new_ops = []
    self.no_insert = no_insert
    self.insert_before = insert_before
    self.insert_after = insert_after
    self.lazy_insert = lazy_insert

  @property
  def rewriter(self) -> PatternRewriter:
    if self._rewriter is None:
      raise ValueError("No rewriter available.")
    return self._rewriter

  def __enter__(self):
    _ctx_stack.append(self)
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    if not self.no_insert and self.lazy_insert:
      if self.insert_after:
        self.rewriter.insert_op_after_matched_op(self.new_ops)
      elif self.insert_before:
        self.rewriter.insert_op_before_matched_op(self.new_ops)

      self.rewriter.has_done_action = True

    _ctx_stack.pop()


def _overload_operation_init_with_location_setter(cls):
  """Overload operation init to set location if not explicitly specified."""
  op_init = cls.__init__

  def new_init(self, *args, **kwargs):
    op_init(self, *args, **kwargs)

    if not _ctx_stack:
      return

    ctx = _ctx_stack[-1]
    ctx.new_ops.append(self)

    if not hasattr(self, "location") or self.location is None:
      # Only set the location if it is not explicitly specified.
      self.location = ctx.location

    if not (ctx.no_insert or ctx.lazy_insert):
      if ctx.insert_after:
        ctx.rewriter.insert_op(self, InsertPoint.after(ctx._anchor_op))
        ctx._anchor_op = self
      elif ctx.insert_before:
        ctx.rewriter.insert_op(self, InsertPoint.before(ctx._anchor_op))

  cls.__init__ = new_init


_overload_operation_init_with_location_setter(xdsl.irdl.operations.Operation)
