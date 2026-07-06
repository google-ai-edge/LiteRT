# Copyright 2026 Google LLC.
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
"""LiteRT mixed precision transformation library."""

from collections.abc import Callable
import pathlib
from typing import Any

from litert.python.mlir import ir
import numpy as np
from xdsl import irdl

from litert.python.tools import model_utils as mu
from litert.python.tools.model_utils.dialect import func
from litert.python.tools.model_utils.dialect import mlir
from litert.python.tools.model_utils.dialect import stablehlo
from litert.python.tools.model_utils.dialect import tfl


def clamp_model_add_ops(
    path: str | pathlib.Path,
) -> bytes:
  if isinstance(path, str):
    path = pathlib.Path(path)

  module, ctx = mu.read_flatbuffer(path)
  clamp_add_after_rms_norm(module, ctx)
  with ctx:
    return mu.write_flatbuffer(module)


def clamp_add_after_rms_norm(
    module: mlir.ModuleOp,
    ctx: ir.Context,
) -> None:
  with ctx:
    for op in module.walk():
      if (
          isinstance(op, stablehlo.CompositeOp)
          and op.composite_name == "odml.rms_norm"
      ):
        for result in op.results:
          for use in result.uses.copy():
            if isinstance(use.operation, tfl.AddOp):
              with mu.OpBuildingContext(
                  use.operation, insert_before=False, insert_after=True
              ):
                add_result_type = use.operation.results[0].type

                min_val = np.array(-65504.0, dtype=np.float32)
                max_val = np.array(65504.0, dtype=np.float32)
                if isinstance(add_result_type, mlir.RankedTensorType):
                  if add_result_type.elty == "f16":
                    min_val = np.array(-65504.0, dtype=np.float16)
                    max_val = np.array(65504.0, dtype=np.float16)

                max_op = tfl.maximum(use.operation.results[0], min_val)
                min_op = tfl.minimum(max_op, max_val)

                for add_use in use.operation.results[0].uses.copy():
                  if (
                      add_use.operation != max_op.owner
                      and add_use.operation != min_op.owner
                  ):
                    add_use.operation.operands[add_use.index] = min_op

    module.cleanup()


def convert_model_to_fp16(
    path: str | pathlib.Path,
    fp32_op_predicate: Callable[[irdl.Operation], bool] | None = None,
) -> bytes:
  if isinstance(path, str):
    path = pathlib.Path(path)

  module, ctx = mu.read_flatbuffer(path)
  convert_to_fp16(module, ctx, fp32_op_predicate)
  with ctx:
    return mu.write_flatbuffer(module)


def get_parent_op_of_type(
    op: irdl.Operation, target_type: type[Any]
) -> irdl.Operation | None:
  parent = op.parent_op()
  while parent is not None:
    if isinstance(parent, target_type):
      return parent
    parent = parent.parent_op()
  return None


def convert_to_fp16(
    module: mlir.ModuleOp,
    ctx: ir.Context,
    fp32_op_predicate: Callable[[irdl.Operation], bool] | None = None,
) -> None:
  with ctx:
    args_to_cast = []
    args_to_update = []
    ops_to_cast = []
    ops_to_update = []
    funcs_to_update = set()
    fp32_ops = set()
    visited = set()

    def _walk(original_op):

      for op in original_op.walk():
        if op not in visited:
          visited.add(op)
        else:
          continue

        parent_func = get_parent_op_of_type(op, func.FuncOp)
        if parent_func and parent_func in fp32_ops:
          continue

        if op == original_op:
          continue

        if isinstance(op, func.ReturnOp):
          continue

        if fp32_op_predicate and fp32_op_predicate(op):
          fp32_ops.add(op)
          if isinstance(op, stablehlo.CompositeOp):
            fp32_ops.add(op.decomposition_func)
          elif isinstance(op, tfl.SelectV2Op):
            if isinstance(op.operands[2].op, tfl.ConstOp):
              fp32_ops.add(op.operands[2].op)
          continue

        if op in fp32_ops:
          continue

        if isinstance(op, func.FuncOp):
          funcs_to_update.add(op)

          for arg in op.body.block.args:
            if not isinstance(arg.type, mlir.RankedTensorType):
              continue

            if arg.type.elty != "f32":
              continue

            args_to_cast.append(arg)

          _walk(op)

        elif isinstance(op, tfl.ConstOp):
          should_add = False
          for result in op.results:
            if (
                isinstance(result.type, mlir.RankedTensorType)
                and result.type.elty == "f32"
            ):
              should_add = True
              break
          if should_add:
            ops_to_cast.append(op)

        elif isinstance(op, stablehlo.CompositeOp):
          funcs_to_update.add(op.decomposition_func)

          for arg in op.decomposition_func.body.block.args:
            if not isinstance(arg.type, mlir.RankedTensorType):
              continue

            if arg.type.elty != "f32":
              continue

            args_to_update.append(arg)

          _walk(op.decomposition_func)
          ops_to_update.append(op)

        else:
          ops_to_update.append(op)

    _walk(module)

    for arg in args_to_cast:
      arg.type = mlir.RankedTensorType(arg.type.shape, "f16")
      for use in arg.uses.copy():
        if use.operation in fp32_ops:
          with mu.OpBuildingContext(use.operation, insert_before=True):
            cast = tfl.cast(arg, "f32")
            use.operation.operands[use.index] = cast

    for arg in args_to_update:
      arg.type = mlir.RankedTensorType(arg.type.shape, "f16")

    for op in ops_to_cast:
      for result in op.results:
        for use in result.uses.copy():
          # Skip if the use is in a fp32 op. Used for constant tensors.
          if use.operation in fp32_ops:
            continue
          with mu.OpBuildingContext(use.operation, insert_before=True):
            cast = tfl.cast(result, "f16")
            use.operation.operands[use.index] = cast

    for op in ops_to_update:
      for result in op.results:
        if not isinstance(result.type, mlir.RankedTensorType):
          continue
        if result.type.elty != "f32":
          continue

        result.type = mlir.RankedTensorType(result.type.shape, "f16")

    for op in fp32_ops:
      for i, operand in enumerate(op.operands):
        if (
            isinstance(operand.type, mlir.RankedTensorType)
            and operand.type.elty == "f16"
        ):
          with mu.OpBuildingContext(op, insert_before=True):
            cast = tfl.cast(operand, "f32")
            op.operands[i] = cast

      for result in op.results:
        if (
            not isinstance(result.type, mlir.RankedTensorType)
            or result.type.elty != "f32"
        ):
          continue

        for use in result.uses.copy():
          if use.operation not in fp32_ops:
            with mu.OpBuildingContext(use.operation, insert_before=True):
              cast = tfl.cast(result, "f16")
              use.operation.operands[use.index] = cast

    for func_op in funcs_to_update:
      func_op.update_function_type()

    module.cleanup()


def parse_fp32_ops(op_strs: list[str]) -> list[type]:
  dialect_map = {
      "tfl": tfl,
      "stablehlo": stablehlo,
  }
  classes = []
  for op_str in op_strs:
    parts = op_str.split(".")
    if len(parts) != 2:
      raise ValueError(
          f"Invalid op name: '{op_str}'. Expected format is"
          " '<dialect>.<OpName>', e.g., 'tfl.AddOp'."
      )
    dialect, op_name = parts
    if dialect not in dialect_map:
      raise ValueError(
          f"Unknown dialect '{dialect}' in '{op_str}'. Supported dialects:"
          f" {list(dialect_map.keys())}"
      )
    module = dialect_map[dialect]
    if not hasattr(module, op_name):
      raise ValueError(f"Op '{op_name}' not found in dialect '{dialect}'.")
    classes.append(getattr(module, op_name))
  return classes


def match_op_by_name(op: irdl.Operation, name_patterns: list[str]) -> bool:
  """Returns True if the operation name/location matches any of the patterns."""
  # Check composite name for stablehlo.CompositeOp
  if isinstance(op, stablehlo.CompositeOp) and op.composite_name:
    if any(pattern in op.composite_name for pattern in name_patterns):
      return True

  # Fallback to checking location
  loc = getattr(op, "location", None)
  if loc:
    loc_str = str(loc)
    if any(pattern in loc_str for pattern in name_patterns):
      return True

  return False
