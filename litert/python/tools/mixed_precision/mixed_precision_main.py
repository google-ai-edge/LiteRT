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
"""Command line tool for mixed precision transformations of TFLite models."""

from collections.abc import Sequence
import pathlib

from absl import app
from absl import flags

import os # import gfile
from litert.python.tools import model_utils as mu
from litert.python.tools.mixed_precision import mixed_precision
from litert.python.tools.model_utils.dialect import stablehlo
from litert.python.tools.model_utils.dialect import tfl

_INPUT_FILE = flags.DEFINE_string(
    "input_file",
    None,
    "Path to the input TFLite model.",
    required=True,
)
_OUTPUT_FILE = flags.DEFINE_string(
    "output_file",
    None,
    "Path to the output TFLite model.",
    required=True,
)
_CLAMP_ADD_OPS_AFTER_RMS_NORM = flags.DEFINE_bool(
    "clamp_add_ops_after_rms_norm",
    False,
    "If true, clamps add operations after RMS norm.",
)
_CONVERT_TO_FP16 = flags.DEFINE_bool(
    "convert_to_fp16",
    False,
    "If true, converts the model to FP16.",
)
_KEEP_GROUPNORM_FP32 = flags.DEFINE_bool(
    "keep_groupnorm_fp32",
    False,
    "If true, keeps groupnorm ops (matching feature/norm_0 and feature/norm_1 "
    "location patterns) and cumsum ops in fp32 during fp16 conversion.",
)
_FP32_OPS = flags.DEFINE_list(
    "fp32_ops",
    [],
    "List of ops to keep in FP32 format during conversion, "
    "e.g., tfl.AddOp,tfl.CumsumOp.",
)
_FP32_NAMES = flags.DEFINE_list(
    "fp32_names",
    [],
    "List of name patterns/substrings to keep in FP32. Matching is done against"
    " the operation location (e.g., 'layer_name', 'node_name') or composite"
    " name.",
)


def _default_fp32_predicate(op) -> bool:
  """Default predicate specifying which operations must remain in Float32.

  By default, keeping 'odml.rms_norm' stablehlo composite operations and
  tfl.Add operations in Float32 is recommended to maintain model accuracy.

  Args:
    op: The operation to inspect.

  Returns:
    True if the operation should be kept in Float32 precision, False otherwise.
  """
  if isinstance(op, stablehlo.CompositeOp):
    if "odml.rms_norm" == op.composite_name:
      return True
  if isinstance(op, tfl.AddOp):
    return True
  return False


def main(argv: Sequence[str]) -> None:
  """Main entry point for the mixed-precision command line tool.

  Parses command line flags, reads the input TFLite model, applies requested
  mixed-precision transformations (clamping add operations and/or converting
  to Float16 precision), and writes the resulting model back to the output path.

  Args:
    argv: List of command-line arguments.

  Raises:
    app.UsageError: If unexpected arguments are provided or flags are invalid.
  """
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  input_path = _INPUT_FILE.value
  output_path = _OUTPUT_FILE.value

  print(f"Reading model from: {input_path}")

  def _groupnorm_fp32_predicate(op) -> bool:
    """Predicate to keep groupnorm and cumsum operations in Float32."""
    if isinstance(op, tfl.CumsumOp):
      return True
    loc_str = str(op.location)
    if "feature/norm_0" in loc_str or "feature/norm_1" in loc_str:
      print("Found op in norm_0/norm_1:", loc_str)
      return True
    return False

  fp16_convert = (
      _CONVERT_TO_FP16.value
      or _KEEP_GROUPNORM_FP32.value
      or bool(_FP32_OPS.value)
      or bool(_FP32_NAMES.value)
  )

  predicates = []
  if _KEEP_GROUPNORM_FP32.value:
    predicates.append(_groupnorm_fp32_predicate)
  if _FP32_OPS.value:
    try:
      op_classes = mixed_precision.parse_fp32_ops(_FP32_OPS.value)
    except ValueError as e:
      raise app.UsageError(str(e)) from e
    predicates.append(lambda op: any(isinstance(op, cls) for cls in op_classes))
  if _FP32_NAMES.value:
    predicates.append(
        lambda op: mixed_precision.match_op_by_name(op, _FP32_NAMES.value)
    )

  if not predicates:
    predicate = _default_fp32_predicate
  else:

    def combined_predicate(op) -> bool:
      """Combines all registered predicates using logical OR."""
      return any(pred(op) for pred in predicates)

    predicate = combined_predicate

  path = pathlib.Path(input_path)
  module, ctx = mu.read_flatbuffer(path)

  if _CLAMP_ADD_OPS_AFTER_RMS_NORM.value:
    mixed_precision.clamp_add_after_rms_norm(module, ctx)

  if fp16_convert:
    mixed_precision.convert_to_fp16(module, ctx, predicate)

  if not _CLAMP_ADD_OPS_AFTER_RMS_NORM.value and not fp16_convert:
    print(
        "No transformation flags provided (e.g. --clamp_add_ops_after_rms_norm"
        " or --convert_to_fp16). Just reading and writing the model."
    )

  with ctx:
    fbs = mu.write_flatbuffer(module)

  print(f"Writing to: {output_path}")
  with open(output_path, "wb") as f:
    f.write(fbs)


if __name__ == "__main__":
  app.run(main)
