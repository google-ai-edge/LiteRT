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
"""TFL dialect passes."""

from litert.python.tools.model_utils.passes import mlir as mlir_pass


class OptimizePass(mlir_pass.MlirPass):
  """tfl-optimize pass."""

  def __init__(self):
    super().__init__("func.func(tfl-optimize)")


class PrepareQuantizePass(mlir_pass.MlirPass):
  """Remove qdq from input and output nodes after quantization."""

  def __init__(
      self,
      quantize_signed: bool = False,
      activation_number_of_bits: int = 8,
      post_training_quantize: bool = False,
      legacy_float_scale: bool = False,
      disable_per_channel: bool = False,
      disable_set_input_nodes_quantization_params: bool = False,
      qdq_conversion_mode: str = "Static",
  ):
    quantize_signed = str(quantize_signed).lower()
    post_training_quantize = str(post_training_quantize).lower()
    legacy_float_scale = str(legacy_float_scale).lower()
    disable_per_channel = str(disable_per_channel).lower()
    disable_set_input_nodes_quantization_params = str(
        disable_set_input_nodes_quantization_params
    ).lower()
    super().__init__(
        "func.func(tfl-prepare-quantize{"
        f"quantize-signed={quantize_signed} "
        f"activation-number-of-bits={activation_number_of_bits} "
        f"post-training-quantize={post_training_quantize} "
        f"legacy-float-scale={legacy_float_scale} "
        f"disable-per-channel={disable_per_channel} "
        f"disable-set-input-nodes-quantization-params={disable_set_input_nodes_quantization_params} "
        f"qdq-conversion-mode={qdq_conversion_mode} "
        "})"
    )


class PropagateQsvPass(mlir_pass.MlirPass):
  """Propagates Quantization Scale/Value (QSV) information through the graph.

  This transformation pass propagates the QSV data across operations in the
  TensorFlow Lite dialect.
  """

  def __init__(self):
    super().__init__("builtin.module(tfl-propagate-qsv)")
