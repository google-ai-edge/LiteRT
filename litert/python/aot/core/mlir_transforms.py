# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Wrapper for suite of MLIR passes."""

from ai_edge_litert.aot.core import components
from ai_edge_litert.aot.core import tflxx_util
from ai_edge_litert.aot.core import types


class MlirTransforms(components.MlirTransformsT):
  """Wrapper for suite of MLIR passes."""

  def __call__(
      self,
      input_model: types.Model,
      output_model: types.Model,
      pass_name: str,
  ):
    if not input_model.in_memory:
      input_model.load()
    input_bytes = input_model.model_bytes
    output_bytes = tflxx_util.call_tflxx(input_bytes, pass_name)
    output_model.set_bytes(output_bytes)
