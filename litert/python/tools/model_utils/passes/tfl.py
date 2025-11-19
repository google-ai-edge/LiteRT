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
