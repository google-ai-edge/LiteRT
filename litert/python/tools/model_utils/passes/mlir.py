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
"""MLIR builtin passes."""

from litert.python.mlir import passmanager
from litert.python.tools.model_utils import core
from litert.python.tools.model_utils import transform
from litert.python.tools.model_utils.dialect import mlir


class MlirPass(core.ModulePassBase):
  """Run registered MLIR pass and pass pipeline via passmanager."""

  def __init__(self, pipeline: str):
    if not pipeline.startswith("builtin.module("):
      pipeline = "builtin.module(" + pipeline + ")"
    self.pipeline = pipeline

  def call(self, module: mlir.ModuleOp):
    pm = passmanager.PassManager.parse(self.pipeline)
    ir_module = transform._python_to_mlir(module)  # pylint: disable=protected-access

    pm.run(ir_module)

    new_module = transform._mlir_to_python(ir_module)  # pylint: disable=protected-access

    module.replace_by(new_module)
    return module


class CsePass(MlirPass):
  """Eliminate common sub-expressions."""

  def __init__(self):
    super().__init__("builtin.module(cse)")


class CanonicalizePass(MlirPass):
  """Canonicalize operations."""

  def __init__(self):
    super().__init__("builtin.module(canonicalize)")
