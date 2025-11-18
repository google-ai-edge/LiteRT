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
"""C++ Python bindings for ModelUtils."""
from litert.python.mlir import ir

# pylint: disable=wildcard-import
from litert.python.mlir._mlir_libs import model_utils_ext
from litert.python.mlir._mlir_libs.model_utils_ext import *
# pylint: enable=wildcard-import


def flatbuffer_to_mlir(buffer: bytes, ir_context: ir.Context | None = None):
  if ir_context is None:
    ir_context = ir.Context.current
  return model_utils_ext.flatbuffer_to_mlir(buffer, ir_context)
