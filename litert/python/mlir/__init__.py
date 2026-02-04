# Copyright 2025 The LiteRT Authors. All Rights Reserved.
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
"""MLIR Python bindings for LiteRT Compiler."""

import importlib
import sys

__all__ = []

ROOT_PACKAGE_NAME = "ai_edge_litert"
MLIR_PACKAGE_NAME = f"{ROOT_PACKAGE_NAME}.mlir"
MLIR_LIBS_PACKAGE_NAME = f"{ROOT_PACKAGE_NAME}.mlir._mlir_libs"
MLIR_LIBS_SO_MODULE_NAMES = [
    "_mlir",
    "model_utils_ext",
    "converter_api_ext",
    "_stablehlo",
]


def _import_mlir_so_modules():
  """Manually imports MLIR so modules and adds them to the import search path."""
  for so_module_name in MLIR_LIBS_SO_MODULE_NAMES:
    old_mlir_so_module_name = f"{MLIR_LIBS_PACKAGE_NAME}.{so_module_name}"
    new_mlir_so_module_name = f"{ROOT_PACKAGE_NAME}.{so_module_name}"

    mlir_so_module = importlib.import_module(new_mlir_so_module_name)
    sys.modules[old_mlir_so_module_name] = mlir_so_module

    # Import the mlir_libs module to make sure it's imported before any other
    # modules that might be imported from it.
    importlib.import_module(MLIR_LIBS_PACKAGE_NAME)

    for sub_module in dir(mlir_so_module):
      if sub_module.startswith("_"):
        continue

      try:
        old_name = f"{old_mlir_so_module_name}.{sub_module}"
        new_name = f"{new_mlir_so_module_name}.{sub_module}"
        sys.modules[old_name] = importlib.import_module(new_name)
      except ModuleNotFoundError:
        continue


_import_mlir_so_modules()
