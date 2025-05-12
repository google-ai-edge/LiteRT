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

"""Constants and other small generic utilities."""

from importlib import resources
import os
import pathlib

TFLITE = "tflite"
DOT_TFLITE = f".{TFLITE}"
NPU = "npu"


_WORKSPACE_PREFIX = "litert"
_AI_EDGE_LITERT_PREFIX = "ai_edge_litert"
_LITERT_ROOT = ""
_PYTHON_ROOT = "python/aot"

MODULE_ROOT = ".".join([
    _WORKSPACE_PREFIX,
    _LITERT_ROOT.replace("/", "."),
    _PYTHON_ROOT.replace("/", "."),
])


def get_resource(litert_relative_path: pathlib.Path) -> pathlib.Path:
  """Returns the path to a resource in the Litert workspace."""
  try:
    resource_root = resources.files(_WORKSPACE_PREFIX)
  except ModuleNotFoundError:
    resource_root = resources.files(_AI_EDGE_LITERT_PREFIX)
  litert_resource = resource_root.joinpath(
      _LITERT_ROOT, str(litert_relative_path)
  )
  if not litert_resource.is_file():
    raise FileNotFoundError(f"Resource {litert_resource} does not exist.")
  return pathlib.Path(str(litert_resource))


def is_tflite(path: pathlib.Path) -> bool:
  return path.exists() and path.is_file() and path.suffix == f".{TFLITE}"


def construct_ld_library_path() -> str:
  """Constructs a string suitable for the LD_LIBRARY_PATH environment variable.

  This function is used in ai_edge_litert python package, when the shared
  libraries are not in a static location. This function will construct the
  LD_LIBRARY_PATH environment variable using the ai_edge_litert directory, and
  all subdirectories.

  If the module is built from source, this function will return an empty string.

  Returns:
    A string suitable for the LD_LIBRARY_PATH environment variable.
  """
  try:
    resource_root = resources.files(_AI_EDGE_LITERT_PREFIX)
  except ModuleNotFoundError:
    # Bulit from source case.
    return ""
  root_package_path = str(resource_root)

  library_paths = set()

  library_paths.add(os.path.abspath(root_package_path))

  for dirpath, _, _ in os.walk(root_package_path):
    library_paths.add(os.path.abspath(dirpath))

  sorted_paths = sorted(list(library_paths))
  new_ld_library_path = os.pathsep.join(sorted_paths)
  current_ld_library_path = os.environ.get("LD_LIBRARY_PATH")

  if current_ld_library_path:
    if current_ld_library_path not in new_ld_library_path:
      lib_paths = f"{new_ld_library_path}{os.pathsep}{current_ld_library_path}"
    else:
      lib_paths = new_ld_library_path
  else:
    lib_paths = new_ld_library_path
  return lib_paths
