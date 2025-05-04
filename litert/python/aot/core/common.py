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
from importlib.resources import abc
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
    resource_root: abc.Traversable = resources.files(_WORKSPACE_PREFIX)
  except ModuleNotFoundError:
    resource_root: abc.Traversable = resources.files(_AI_EDGE_LITERT_PREFIX)
  litert_resource: abc.Traversable = resource_root.joinpath(
      _LITERT_ROOT, str(litert_relative_path)
  )
  if not litert_resource.is_file():
    raise FileNotFoundError(f"Resource {litert_resource} does not exist.")
  return pathlib.Path(str(litert_resource))


def is_tflite(path: pathlib.Path) -> bool:
  return path.exists() and path.is_file() and path.suffix == f".{TFLITE}"
