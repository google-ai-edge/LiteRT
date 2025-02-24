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

"""Utility for dynamically importing vendor backends based on ID."""

import importlib

from google3.third_party.odml.litert.litert.python.google.core import common
from google3.third_party.odml.litert.litert.python.google.core import types


class VendorModule:
  """A vendor module."""

  def __init__(self, class_name: str, *args):
    self._class_name = class_name
    self._module_path = ".".join([common.MODULE_ROOT, "vendors"] + list(args))

  @property
  def class_name(self) -> str:
    return self._class_name

  @property
  def module_path(self) -> str:
    return self._module_path


_VENDOR_IMPORTS = {
    "example": VendorModule("ExampleBackend", "example", "example_backend")
}


def import_vendor(backend_id: str) -> types.BackendT:
  """Imports a vendor backend class based on its ID.

  Args:
    backend_id: The ID of the backend to import.

  Returns:
    The imported backend class.

  Raises:
    ValueError: If the backend ID is not supported.
  """
  vendor_module = _VENDOR_IMPORTS.get(backend_id, None)
  if vendor_module is None:
    raise ValueError(f"Unsupported backend id: {backend_id}")

  module = importlib.import_module(vendor_module.module_path)
  backend_class = getattr(module, vendor_module.class_name)
  return backend_class
