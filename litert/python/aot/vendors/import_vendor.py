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

import copy
import dataclasses
from typing import Any, Iterable

from litert.python.aot.core import types
from litert.python.aot.vendors import fallback_backend


@dataclasses.dataclass
class VendorRegistry:
  """A vendor registry."""

  backend_class: types.BackendT


_VENDOR_REGISTRY: dict[str, VendorRegistry] = {}


def register_backend(
    backend_class: types.BackendT,
):
  backend_id = backend_class.id()
  _VENDOR_REGISTRY[backend_id] = VendorRegistry(backend_class)

  return backend_class

register_backend(fallback_backend.FallbackBackend)


def import_vendor(backend_id: str) -> types.BackendT:
  """Imports a vendor backend class based on its ID.

  Args:
    backend_id: The ID of the backend to import.

  Returns:
    The imported backend class.

  Raises:
    ValueError: If the backend ID is not supported.
  """
  vendor_module = _VENDOR_REGISTRY.get(backend_id, None)
  if vendor_module is None:
    raise ValueError(f'Unsupported backend id: {backend_id}')

  return vendor_module.backend_class


class AllRegisteredTarget(types.Target):
  """A virtual Compilation target."""

  def __hash__(self) -> int:
    raise NotImplementedError()

  def __eq__(self, other) -> bool:
    raise NotImplementedError()

  def __repr__(self) -> str:
    raise NotImplementedError()

  @classmethod
  def backend_id(cls) -> str:
    return 'all'

  def flatten(self) -> dict[str, Any]:
    return {'backend_id': self.backend_id()}


@register_backend
class AllRegisteredBackend(types.Backend):
  """A virtual backend that represents all registered backends."""

  # NOTE: Only initialize through "create".
  def __init__(self, config: types.Config):
    self._config = config

  @classmethod
  def create(cls, config: types.Config) -> 'AllRegisteredBackend':
    return AllRegisteredBackend(config)

  @classmethod
  def id(cls) -> str:
    return 'all'

  @property
  def target(self) -> AllRegisteredTarget:
    return AllRegisteredTarget()

  @property
  def target_id(self) -> str:
    return ''

  @property
  def config(self) -> types.Config:
    return self._config

  def call_component(
      self,
      input_model: types.Model,
      output_model: types.Model,
      component: types.Component,
  ):
    del input_model, output_model, component
    raise NotImplementedError(
        'AllRegisteredBackend does not support any component.'
    )

  def specialize(self) -> Iterable[types.Backend]:
    for backend_id, vendor_module in _VENDOR_REGISTRY.items():
      if backend_id == 'all':
        continue
      config = copy.deepcopy(self.config)
      config['backend_id'] = backend_id
      backend = vendor_module.backend_class.create(config)
      yield from backend.specialize()
