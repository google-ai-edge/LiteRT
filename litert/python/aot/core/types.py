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

"""Basic types used in the LiteRt AOT flow."""

import abc
from collections.abc import Iterable
import dataclasses
import pathlib
import sys
from typing import Any, MutableMapping, Protocol, Type

# pylint: disable=g-importing-member
# pylint: disable=g-import-not-at-top
# pylint: disable=g-bad-import-order
if sys.version_info < (3, 10):
  from typing_extensions import TypeAlias
else:
  from typing import TypeAlias
# pylint: enable=g-bad-import-order
# pylint: enable=g-import-not-at-top
# pylint: enable=g-importing-member


@dataclasses.dataclass(frozen=True)
class SubgraphPartitionStats:
  """Subgraph partition stats."""

  subgraph_index: int
  num_ops_offloaded: int
  num_total_ops: int
  num_partitions_offloaded: int

  def __str__(self) -> str:
    is_full_offload = self.num_ops_offloaded == self.num_total_ops
    return (
        'Subgraph'
        f' {self.subgraph_index} {"fully" if is_full_offload else "partially"}'
        f' compiled:\t{self.num_ops_offloaded} /'
        f' {self.num_total_ops} ops offloaded to'
        f' {self.num_partitions_offloaded} partitions.'
    )


@dataclasses.dataclass(frozen=True)
class PartitionStats:
  """Model partition stats."""

  subgraph_stats: list[SubgraphPartitionStats]

  def __str__(self) -> str:
    return '\n'.join(str(s) for s in self.subgraph_stats)


class Model:
  """A model.

  Note: If the model is not in memory, data_ will be a path to a file on disk.
  If the model is in memory, data_ will be the model bytes.

  However, there's no guarantee that the path will be a valid path to a file
  on disk, and/or that the file are a valid TFLite model.
  """

  data_: pathlib.Path | bytes
  partition_stats: PartitionStats | None = None

  def __init__(
      self,
      path: pathlib.Path | str | None = None,
      model_bytes: bytes | None = None,
  ):
    if path is not None:
      if isinstance(path, str):
        path = pathlib.Path(path)
      if model_bytes:
        raise ValueError('Cannot specify both path and model_bytes.')
      self.data_ = path
    else:
      if model_bytes is None:
        raise ValueError('Cannot specify neither path nor model_bytes.')
      self.data_ = model_bytes

  @property
  def in_memory(self) -> bool:
    return isinstance(self.data_, bytes)

  @property
  def path(self) -> pathlib.Path:
    if not isinstance(self.data_, pathlib.Path):
      raise ValueError('Model is not on disk.')
    return self.data_

  @property
  def model_bytes(self) -> bytes:
    if not isinstance(self.data_, bytes):
      raise ValueError('Model is not in memory.')
    return self.data_

  @classmethod
  def create_from_path(cls, path: pathlib.Path) -> 'Model':
    return Model(path=path, model_bytes=None)

  @classmethod
  def create_from_bytes(cls, model_bytes: bytes) -> 'Model':
    return Model(path=None, model_bytes=model_bytes)

  def set_path(self, path: pathlib.Path | str):
    if isinstance(path, str):
      path = pathlib.Path(path)
    self.data_ = path

  def set_bytes(self, model_bytes: bytes):
    self.data_ = model_bytes

  def load(self):
    """Loads the model from the given path.

    Raises:
      ValueError: If the model is already in memory.
    """
    if not isinstance(self.data_, pathlib.Path):
      raise ValueError('Cannot load a model that is already in memory.')
    self.data_ = self.data_.read_bytes()

  def save(self, path: pathlib.Path | str, export_only: bool = False):
    """Saves the model to the given path from the in-memory model content.

    If export_only is True, the model will be copied to the given path without
    modifying the internal state, regardless of whether the model is already on
    disk or in memory.

    Args:
      path: The path to save the model to.
      export_only: Whether to only export the model without modifying the
        internal stat (i.e. transfer the in-memory model to disk).

    Raises:
      ValueError: If export_only is False and the model is not in memory.
    """
    if isinstance(path, str):
      path = pathlib.Path(path)
    if isinstance(self.data_, pathlib.Path):
      if not export_only:
        raise ValueError(
            'Cannot save a model that is not in memory. Use export_only=True'
            ' for copying the model to a new path.'
        )
      with open(self.data_, 'rb') as f:
        model_content = f.read()
    else:
      model_content = self.data_
    path.write_bytes(model_content)
    if not export_only:
      self.data_ = path


@dataclasses.dataclass()
class CompilationResult:
  """Compilation result, as a collection of compiled models."""

  models_with_backend: list[tuple['Backend', Model]] = dataclasses.field(
      default_factory=list
  )
  failed_backends: list[tuple['Backend', str]] = dataclasses.field(
      default_factory=list
  )

  @property
  def models(self) -> list[Model]:
    return [model for _, model in self.models_with_backend]

  def load(self):
    for _, model in self.models_with_backend:
      if not model.in_memory:
        model.load()

  def export(self, output_dir: pathlib.Path | str, model_name: str = 'model'):
    if isinstance(output_dir, str):
      output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for backend, model in self.models_with_backend:
      model.save(
          output_dir / (model_name + backend.target_id_suffix + '.tflite'),
          export_only=True,
      )

  def compilation_report(self) -> str:
    """Returns a human readable compilation report."""
    report = []
    for backend, model in self.models_with_backend:
      report.append(f'{backend.target_id}')
      report.append('==========================')
      report.append(f'Partition Stats:\n{model.partition_stats}\n')
    report = '\n'.join(report)

    failed_report = []
    if self.failed_backends:
      failed_report.append('==========================')
      failed_report.append('COMPILATION FAILURES:')
      failed_report.append('==========================')
      for backend, error in self.failed_backends:
        failed_report.append(f'{backend.target_id}\t{error}')
    failed_report = '\n'.join(failed_report)
    return '\n'.join([report, failed_report])


class Component(Protocol):
  """An arbitrary module in the AOT flow that inputs and outputs a Model.

  For example quantizer, graph rewriter, compiler plugin etc.
  """

  @property
  def component_name(self) -> str:
    ...

  def __call__(self, input_model: Model, output_model: Model, *args, **kwargs):
    ...


# A user provided configuration. This will contain all the information needed
# to select the proper backend and run components (e.g. quant recipe,
# backend id etc). Backends will validate and resolve configurations and are
# ultimately responsible deciding how to configure the components.
# NOTE: Consider a typed config approach (proto, data class, etc.)
Config: TypeAlias = MutableMapping[str, Any]


# Backend specific compilation configuration.
BackendCompilationConfig: TypeAlias = MutableMapping[str, Any]


# The following is experimental and for protyping only.
class CompilationConfig:
  """A typed configuration."""

  target: 'Target'
  compilation_config: BackendCompilationConfig = dataclasses.field(
      default_factory=dict
  )
  quant_recipe: str | None = None

  def __init__(self, target: 'Target', **kwargs: Any):
    self.target = target
    self.quant_recipe = kwargs.pop('quantize_recipe', None)
    self.compilation_config = kwargs

  def to_dict(self) -> dict[str, Any]:
    ret = self.target.flatten() | self.compilation_config
    if self.quant_recipe is not None:
      ret['quantize_recipe'] = self.quant_recipe
    return ret


class Backend(metaclass=abc.ABCMeta):
  """A backend pertaining to a particular SoC vendor.

  Mainly responsible for resolving configurations and managing vendor specific
  resources (e.g. .so etc).
  """

  # NOTE: Only initialize through "create".
  def __init__(self, config: Config):
    self._config = config

  @classmethod
  @abc.abstractmethod
  def create(cls, config: Config) -> 'Backend':
    """Creates a backend instance.

    If no target is specified, the backend will represent all targets.

    Args:
      config: The compilation configuration.

    Returns:
      The backend instance.
    """

  @classmethod
  @abc.abstractmethod
  def id(cls) -> str:
    pass

  @property
  @abc.abstractmethod
  def target(self) -> 'Target':
    pass

  @property
  @abc.abstractmethod
  def target_id(self) -> str:
    pass

  @property
  def target_id_suffix(self) -> str:
    if self.target_id:
      return '_' + self.target_id
    return ''

  @property
  def config(self) -> Config:
    return self._config

  @property
  def soc_manufacturer(self) -> str:
    """Manufacturer name or enum."""
    raise NotImplementedError()

  @property
  def soc_model(self) -> str:
    """Model name or enum."""
    raise NotImplementedError()

  @property
  def shared_pass_names(self) -> list[str]:
    """Names of shared passes."""
    raise NotImplementedError()

  @property
  def quantize_recipe(self) -> str | None:
    """Optional quantization recipe."""
    return None

  @abc.abstractmethod
  def call_component(
      self, input_model: Model, output_model: Model, component: Component
  ):
    pass

  def specialize(self) -> Iterable['Backend']:
    yield self


BackendT: TypeAlias = Type[Backend]


class Target(metaclass=abc.ABCMeta):
  """Compilation target."""

  @abc.abstractmethod
  def __hash__(self) -> int:
    pass

  @abc.abstractmethod
  def __eq__(self, other) -> bool:
    pass

  @abc.abstractmethod
  def __repr__(self) -> str:
    pass

  @classmethod
  @abc.abstractmethod
  def backend_id(cls) -> str:
    pass

  @abc.abstractmethod
  def flatten(self) -> dict[str, Any]:
    return {'backend_id': self.backend_id()}
