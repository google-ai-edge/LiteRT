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
"""Instruction classes for Match DAG."""

import abc
import dataclasses
from typing import Any
import uuid


class Instruction(abc.ABC):
  pass


class Null(Instruction):
  pass


@dataclasses.dataclass(frozen=True)
class Comment(Instruction):
  content: str


@dataclasses.dataclass(frozen=True)
class Var(Instruction):
  _id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)


@dataclasses.dataclass(frozen=True)
class SetVar(Instruction):
  var: Var
  value: Instruction


@dataclasses.dataclass(frozen=True)
class SetResult(Instruction):
  name: str
  value: Instruction


@dataclasses.dataclass(frozen=True)
class LookupDef(Instruction):
  name: str


@dataclasses.dataclass(frozen=True)
class Invoke(Instruction):
  fn: Instruction
  args: Instruction


@dataclasses.dataclass(frozen=True)
class Value(Instruction):
  value: Any


@dataclasses.dataclass(frozen=True)
class Extract(Instruction):
  var: Var
  idx: int
