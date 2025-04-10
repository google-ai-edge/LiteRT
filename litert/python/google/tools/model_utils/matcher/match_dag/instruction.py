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
