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
"""Match DAG instruction executors."""

from typing import Any, Sequence
from xdsl import irdl
from litert.python.tools import model_utils as mu
from litert.python.tools.model_utils.match import context
from litert.python.tools.model_utils.match._match_dag import generator
from litert.python.tools.model_utils.match._match_dag import instruction as inst
from litert.python.tools.model_utils.match._match_dag import tblgen_def

_RESULT_PREFIX = "__RESULT__"

SSAValue = irdl.SSAValue


class MatchResults(dict):

  def __getattr__(self, key):
    return self.__getitem__(key)

  def __getitem__(self, key):
    if super().__contains__("$" + key):
      return super().__getitem__("$" + key)
    return super().__getitem__(key)


def _execute_instruction(
    code: inst.Instruction, mapping: dict[inst.Var, Any]
) -> tuple[Any, dict[inst.Var, Any]]:
  """Executes a Match DAG instruction."""
  if isinstance(code, (inst.Null, inst.Comment)):
    return None, mapping
  if isinstance(code, inst.Var):
    return mapping[code], mapping
  if isinstance(code, inst.SetVar):
    value, mapping = _execute_instruction(code.value, mapping)
    mapping[code.var] = value
    return None, mapping
  if isinstance(code, inst.SetResult):
    value, mapping = _execute_instruction(code.value, mapping)
    mapping[_RESULT_PREFIX + code.name] = value
    return None, mapping
  if isinstance(code, inst.LookupDef):
    reg = tblgen_def.registry[code.name]
    return reg, mapping
  if isinstance(code, inst.Invoke):
    fn, mapping = _execute_instruction(code.fn, mapping)
    args, mapping = _execute_instruction(code.args, mapping)
    if isinstance(args, (tuple, list)):
      ret = fn(*args)
    else:
      ret = fn(args)
    return ret, mapping
  if isinstance(code, inst.Value):
    return code.value, mapping

  if isinstance(code, inst.Extract):
    v, mapping = _execute_instruction(code.var, mapping)
    assert isinstance(v, SSAValue)  # NOTE: Yes this is true
    if code.idx == 0:
      return v, mapping

    op = v.owner
    if not isinstance(op, mu.core.MlirOpBase):
      raise context.NoMatchError()

    i = code.idx - 1
    if len(op.operands) <= i:
      raise context.NoMatchError()
    v = op.operands[i]
    return v, mapping

  raise NotImplementedError(f"Executor for {type(code)} is not implemented")


def execute_match_dag(
    code: Sequence[inst.Instruction],
    root_op: mu.core.MlirOpBase,
):
  """Executes a Match DAG code (sequence of instructions)."""
  mapping = {generator.ROOT_VAR: root_op}
  for instruction in code:
    _, mapping = _execute_instruction(instruction, mapping)

  results = MatchResults()
  for key, value in mapping.items():
    if isinstance(key, str) and key.startswith(_RESULT_PREFIX):
      results[key[len(_RESULT_PREFIX) :]] = value
  return results
