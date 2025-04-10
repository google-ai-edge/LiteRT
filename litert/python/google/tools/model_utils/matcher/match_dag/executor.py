from typing import Any, Sequence
from xdsl.irdl import SSAValue
from litert.python.google.tools import model_utils as mu
from litert.python.google.tools.model_utils.matcher import context
from litert.python.google.tools.model_utils.matcher.match_dag import generator
from litert.python.google.tools.model_utils.matcher.match_dag import instruction as inst
from litert.python.google.tools.model_utils.matcher.match_dag import tblgen_def

_RESULT_PREFIX = "__RESULT__"


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
      raise context.NoMatchException()

    i = code.idx - 1
    if len(op.operands) <= i:
      raise context.NoMatchException()
    v = op.operands[i]
    return v, mapping

  raise NotImplementedError(f"Executor for {type(code)} is not implemented")


def execute_match_dag(
    code: Sequence[inst.Instruction],
    root_op: mu.core.MlirOpBase,
):
  mapping = {generator.ROOT_VAR: root_op}
  for instruction in code:
    _, mapping = _execute_instruction(instruction, mapping)

  results = MatchResults()
  for key, value in mapping.items():
    if isinstance(key, str) and key.startswith(_RESULT_PREFIX):
      results[key[len(_RESULT_PREFIX) :]] = value
  return results
