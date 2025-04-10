from xdsl.irdl import SSAValue

from litert.python.google.tools import model_utils as mu
from litert.python.google.tools.model_utils.matcher import context
from litert.python.google.tools.model_utils.matcher.match_dag import executor
from litert.python.google.tools.model_utils.matcher.match_dag import generator


def MatchDag(dag: str, op_or_value: mu.core.MlirOpBase | SSAValue):

  code = generator.parse_match_dag(dag)

  if not isinstance(op_or_value, SSAValue):
    op = op_or_value
    if not op.results:
      raise context.NoMatchException()
    value = op.results[0]
  else:
    value = op_or_value

  results = executor.execute_match_dag(code, value)
  return results
