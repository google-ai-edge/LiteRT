"""Match DAG functions."""

from xdsl import irdl

from litert.python.tools import model_utils as mu
from litert.python.tools.model_utils.match import context
from litert.python.tools.model_utils.match._match_dag import executor
from litert.python.tools.model_utils.match._match_dag import generator

SSAValue = irdl.SSAValue


def match_dag(
    dag: str, op_or_value: mu.core.MlirOpBase | SSAValue
) -> executor.MatchResults:
  """Matches a Match DAG code against an op's result (SSAValue)."""

  code = generator.parse_match_dag(dag)

  if not isinstance(op_or_value, SSAValue):
    op = op_or_value
    if not op.results:
      raise context.NoMatchError()
    value = op.results[0]
  else:
    value = op_or_value

  results = executor.execute_match_dag(code, value)
  return results
