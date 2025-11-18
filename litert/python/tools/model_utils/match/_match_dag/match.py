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

  code = generator.generate_match_dag_instructions(dag)

  if not isinstance(op_or_value, SSAValue):
    op = op_or_value
    if not op.results:
      raise context.NoMatchError()
    value = op.results[0]
  else:
    value = op_or_value

  results = executor.execute_match_dag(code, value)
  return results
