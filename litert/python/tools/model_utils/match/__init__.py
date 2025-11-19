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
"""Pattern matching library."""

from . import _match_dag
from . import _match_op
from . import _match_pred
from . import context
from . import predicate
from .context import *
from .predicate import *


# Shortcuts
match_op = _match_op.match_op
match_dag = _match_dag.match_dag
match_pred = _match_pred.match_pred

dag = match_dag
op = match_op
pred = match_pred
match = match_pred
ANY = context.ANY


def fail():
  """Raises NoMatchError."""
  raise context.NoMatchError()
