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
