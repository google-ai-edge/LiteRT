from . import match_dag
from . import match_op
from . import match_pred
from .context import *
from .predicate import *


# Shortcuts
MatchOp = match_op.MatchOp
MatchDag = match_dag.MatchDag
MatchPred = match_pred.MatchPred

Dag = MatchDag
Op = MatchOp
Pred = MatchPred

Match = MatchPred
M = MatchPred
Fail = lambda: MatchPred(False)
