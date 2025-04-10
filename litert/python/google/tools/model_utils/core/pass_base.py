import abc
import dataclasses
import functools
from typing import Callable, Sequence
import jax
from xdsl import pattern_rewriter
from . import dialect_base

__all__ = ["ModulePassBase", "RewritePatternPassBase"]


@dataclasses.dataclass(kw_only=True)
class ModulePassBase(abc.ABC):
  name = None
  description = ""

  def call(self, op: dialect_base.MlirOpBase):
    raise NotImplementedError()

  def __call__(self, op: dialect_base.MlirOpBase):
    return self.call(op)


class RewritePatternPassBase(ModulePassBase, abc.ABC):

  def __init__(self, *args, **kwargs):
    self.patterns = []
    if hasattr(self, "_pattern_clss"):
      self.patterns = [cls() for cls in self._pattern_clss]

    super().__init__(*args, **kwargs)

  def call(self, module: dialect_base.MlirOpBase):
    walker = pattern_rewriter.PatternRewriteWalker(
        pattern_rewriter.GreedyRewritePatternApplier(self.patterns),
        walk_regions_first=True,
        apply_recursively=True,
        walk_reverse=False,
    )
    walker.rewrite_module(module)

  @classmethod
  def register_rewrite_pattern(
      cls,
      ops: (
          str
          | Sequence[str]
          | dialect_base.MlirOpBase
          | Sequence[dialect_base.MlirOpBase]
          | None
      ) = None,
  ):
    if not hasattr(cls, "_pattern_clss"):
      cls._pattern_clss = []

    if ops is None:
      target_ops = None
    else:
      target_ops, _ = jax.tree_util.tree_flatten(ops)
      target_ops = {op if isinstance(op, str) else op.name for op in target_ops}

    def reg(match_and_rewrite: Callable[..., None]):
      @functools.wraps(match_and_rewrite)
      def match_and_rewrite_(
          self,
          op: dialect_base.MlirOpBase,
          rewriter: pattern_rewriter.PatternRewriter,
      ):
        if target_ops is not None and op.name not in target_ops:
          return
        return match_and_rewrite(op, rewriter)

      pattern_cls = type(
          match_and_rewrite.__name__,
          (pattern_rewriter.RewritePattern,),
          {"match_and_rewrite": match_and_rewrite_},
      )
      cls._pattern_clss.append(pattern_cls)
      return match_and_rewrite

    return reg
