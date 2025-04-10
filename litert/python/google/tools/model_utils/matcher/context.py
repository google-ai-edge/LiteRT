import contextlib
import dataclasses
from xdsl.ir.core import *
from xdsl.irdl import *


ANY = None


class NoMatchException(Exception):
  pass


@dataclasses.dataclass
class MatchingContext:
  """A context manager for matching operations.

  This class provides a context where you can perform matching operations
  and track whether any NoMatchException occurred.
  """

  failed: bool = False

  def __enter__(self):
    """Enters the context for matching operations."""
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Exits the context and handles any NoMatchException."""
    if exc_type is NoMatchException:
      self.failed = True
      return True  # Suppress the exception

  @property
  @contextlib.contextmanager
  def ctx(self):
    with self:
      yield self

  @property
  def matched(self):
    """Returns True if no NoMatchException occurred within the context."""
    return not self.failed
