"""Context for pattern matching."""

import contextlib
import dataclasses


ANY = None


class NoMatchError(RuntimeError):
  """Exception raised when a pattern match fails."""

  def __init__(self):
    super().__init__(
        "NoMatchError is not caught: is the matching run in a MatchingContext?"
    )


@dataclasses.dataclass
class MatchingContext:
  """A context manager for matching operations.

  This class provides a context where you can perform matching operations
  and track whether any NoMatchError occurred, and early exit the matching
  process if so.
  """

  failed: bool = False

  def __enter__(self):
    """Enters the context for matching operations."""
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Exits the context and handles any NoMatchError."""
    if exc_type is NoMatchError:
      self.failed = True
      return True  # Suppress the exception

  @property
  @contextlib.contextmanager
  def ctx(self):
    with self:
      yield self

  @property
  def matched(self):
    """Returns True if no NoMatchError occurred within the context."""
    return not self.failed
