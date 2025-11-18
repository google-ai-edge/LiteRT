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
