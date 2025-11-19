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
"""Match predicate functions."""

from typing import Any, Callable, TypeVar, overload
from . import context


ANY = context.ANY


@overload
def match_pred(pred: Callable[[], bool] | bool):
  """Raises NoMatchException if the predicate is False or returns False."""
  ...


T = TypeVar('T')


@overload
def match_pred(value: T, pred: Callable[[T], bool]):
  """Matches a predicate against a value."""
  ...


def match_pred(value, pred: Callable[[Any], bool] | bool = None):
  if pred is None:
    pred = value
    if callable(pred):
      pred = pred()
  else:
    pred = pred(value)

  if not pred:
    raise context.NoMatchError()
