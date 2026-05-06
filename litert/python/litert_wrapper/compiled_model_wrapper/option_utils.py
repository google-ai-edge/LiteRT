# Copyright 2026 Google LLC.
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

"""Internal helpers for LiteRT compiled-model options."""

import enum
from typing import Optional


def optional_bool_to_int(value: Optional[bool]) -> int:
  """Returns the pybind sentinel representation for an optional bool."""
  if value is None:
    return -1
  return 1 if value else 0


def optional_enum_to_int(value: Optional[enum.IntEnum]) -> int:
  """Returns the pybind sentinel representation for an optional enum."""
  if value is None:
    return -1
  return int(value)
