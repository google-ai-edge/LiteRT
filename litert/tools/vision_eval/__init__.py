# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LiteRT vision evaluation dataset preparation and metric verification tools."""

import pathlib
import sys

for _entry in sys.path:
  _candidate = pathlib.Path(_entry, *__name__.split('.'))
  if _candidate.is_dir() and str(_candidate) not in __path__:
    __path__.append(str(_candidate))
