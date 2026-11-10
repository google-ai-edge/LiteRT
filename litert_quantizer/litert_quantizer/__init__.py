# Copyright 2024 Google LLC
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
# ==============================================================================

"""Init file for the AI Edge quantizer package."""

import sys
import warnings

# pylint: disable=unused-import
from .quantizer import Quantizer

# Python 3.10 reached upstream end-of-life on 2026-10-01 (PEP 619). Support
# for it is deprecated and will be removed in the next release. FutureWarning
# (rather than DeprecationWarning) so end users see it even when this package
# is imported transitively (PEP 565 hides DeprecationWarning outside __main__).
if sys.version_info < (3, 11):
  warnings.warn(
      "litert-quantizer support for Python 3.10 is deprecated and will be"
      " removed in the next release, following Python 3.10's upstream"
      " end-of-life (https://peps.python.org/pep-0619/). Please upgrade to"
      " Python 3.11 or newer.",
      FutureWarning,
      stacklevel=2,
  )
