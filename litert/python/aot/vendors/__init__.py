# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Vendor backends for LiteRt."""
import os

from litert.python.aot.vendors.mediatek import mediatek_backend as _
from litert.python.aot.vendors.qualcomm import qualcomm_backend as _

if any(
    key in os.environ
    for key in ("GOOGLE_TENSOR_COMPILER_LIB", "GOOGLE_TENSOR_BACKEND_ENABLED")
):
  from litert.python.aot.vendors.google_tensor import google_tensor_backend as _  # pylint: disable=g-import-not-at-top,g-bad-import-order
