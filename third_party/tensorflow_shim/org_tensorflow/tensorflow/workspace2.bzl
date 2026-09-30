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

"""No-op stand-in for @org_tensorflow//tensorflow:workspace2.bzl.

The WORKSPACE calls `tf_workspace2()` in both modes. It only does work when
building with the real TensorFlow (LITERT_WITH_TENSORFLOW=1).
"""

def tf_workspace2():
    """Does nothing. The shim does not need TensorFlow's dependencies."""
