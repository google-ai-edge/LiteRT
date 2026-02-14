# Copyright 2023 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Support for common environment variables affecting rules_apple Bazel scripts."""

visibility([
    "//tools/...",
])

def _common_python_utf8_env():
    # Presents a set of environment variables to force UTF-8 encoding in Python tools.
    #
    # This particular solution avoids issues where the host platform has a different default
    # encoding set in its configured environment, as observed in b/270717116, which can happen on a
    # user's Mac if they have supplied their own Python or within the default shell environment on
    # the OS itself if it is configured for a different set of locale/encoding settings.
    return {
        "PYTHONIOENCODING": "UTF-8",  # Always encode stdout/stderr as UTF-8.
        "PYTHONUTF8": "1",  # Don't disable UTF-8 defaults per https://peps.python.org/pep-0686/.
    }

binary_env = struct(
    common_python_utf8_env = _common_python_utf8_env,
)
