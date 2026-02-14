# Copyright 2019 The Bazel Authors. All rights reserved.
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

"""Support for the xctoolrunner tool."""

def _prefixed_path(path):
    """Prefix paths with a token used in the xctoolrunner tool.

    Prefix paths with a token to indicate that certain arguments are paths, so they can be
    processed accordingly. This prefix must match the prefix used here in
    tools/xctoolrunner/xctoolrunner.py

    Args:
        path: Path to the resource to be prefixed.

    Returns:
        The path prefixed for xctoolrunner.
    """
    prefix = "[ABSOLUTE]"
    return prefix + path

# Define the loadable module that lists the exported symbols in this file.
xctoolrunner = struct(
    prefixed_path = _prefixed_path,
)
