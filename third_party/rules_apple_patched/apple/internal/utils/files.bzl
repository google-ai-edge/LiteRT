# Copyright 2022 The Bazel Authors. All rights reserved.
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

"""Support methods to work with Bazel files."""

load("@bazel_skylib//lib:paths.bzl", "paths")

def _get_file_with_name(*, files, name):
    """Traverse a given file list and return file matching given name (without extension).

    Args:
        files: List of files to traverse.
        name: File name (w/o extension) to match.
    Returns:
        File matching name, None otherwise.
    """
    for file in files:
        file_name, _ = paths.split_extension(file.basename)
        if name and file_name == name:
            return file

    return None

files = struct(
    get_file_with_name = _get_file_with_name,
)
