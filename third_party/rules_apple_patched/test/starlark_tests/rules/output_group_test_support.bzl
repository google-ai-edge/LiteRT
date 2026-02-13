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

"""Shared support methods for testing output groups."""

visibility("//test/starlark_tests/...")

def _output_group_file_from_target(
        *,
        output_group_name,
        output_group_file_shortpath,
        providing_target):
    """Retrieves a specific File from the target's OutputGroupInfo provider.

    Args:
        output_group_name: String. The name of the output group the File is referenced in.
        output_group_file_shortpath: String. A shortpath indicating where the File is expected to be
            found in the filesystem.
        providing_target: Label. A target that has the requested output group.
    """
    if not OutputGroupInfo in providing_target:
        fail(("Target %s does not provide OutputGroupInfo") % providing_target.label)

    output_group = getattr(providing_target[OutputGroupInfo], output_group_name, None)
    if not output_group:
        fail("OutputGroupInfo does not have %s" % output_group_name)

    output_group_files = output_group.to_list()
    output_group_file = ""
    for found_output_group_file in output_group_files:
        if found_output_group_file.short_path == output_group_file_shortpath:
            output_group_file = found_output_group_file
    if not output_group_file:
        fail("{output_group_file_shortpath} not found; instead found {output_group_files}".format(
            output_group_file_shortpath = output_group_file_shortpath,
            output_group_files = output_group_files,
        ))
    return output_group_file

output_group_test_support = struct(
    output_group_file_from_target = _output_group_file_from_target,
)
