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

"""Workspace definition for QAIRT."""

load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")

def qairt_repo(path = ""):
    """
    Defines the qairt repository.

    Args:
      path: The path to the qairt repository. If empty, an empty repository is used.
    """
    if not path:
        # Set valid but empty repo for now.
        new_local_repository(
            name = "qairt",
            build_file_content = "package(default_visibility = [\"//visibility:public\"])\ncc_library(name = \"qnn_lib_headers\")",
            path = ".",
        )

        return

    # TODO: Extend the bazel repo rules to switch between hosted and local repos.
    new_local_repository(
        name = "qairt",
        build_file = "@//third_party/qairt:qairt.BUILD",
        path = path,
    )
