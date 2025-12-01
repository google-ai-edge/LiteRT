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
"""Workspace definition for Openvino."""

load("//litert/sdk_util:repo.bzl", "configurable_repo")

def openvino_configure():
    configurable_repo(
        name = "intel_openvino",
        build_file = "@//third_party/intel_openvino:openvino.bazel",
        local_path_env = "OPENVINO_NATIVE_DIR",
        url = "https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.3/windows/openvino_toolkit_windows_2025.3.0.19807.44526285f24_x86_64.zip",
        file_extension = "zip",
        symlink_mapping = {
            "openvino": "openvino_toolkit_windows_2025.3.0.19807.44526285f24_x86_64",
        },
    )
