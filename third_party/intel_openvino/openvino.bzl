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
    """Configure OpenVINO for multiple platforms."""

    # On Linux hosts, both openvino/ (Linux SDK) and openvino_android/ (Android SDK)
    # are downloaded. Bazel's select() picks the correct one at build time based on
    # target platform, enabling Android cross-compilation from Linux.
    # On Windows hosts, only the Windows SDK is downloaded.
    configurable_repo(
        name = "intel_openvino",
        build_file = "@//third_party/intel_openvino:openvino.bazel",
        local_path_env = "OPENVINO_NATIVE_DIR",
        packages = json.encode([
            {
                "url": "https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/2026.1.0-20996-3ef8425ae12/openvino_toolkit_windows_2026.1.0.dev20260131_x86_64.zip",
                "host_os": "windows",
                "file_extension": "zip",
                "symlink_mapping": {
                    "openvino": "openvino_toolkit_windows_2026.1.0.dev20260131_x86_64",
                },
            },
            {
                "url": "https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/2026.1.0-20996-3ef8425ae12/openvino_toolkit_ubuntu22_2026.1.0.dev20260131_x86_64.tgz",
                "host_os": "linux",
                "file_extension": "tgz",
                "symlink_mapping": {
                    "openvino": "openvino_toolkit_ubuntu22_2026.1.0.dev20260131_x86_64",
                },
            },
            {
                "url": "https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/2026.1.0-20996-3ef8425ae12/openvino_toolkit_android_2026.1.0.dev20260131_x86_64.tgz",
                "host_os": "linux",
                "file_extension": "tgz",
                "symlink_mapping": {
                    "openvino_android": "openvino_toolkit_android_2026.1.0.dev20260131_x86_64",
                },
            },
        ]),
    )
