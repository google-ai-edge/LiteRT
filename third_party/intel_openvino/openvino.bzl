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
load(
    "//third_party/intel_openvino:openvino_version.bzl",
    "OPENVINO_DIRS",
    "OPENVINO_URLS",
)

def openvino_configure():
    """Configure OpenVINO for multiple platforms."""

    # On Linux hosts, both openvino/ (Linux SDK) and openvino_android/ (Android SDK)
    # are downloaded. Bazel's select() picks the correct one at build time based on
    # target platform, enabling Android cross-compilation from Linux.
    # On Windows hosts, only the Windows SDK is downloaded.
    #
    # The OpenVINO build pinned here comes from openvino_version.bzl, the
    # single source of truth shared with the pip SDK built by
    # ci/tools/python/vendor_sdk/intel/setup.py — this keeps the Intel OV
    # compiler plugin (built against the SDK below) paired with a matching
    # libopenvino_intel_npu_compiler at runtime. To bump the build or switch
    # channel, run ci/tools/update_openvino_version.py; do not hand-edit
    # openvino_version.bzl.
    configurable_repo(
        name = "intel_openvino",
        build_file = Label("@//third_party/intel_openvino:openvino.bazel"),
        local_path_env = "OPENVINO_NATIVE_DIR",
        packages = json.encode([
            {
                "url": OPENVINO_URLS["windows"],
                "host_os": "windows",
                "file_extension": "zip",
                "symlink_mapping": {
                    "openvino": OPENVINO_DIRS["windows"],
                },
            },
            {
                "url": OPENVINO_URLS["ubuntu24"],
                "host_os": "linux",
                "file_extension": "tgz",
                "symlink_mapping": {
                    "openvino": OPENVINO_DIRS["ubuntu24"],
                },
            },
            {
                "url": OPENVINO_URLS["android"],
                "host_os": "linux",
                "file_extension": "tgz",
                "symlink_mapping": {
                    "openvino_android": OPENVINO_DIRS["android"],
                },
            },
        ]),
    )
