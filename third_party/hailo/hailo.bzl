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
"""Workspace definition for HailoRT SDK."""

def _hailo_sdk_impl(ctx):
    sdk_path = ctx.os.environ.get("HAILO_RT_DIR", "")
    if sdk_path:
        # Symlink the real SDK files
        host_os = ctx.os.name.lower()
        if "windows" not in host_os and not sdk_path.startswith("/"):
            fail("Local path must be absolute.")
        ctx.symlink(sdk_path + "/include", "include")
        ctx.symlink(sdk_path + "/lib", "lib")
        ctx.symlink(ctx.attr.build_file, "BUILD.bazel")
    else:
        # Use mock SDK!
        mock_build = ctx.path(Label("@//third_party/hailo/mock_sdk:BUILD"))
        mock_dir = mock_build.dirname
        ctx.symlink(mock_build, "BUILD.bazel")
        ctx.symlink(mock_dir.get_child("include"), "include")
        ctx.symlink(mock_dir.get_child("src"), "src")

_hailo_sdk_repo = repository_rule(
    implementation = _hailo_sdk_impl,
    environ = ["HAILO_RT_DIR"],
    attrs = {
        "build_file": attr.label(allow_single_file = True),
    },
)

def hailo_configure():
    """Configure HailoRT SDK for building the NPU backend."""
    _hailo_sdk_repo(
        name = "hailo_sdk",
        build_file = Label("@//third_party/hailo:hailo.bazel"),
    )
