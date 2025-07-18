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

def _openvino_native_impl(repository_ctx):
    openvino_native_dir = repository_ctx.os.environ.get("OPENVINO_NATIVE_DIR")
    if openvino_native_dir:
        repository_ctx.symlink(openvino_native_dir, "openvino")
        build_file_content = repository_ctx.read(repository_ctx.attr.build_file)
        repository_ctx.file("BUILD", build_file_content)
    else:
        # Variable not set, create an empty BUILD file
        repository_ctx.file("BUILD", "# OPENVINO_NATIVE_DIR not set, skipping OpenVINO setup.")

openvino_configure = repository_rule(
    implementation = _openvino_native_impl,
    local = True,
    environ = ["OPENVINO_NATIVE_DIR"],
    attrs = {
        # Define an attribute to hold the label of the external BUILD file content
        "build_file": attr.label(
            doc = "The label of the BUILD file content to be written.",
            allow_single_file = True,  # This attribute expects a single file
            mandatory = True,
        ),
    },
)
