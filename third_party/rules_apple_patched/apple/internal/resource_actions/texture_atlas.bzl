# Copyright 2018 The Bazel Authors. All rights reserved.
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

"""Texture atlas related actions."""

load(
    "@build_bazel_apple_support//lib:apple_support.bzl",
    "apple_support",
)

def compile_texture_atlas(
        *,
        actions,
        input_files,
        input_path,
        output_dir,
        platform_prerequisites):
    """Creates an action that compiles texture atlas bundles (i.e. .atlas).

    Args:
      actions: The actions provider from `ctx.actions`.
      input_files: The atlas file inputs that will be compiled.
      input_path: The path to the .atlas directory to compile.
      output_dir: The file reference for the compiled output directory.
      platform_prerequisites: Struct containing information on the platform being targeted.
    """
    apple_support.run(
        actions = actions,
        apple_fragment = platform_prerequisites.apple_fragment,
        arguments = [
            "TextureAtlas",
            input_path,
            output_dir.path,
        ],
        executable = "/usr/bin/xcrun",
        inputs = input_files,
        mnemonic = "CompileTextureAtlas",
        outputs = [output_dir],
        xcode_config = platform_prerequisites.xcode_version_config,
    )
