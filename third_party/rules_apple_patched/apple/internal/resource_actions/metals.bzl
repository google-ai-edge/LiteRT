# Copyright 2020 The Bazel Authors. All rights reserved.
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

"""Metal related actions."""

load(
    "@bazel_skylib//lib:paths.bzl",
    "paths",
)
load(
    "@build_bazel_apple_support//lib:apple_support.bzl",
    "apple_support",
)

def _metal_apple_target_triple(platform_prerequisites):
    """Returns a Metal target triple string for an Apple platform.

    Args:
        platform_prerequisites: The target's platform_prerequisites.
    Returns:
        A target triple string describing the platform.
    """
    target_os_version = platform_prerequisites.minimum_os

    platform = platform_prerequisites.apple_fragment.single_arch_platform
    platform_string = str(platform.platform_type)
    if platform_string == "macos":
        platform_string = "macosx"

    environment = "" if platform.is_device else "-simulator"

    return "air64-apple-{platform}{version}{environment}".format(
        environment = environment,
        platform = platform_string,
        version = target_os_version,
    )

def compile_metals(*, actions, input_files, output_file, platform_prerequisites, **_kwargs):
    """Creates actions that compile .metal files into a single .metallib file.

    Args:
        actions: The actions provider from `ctx.actions`.
        platform_prerequisites: Struct containing information on the platform being targeted.
        input_files: The input metal files.
        output_file: The output metallib file.
        **_kwargs: Ignored
    """
    air_files = []
    target = _metal_apple_target_triple(platform_prerequisites)

    if not input_files:
        fail("Input .metal files can't be empty")

    hdrs = []
    metal_files = []
    for file in input_files:
        if file.extension == "metal":
            metal_files.append(file)
        elif file.extension == "h":
            hdrs.append(file)
        else:
            fail("Unhandled filetype: {}".format(file.extension))

    # Compile each .metal file into a single .air file
    for input_metal in metal_files:
        air_file = actions.declare_file(
            paths.replace_extension(input_metal.basename, ".air"),
        )
        air_files.append(air_file)

        args = actions.args()
        args.add("metal")
        args.add("-c")
        args.add("-target", target)
        args.add("-ffast-math")
        args.add("-o", air_file)
        args.add(input_metal)

        apple_support.run(
            actions = actions,
            executable = "/usr/bin/xcrun",
            inputs = [input_metal] + hdrs,
            outputs = [air_file],
            arguments = [args],
            mnemonic = "MetalCompile",
            apple_fragment = platform_prerequisites.apple_fragment,
            xcode_config = platform_prerequisites.xcode_version_config,
        )

    # Compile .air files into a single .metallib file, which stores the Metal
    # library
    args = actions.args()
    args.add("metallib")
    args.add("-o", output_file)
    args.add_all(air_files)

    apple_support.run(
        actions = actions,
        executable = "/usr/bin/xcrun",
        inputs = air_files,
        outputs = [output_file],
        arguments = [args],
        mnemonic = "MetallibCompile",
        apple_fragment = platform_prerequisites.apple_fragment,
        xcode_config = platform_prerequisites.xcode_version_config,
    )
