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

"""Support functions related to getting main thread checker libraries."""

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")

def _should_package_main_thread_checker_dylib(*, features):
    """Returns whether the libMainThreadChecker.dylib should be bundled."""

    return "apple.include_main_thread_checker" in features

def _get_from_toolchain(ctx):
    if hasattr(ctx.attr, "_cc_toolchain"):
        cc_toolchain = find_cpp_toolchain(ctx)
        dylibs = [
            x
            for x in cc_toolchain.all_files.to_list()
            if x.basename.startswith("libMainThreadChecker") and x.basename.endswith(".dylib")
        ]
    else:
        dylibs = []

    return dylibs

main_thread_checker_dylibs = struct(
    should_package_main_thread_checker_dylib = _should_package_main_thread_checker_dylib,
    get_from_toolchain = _get_from_toolchain,
)
