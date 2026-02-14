# Copyright 2024 The Bazel Authors. All rights reserved.
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

"""Partial implementation for gathering cc_info dylibs."""

load("@bazel_skylib//lib:partial.bzl", "partial")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load(
    "//apple/internal:processor.bzl",
    "processor",
)

def _cc_info_dylibs_partial_impl(
        *,
        embedded_targets):
    """Implementation for the CcInfo dylibs processing partial."""
    bundle_files = []

    for target in embedded_targets:
        if CcInfo not in target:
            continue
        cc_info = target[CcInfo]
        for linker_input in cc_info.linking_context.linker_inputs.to_list():
            for library in linker_input.libraries:
                if library.dynamic_library and library.dynamic_library.extension == "dylib":
                    bundle_files.append(
                        (processor.location.framework, None, depset([library.dynamic_library])),
                    )

    return struct(bundle_files = bundle_files)

def cc_info_dylibs_partial(
        *,
        embedded_targets):
    """Constructor for the CcInfo dylibs processing partial.

    This partial searches through the embedded_targets for dynamic_libraries and adds them
    as bundle_files destined for the Framework folder. The cc_* targets (like cc_library) can
    depend on dynamic libraries (.dylibs), usually pulled in from a cc_import. These .dylibs are required
    at runtime and are similar to .dlls on windows. The macos_application adds the /Contents/Frameworks
    folder as an @rpath search path so placing all of the dylibs in this folder seems reasonable.
    Generally, .dylibs should have their install name set to @rpath/libname.dylib for this strategy to work.
    You can check the install name using
        otool -L libname.dylib
    and change it using
        install_name_tool -id @rpath/libname.dylib libname.dylib

    Args:
        embedded_targets: The list of targets that may have CcInfo specifying dylibs that need to be bundled.

    Returns:
      A partial that returns the bundle location of all dylibs contained in the embedded_targets CcInfo, if there were any to
      bundle.
    """
    return partial.make(
        _cc_info_dylibs_partial_impl,
        embedded_targets = embedded_targets,
    )
