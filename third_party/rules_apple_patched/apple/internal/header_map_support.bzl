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

"""Provides utilities for working with header_maps within macros and rules"""

load("//apple/internal:header_map.bzl", "header_map")

def _create_header_map_context(
        name,
        module_name,
        hdrs,
        deps,
        testonly = False):
    """
    Creates header_map(s) for the target with the given name.

    Args:
        name: The name of the target.
        module_name: The name of the module to use in the header map.
        hdrs: The public headers to include in the header mapm, these will be rooted
            under module_name and flattened.
        deps: The direct dependencies to include in the header map, any headers in the deps will be rooted
            under this `module_name` and flattened.
        testonly: Whether or not this is a header map being used for a test only target.

    Returns a struct (or `None` if no headers) with the following attributes:
        copts: The compiler options to use when compiling a C family library with header maps.
        includes: The includes to use when compiling a C family library with header maps.
        swift_copts: The compiler options to use when compiling a Swift library with the header map.
        header_maps: The labels to any generated header maps, these should be the inputs used with the `copts`.
    """

    public_hdrs_filegroup = name + ".public.hmap_hdrs"
    public_hmap_name = name + ".public"

    native.filegroup(
        name = public_hdrs_filegroup,
        srcs = hdrs,
    )

    header_map(
        name = public_hmap_name,
        module_name = module_name,
        hdrs = [public_hdrs_filegroup],
        deps = deps,
        testonly = testonly,
    )

    copts = [
        "-I.",  # makes the package's bin dir available for use with #import <foo/foo.h>
    ]
    includes = [
        "{}.hmap".format(public_hmap_name),  # propogates the header map include to target and things that depend on it
    ]
    header_maps = [
        ":{}".format(public_hmap_name),
    ]

    swift_copts = []
    for copt in copts:
        swift_copts.extend(["-Xcc", copt])

    return struct(
        copts = copts,
        includes = includes,
        swift_copts = swift_copts,
        header_maps = header_maps,
    )

header_map_support = struct(
    create_header_map_context = _create_header_map_context,
)
