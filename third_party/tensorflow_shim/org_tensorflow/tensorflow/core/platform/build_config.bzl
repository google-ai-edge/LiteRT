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

"""Minimal stand-in for @org_tensorflow//tensorflow/core/platform:build_config.bzl."""

load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@rules_python//python:py_library.bzl", "py_library")

_WELL_KNOWN_PROTOS = [
    "@com_google_protobuf//:any_proto",
    "@com_google_protobuf//:duration_proto",
    "@com_google_protobuf//:empty_proto",
    "@com_google_protobuf//:struct_proto",
    "@com_google_protobuf//:timestamp_proto",
    "@com_google_protobuf//:wrappers_proto",
]

def tf_platform_alias(name, platform_dir = "@xla//xla/tsl/platform/"):
    return [platform_dir + "default:" + name]

def tf_proto_library(
        name,
        srcs = [],
        deps = [],
        protodeps = [],
        exports = [],
        testonly = 0,
        visibility = None,
        compatible_with = None,
        tags = [],
        **kwargs):
    """Subset of TSL's tf_proto_library: proto, C++ and Python targets only.

    Obsolete TSL arguments (make_default_target_header_only, cc_libs, js_codegen,
    create_*, ...) are accepted and ignored. gRPC services are not supported.

    Args:
      name: Name of the proto_library. The C++ and Python targets get the
        `_cc` and `_py` suffixes.
      srcs: .proto files.
      deps: proto_library dependencies.
      protodeps: More proto_library dependencies, merged with `deps`.
      exports: Passed to proto_library.
      testonly: Passed to all targets.
      visibility: Passed to all targets.
      compatible_with: Passed to all targets.
      tags: Passed to proto_library.
      **kwargs: Passed to proto_library, without the obsolete TSL arguments.
    """
    for key in list(kwargs.keys()):
        if key.startswith("create_") or key in [
            "cc_grpc_version",
            "cc_libs",
            "has_services",
            "j2objc_api_version",
            "js_codegen",
            "local_defines",
            "make_default_target_header_only",
            "use_grpc_namespace",
        ]:
            kwargs.pop(key)

    all_deps = deps + protodeps
    proto_library(
        name = name,
        srcs = srcs,
        deps = all_deps + [p for p in _WELL_KNOWN_PROTOS if p not in all_deps],
        exports = exports,
        compatible_with = compatible_with,
        visibility = visibility,
        testonly = testonly,
        tags = tags,
        **kwargs
    )

    cc_proto_library(
        name = name + "_cc",
        testonly = testonly,
        compatible_with = compatible_with,
        visibility = visibility,
        deps = [":" + name],
    )
    for suffix in ["_cc_impl", "_cc_headers_only"]:
        native.alias(
            name = name + suffix,
            testonly = testonly,
            actual = ":" + name + "_cc",
            compatible_with = compatible_with,
            visibility = visibility,
        )

    py_proto_library(
        name = name + "_py_proto",
        testonly = testonly,
        compatible_with = compatible_with,
        visibility = visibility,
        deps = [":" + name],
    )
    py_library(
        name = name + "_py",
        testonly = testonly,
        compatible_with = compatible_with,
        visibility = visibility,
        deps = [":" + name + "_py_proto"],
    )
