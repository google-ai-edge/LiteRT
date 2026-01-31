# Copyright 2024 Google LLC.
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

load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib", "make_rpaths")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:private"],
)

litert_dynamic_lib(
    name = "compiler_plugin",
    srcs = ["compiler_plugin.cc"],
    hdrs = ["//litert/vendors/c:litert_compiler_plugin.h"],
    copts = [
        "-Os",
        "-fno-exceptions",
        "-fno-unwind-tables",
        "-fno-asynchronous-unwind-tables",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    export_litert_only = True,
    linkstatic = 1,
    shared_lib_name = "google_tensor_compiler_plugin_so",
    so_name = "libLiteRtCompilerPlugin_google_tensor.so",
    tags = [
        # Don't build/test in OS until google tensor is available.
        "nobuilder",
        "no_oss",
        "notap",
    ],
    target_compatible_with = select({
        "@platforms//os:linux": [],
        "@platforms//os:android": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    ungrte = True,
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
        "//platforms/darwinn/tests/sdk_3p/litert_tools:__subpackages__",
        "//third_party/darwinn/litert:__subpackages__",
    ],
    deps = [
        ":google_tensor_options_cc_proto",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime/options:litert_google_tensor_options",
        "//litert/vendors/cc:options_helper",
        "//litert/vendors/google_tensor:adapter",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "compiler_plugin_test",
    srcs = [
        "compiler_plugin_test.cc",
    ],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": [],
        "//conditions:default": [
            make_rpaths([
                "//litert/c:litert_runtime_c_api_so",
            ]),
        ],
    }),
    linkstatic = True,
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        # Don't build/test in OS until google tensor is available.
        "nobuilder",
        "no_oss",
        # Sanatizer runtime doesn't work with anything that loads a shared library.
        "nosan",
        "manual",
    ],
    # This test can only be run on Android and Linux.
    target_compatible_with = select({
        "@platforms//os:linux": [],
        "@platforms//os:android": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":compiler_plugin",  # buildcleaner: keep
        "//litert/c/options:litert_google_tensor_options",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/vendors/cc:litert_compiler_plugin",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

proto_library(
    name = "google_tensor_options_proto",
    srcs = ["google_tensor_options.proto"],
    visibility = ["//platforms/darwinn/compiler:__subpackages__"],
)

cc_proto_library(
    name = "google_tensor_options_cc_proto",
    visibility = ["//platforms/darwinn/compiler:__subpackages__"],
    deps = [":google_tensor_options_proto"],
)
