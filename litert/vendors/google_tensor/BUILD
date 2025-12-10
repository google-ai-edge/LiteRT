load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

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
load("//litert/build_common:litert_build_defs.bzl", "litert_lib", "make_rpaths")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

litert_lib(
    name = "adapter_aot",
    srcs = [
        "adapter_aot.cc",
        "adapter_aot.h",
    ],
    hdrs = ["adapter.h"],
    data = [
        # copybara:uncomment_begin(google-only)
        # "//platforms/darwinn/compiler:compiler_api_wrapper",
        # copybara:uncomment_end
    ],
    defines = ["LITERT_GOOGLE_TENSOR_AOT=1"],
    linkopts = [
        # TODO(abhirs): Make this work for OS.
        #copybara:comment_begin(google-only)
        make_rpaths(["platforms/darwinn/compiler"]),
        # copybara:uncomment_end
    ],
    tags = [
        # Don't build/test in OS until libcompiler_api_wrapper.so is available.
        "nobuilder",
        "no_oss",
    ],
    ungrte = False,
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/core:dynamic_loading",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

proto_library(
    name = "edgetpu_compiler_options_proto",
    srcs = ["edgetpu_compiler_options.proto"],
)

cc_proto_library(
    name = "edgetpu_compiler_options_cc_proto",
    visibility = [
        # copybara:uncomment_begin(google-only)
        # "//third_party/darwinn/litert:__subpackages__",
        # copybara:uncomment_end
    ],
    deps = [
        ":edgetpu_compiler_options_proto",
    ],
)

cc_library(
    name = "edgetpu_compiler_api",
    hdrs = ["edgetpu_compiler_api.h"],
    visibility = [
        # copybara:uncomment_begin(google-only)
        # "//third_party/darwinn/litert:__subpackages__",
        # copybara:uncomment_end
    ],
)

litert_lib(
    name = "adapter_odc",
    srcs = [
        "adapter_odc.cc",
        "adapter_odc.h",
    ],
    hdrs = ["adapter.h"],
    defines = ["LITERT_GOOGLE_TENSOR_ODC=1"],
    deps = [
        ":edgetpu_compiler_api",
        ":edgetpu_compiler_options_cc_proto",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_shared_library",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/strings",
    ],
)

litert_lib(
    name = "adapter",
    hdrs = ["adapter.h"],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
    ] + select({
        "@org_tensorflow//tensorflow:android": [":adapter_odc"],
        "//conditions:default": [":adapter_aot"],
    }),
)

cc_test(
    name = "adapter_test",
    srcs = ["adapter_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    linkstatic = 1,
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        #Don't build/test in OS until libcompiler_api_wrapper.so is available.
        "nobuilder",
        "no_oss",
        # Sanitizer runtime doesn't work with anything that uses dlopen.
        "nosan",
        "manual",
    ],
    # This test can only be run on Android and Linux.
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":adapter",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc/dynamic_runtime:litert_model",
        "//litert/cc/dynamic_runtime/options:litert_google_tensor_options",
        "//litert/test:common",
        "//litert/test:matchers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
