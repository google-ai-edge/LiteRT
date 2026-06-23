# Copyright 2025 The ODML Authors.
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_rust//rust:defs.bzl", "rust_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_hdrs_check = "strict",
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "convert_tensor_buffer",
    hdrs = ["convert_tensor_buffer.h"],
    deps = [
        "//tflite/types:half",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:span",
    ] + select({
        "//litert:litert_link_capi_so": [
            "//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "//litert/cc:litert_common",
            "//litert/cc:litert_element_type",
            "//litert/cc:litert_environment",
            "//litert/cc:litert_expected",
            "//litert/cc:litert_layout",
            "//litert/cc:litert_macros",
            "//litert/cc:litert_ranked_tensor_type",
            "//litert/cc:litert_tensor_buffer",
            "//litert/cc:litert_tensor_buffer_types",
        ],
    }),
)

cc_test(
    name = "convert_tensor_buffer_test",
    srcs = ["convert_tensor_buffer_test.cc"],
    tags = ["requires-mac-inputs:hard"],
    deps = [
        ":convert_tensor_buffer",
        "//litert/cc:litert_common",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_tensor_buffer",
        "//litert/cc:litert_tensor_buffer_types",
        "//litert/test:matchers",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_status_util",
    hdrs = ["status_macros.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        # placeholder for task_status
        # placeholder for ret_check,
    ],
)

cc_test(
    name = "status_macros_test",
    srcs = ["status_macros_test.cc"],
    deps = [
        ":litert_status_util",
        ":test_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "scoped_file",
    hdrs = ["scoped_file.h"],
    deps = [
        "//litert/cc/internal:scoped_file",
    ],
)

cc_library(
    name = "memory_mapped_file",
    srcs = select({
        "@platforms//os:windows": ["memory_mapped_file_win.cc"],
        "//conditions:default": ["memory_mapped_file_posix.cc"],
    }),
    hdrs = ["memory_mapped_file.h"],
    copts = [
        "-D_WIN32_WINNT=_WIN32_WINNT_WIN10",
    ],
    deps = [
        ":litert_status_util",
        ":scoped_file",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "memory_mapped_file_test",
    srcs = ["memory_mapped_file_test.cc"],
    deps = [
        ":memory_mapped_file",
        ":scoped_file",
        ":test_utils",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_utils",
    testonly = 1,
    hdrs = ["test_utils.h"],
    deps = [
        "//litert/cc:litert_macros",
        "//testing/base/public:gunit_for_library_testonly",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "io_types",
    srcs = ["io_types.cc"],
    hdrs = ["io_types.h"],
    deps = [
        ":litert_status_util",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_tensor_buffer",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "io_types_test",
    srcs = ["io_types_test.cc"],
    deps = [
        ":convert_tensor_buffer",
        ":io_types",
        ":test_utils",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_model",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc:litert_tensor_buffer",
        "//litert/cc:litert_tensor_buffer_types",
        "//litert/test:matchers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

rust_library(
    name = "global_allocator",
    srcs = ["global_allocator.rs"],
)

cc_library(
    name = "alloc_defs",
    srcs = ["alloc_defs.cc"],
    alwayslink = True,
)
