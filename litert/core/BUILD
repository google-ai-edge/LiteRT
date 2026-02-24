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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//litert/build_common:litert_build_defs.bzl", "cc_library_with_testonly_vis")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_internal_users",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
)

cc_library(
    name = "options",
    hdrs = [
        "options.h",
        "//litert/c:litert_custom_op_kernel.h",
        "//litert/c:litert_options.h",
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_layout",
        "//litert/cc/internal:scoped_weight_source",
    ],
)

cc_library(
    name = "build_stamp",
    srcs = ["build_stamp.cc"],
    hdrs = ["build_stamp.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "build_stamp_test",
    srcs = ["build_stamp_test.cc"],
    data = [
        "//litert/test:tflite_test_data",
    ],
    deps = [
        ":build_stamp",
        "//litert/c:litert_common",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "dynamic_loading",
    srcs = select({
        "@org_tensorflow//tensorflow:windows": ["dynamic_loading_windows.cc"],
        "//conditions:default": ["dynamic_loading.cc"],
    }),
    hdrs = ["dynamic_loading.h"],
    linkopts = select({
        "@org_tensorflow//tensorflow:windows": [],
        "//conditions:default": ["-ldl"],
    }),
    # TODO: factor out the logic here that is used by vendors and move to cc/shared_lib utilities.
    # Then, we can remove this visibility.
    visibility = ["//litert:litert_public"],
    deps = [
        "//litert/c:litert_common",
        "@com_google_absl//absl/strings:string_view",
    ] + select({
        "@org_tensorflow//tensorflow:windows": [
            ":filesystem",
            "//litert/c/internal:litert_logging",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/strings:str_format",
        ],
        "//conditions:default": [
            ":filesystem",
            "//litert/c/internal:litert_logging",  # buildcleaner: keep
            "//litert/cc:litert_macros",
            "@com_google_absl//absl/strings",
        ],
    }),
)

cc_library(
    name = "insert_order_map",
    hdrs = ["insert_order_map.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_test(
    name = "insert_order_map_test",
    srcs = ["insert_order_map_test.cc"],
    deps = [
        ":insert_order_map",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "environment",
    srcs = ["environment.cc"],
    hdrs = [
        "environment.h",
        "//litert/c:litert_environment.h",
    ],
    deps = [
        ":environment_options",
        ":error_reporter",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options_header",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/runtime:accelerator_registry",
        "//litert/runtime:gpu_environment",
        "//litert/runtime:tensor_buffer_registry_header",
        "//tflite/core/api:error_reporter",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "environment_options",
    srcs = ["environment_options.cc"],
    hdrs = ["environment_options.h"],
    deps = [
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_custom_tensor_buffer",
        "//litert/c:litert_environment_options_header",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_expected",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "environment_options_test",
    srcs = ["environment_options_test.cc"],
    deps = [
        ":environment_options",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options",
        "//litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "environment_test",
    srcs = ["environment_test.cc"],
    deps = [
        ":environment",
        "//litert/c:litert_any",
        "//litert/cc:litert_any",
        "//litert/runtime:gpu_environment",
        "//litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library_with_testonly_vis(
    name = "filesystem",
    srcs = ["filesystem.cc"],
    hdrs = ["filesystem.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "dispatch_op_schema",
    srcs = ["dispatch_op_schema.cc"],
    hdrs = ["dispatch_op_schema.h"],
    copts = ["-DFLATBUFFERS_LOCALE_INDEPENDENT=0"],
    deps = [
        "//litert/cc:litert_buffer_ref",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "filesystem_test",
    srcs = ["filesystem_test.cc"],
    deps = [
        ":filesystem",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

# copybara:uncomment_begin(no OSS for unique-test-directory)
# cc_test(
#     name = "dynamic_loading_test",
#     srcs = ["dynamic_loading_test.cc"],
#     tags = [
#         # Sanitizer runtimes are incompatible with RTLD_DEEPBIND.
#         "noasan",
#         "nomsan",
#         "nosan",
#     ],
#     deps = [
#         ":dynamic_loading",
#         ":filesystem",
#         "@com_google_googletest//:gtest_main",
#         "@com_google_absl//absl/strings:string_view",
#         "//litert/c/internal:litert_logging",  # buildcleaner: keep
#         "//litert/test:common",
#         "//litert/test:matchers",
#     ],
# )
# copybara:uncomment_end

cc_test(
    name = "dispatch_op_schema_test",
    srcs = ["dispatch_op_schema_test.cc"],
    deps = [
        ":dispatch_op_schema",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "version",
    hdrs = ["version.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
    ],
)

cc_library(
    name = "error_reporter",
    srcs = ["error_reporter.cc"],
    hdrs = ["error_reporter.h"],
    deps = [
        ":buffer_error_reporter",
        "//litert/c/internal:litert_logging",
        "//tflite/core/api:error_reporter",
    ],
)

cc_library(
    name = "buffer_error_reporter",
    srcs = ["buffer_error_reporter.cc"],
    hdrs = ["buffer_error_reporter.h"],
    deps = [
        "//tflite/core/api:error_reporter",
    ],
)

cc_test(
    name = "error_reporter_test",
    srcs = ["error_reporter_test.cc"],
    deps = [
        ":buffer_error_reporter",
        ":error_reporter",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_toml_parser",
    srcs = ["litert_toml_parser.cc"],
    hdrs = ["litert_toml_parser.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_toml_parser_test",
    srcs = ["litert_toml_parser_test.cc"],
    deps = [
        ":litert_toml_parser",
        "//litert/c:litert_common",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@tomlplusplus",
    ],
)
