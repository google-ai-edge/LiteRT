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
load("//litert/build_common:special_rule.bzl", "litert_metal_linkopts", "litert_metal_opts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

cc_library(
    name = "compiler_plugin",
    srcs = ["compiler_plugin.cc"],
    hdrs = ["compiler_plugin.h"],
    copts = litert_metal_opts(),
    linkopts = litert_metal_linkopts(),
    deps = [
        ":algo",
        ":litert_compiler_options",
        "//litert/c:litert_any",
        "//litert/c:litert_builder",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options_header",
        "//litert/c:litert_opaque_options",
        "//litert/c:litert_options",
        "//litert/c/internal:litert_logging",
        "//litert/c/options:litert_compiler_options",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_op_options",
        "//litert/cc/internal:litert_shared_library",
        "//litert/cc/options:litert_compiler_options",
        "//litert/core:build_stamp",
        "//litert/core:dynamic_loading",
        "//litert/core:environment",
        "//litert/core:filesystem",
        "//litert/core:options",
        "//litert/core:version",
        "//litert/core/model",
        "//litert/core/util:perfetto_profiling",
        "//litert/vendors/c:litert_compiler_plugin",
        "//litert/vendors/c:litert_compiler_plugin_api",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "compiler_plugin_test",
    srcs = ["compiler_plugin_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
        "//litert/vendors/examples:example_plugin_so",
    ],
    # Workaround to disable mismatched-size-delete.
    # A compiler plugin uses `libLiteRt.so` but `compiler_plugin` is a static library for runtime.
    # So mixing of the two will cause mismatched-size-delete error.
    # copybara:uncomment malloc = "//base:system_malloc",
    tags = [
        # Sanitizer runtimes are incompatible with RTLD_DEEPBIND.
        "noasan",
        "nomsan",
        "nosan",
        "notsan",
    ],
    deps = [
        ":compiler_plugin",
        # copybara:uncomment_begin(no OSS for gunit_main_no_heapcheck)
        # "@com_google_googletest//:gtest_main_no_heapcheck",
        # copybara:uncomment_end_and_comment_begin
        "@com_google_googletest//:gtest_main",
        # copybara:comment_end
        "@com_google_absl//absl/strings:string_view",
        "//litert/c:litert_builder",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/c/options:litert_compiler_options",
        "//litert/c:litert_opaque_options_header",
        "//litert/cc/internal:litert_handle",
        "//litert/cc/options:litert_compiler_options",
        "//litert/cc:litert_environment",
        "//litert/cc/internal:litert_op_options",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_opaque_options",
        "//litert/cc:litert_options_testonly",
        "//litert/core:build_stamp",
        "//litert/core:filesystem",
        "//litert/core/model",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/tools:dump",
        "//litert/vendors/c:litert_compiler_plugin",
    ],
)

cc_test(
    name = "compiler_plugin_transformation_test",
    srcs = ["compiler_plugin_transformation_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
        "//litert/vendors/examples:example_plugin_so",
    ],
    # copybara:uncomment malloc = "//base:system_malloc",
    tags = [
        # Sanitizer runtimes are incompatible with RTLD_DEEPBIND.
        "noasan",
        "nomsan",
        "nosan",
        "notsan",
        # TODO(yunandrew): Re-enable once the test is fixed.
        "notap",
    ],
    deps = [
        ":compiler_plugin",
        # copybara:uncomment_begin(no OSS for gunit_main_no_heapcheck)
        # "@com_google_googletest//:gtest_main_no_heapcheck",
        # copybara:uncomment_end_and_comment_begin
        "@com_google_googletest//:gtest_main",
        # copybara:comment_end
        "@com_google_absl//absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/core/model",
        "//litert/core/model:model_serialize",
        "//litert/core/util:flatbuffer_tools_testonly",
        "//litert/test:common",
    ],
)

cc_library(
    name = "algo",
    srcs = ["algo.cc"],
    hdrs = ["algo.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_layout",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_detail",
        "//litert/core:insert_order_map",
        "//litert/core/model",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_test(
    name = "algo_test",
    srcs = ["algo_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
    ],
    deps = [
        ":algo",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc/internal:litert_extended_model",
        "//litert/cc/internal:litert_model_predicates",
        "//litert/core/model",
        "//litert/core/model:graph_validation",
        "//litert/core/model:model_serialize",
        "//litert/test:common",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_compiler_options",
    srcs = ["litert_compiler_options.cc"],
    hdrs = ["litert_compiler_options.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c/options:litert_compiler_options",
        "//litert/cc:litert_macros",
        "//litert/core:litert_toml_parser",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_compiler_options_test",
    srcs = ["litert_compiler_options_test.cc"],
    deps = [
        ":litert_compiler_options",
        "//litert/c:litert_common",
        "//litert/c/options:litert_compiler_options",
        "@com_google_googletest//:gtest_main",
        "@tomlplusplus",
    ],
)
