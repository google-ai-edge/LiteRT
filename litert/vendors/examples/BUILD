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

load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:private"],
)

litert_dynamic_lib(
    name = "example_plugin",
    srcs = [
        "example_plugin.cc",
        "example_plugin_common.cc",
        "example_plugin_common.h",
    ],
    hdrs = ["//litert/vendors/c:litert_compiler_plugin.h"],
    export_litert_only = True,
    linkstatic = 1,
    shared_lib_name = "example_plugin_so",
    so_name = "libLiteRtCompilerPlugin_Example.so",
    visibility = ["//litert:__subpackages__"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options",
        "//litert/c:litert_logging",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/c:litert_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_model_predicates",
        "//litert/cc:litert_op_options",
    ],
)

cc_test(
    name = "example_plugin_test",
    srcs = [
        "example_plugin_test.cc",
    ],
    data = ["//litert/test:mlir_test_data"],
    deps = [
        ":example_plugin",  # buildcleaner: keep
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_model",
        "//litert/core/model:model_testonly",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/vendors/cc:litert_compiler_plugin",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "example_conversion_impl",
    srcs = ["example_conversion_impl.cc"],
    hdrs = ["example_conversion_impl.h"],
    visibility = ["//litert/vendors/cc:__pkg__"],
    deps = [
        ":example_ir",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/c:litert_op_options",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_model",
        "//litert/vendors/cc:backend_ir",
        "//litert/vendors/cc:conversion",
        "//litert/vendors/cc:ir_types",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "example_conversion_impl_test",
    srcs = ["example_conversion_impl_test.cc"],
    deps = [
        ":example_conversion_impl",
        ":example_ir",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_model",
        "//litert/core/model:model_graph_testonly",
        "//litert/core/model:model_testonly",
        "//litert/core/util:flatbuffer_tools_testonly",
        "//litert/test:matchers",
        "//litert/vendors/cc:conversion",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "example_ir",
    srcs = ["example_ir.cc"],
    hdrs = ["example_ir.h"],
    visibility = ["//litert/vendors/cc:__pkg__"],
    deps = [
        "//litert/c:litert_common",
        "//litert/vendors/cc:backend_ir",
        "//litert/vendors/cc:ir_types",
    ],
)

cc_library(
    name = "example_plugin_with_conversions",
    srcs = [
        "example_plugin_common.cc",
        "example_plugin_common.h",
        "example_plugin_with_conversions.cc",
    ],
    deps = [
        ":example_conversion_impl",
        ":example_ir",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options",
        "//litert/c:litert_model",
        "//litert/c:litert_options",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/vendors/c:litert_compiler_plugin",
        "//litert/vendors/cc:convert_graph",
        "//litert/vendors/cc:partition_with_capabilities",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "example_plugin_with_conversions_test",
    srcs = ["example_plugin_with_conversions_test.cc"],
    data = ["//litert/test:mlir_test_data"],
    deps = [
        ":example_plugin_with_conversions",  # buildcleaner: keep
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_model",
        "//litert/core/model:model_testonly",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/vendors/c:litert_compiler_plugin",
        "//litert/vendors/cc:litert_compiler_plugin",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
