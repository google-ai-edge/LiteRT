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

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_stable_abi_users"],
)

cc_library(
    name = "litert_compiler_plugin",
    hdrs = ["litert_compiler_plugin.h"],
    deps = [
        "//litert/c:litert_environment_options",
        "//litert/c:litert_options",
        "//litert/cc:litert_macros",
        "//litert/vendors/c:litert_compiler_plugin",
    ],
)

cc_library(
    name = "conversion",
    hdrs = ["conversion.h"],
    deps = [
        ":backend_ir",
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_model",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "backend_ir",
    hdrs = ["backend_ir.h"],
    deps = ["//litert/c:litert_common"],
)

cc_library(
    name = "partition_with_capabilities",
    hdrs = ["partition_with_capabilities.h"],
    deps = [
        ":conversion",
        "//litert/c:litert_logging",
        "//litert/c:litert_model",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_model",
    ],
)

cc_library(
    name = "convert_graph",
    hdrs = ["convert_graph.h"],
    deps = [
        ":conversion",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_model",
    ],
)

cc_library(
    name = "ir_types",
    hdrs = ["ir_types.h"],
    deps = [
        ":backend_ir",
        ":conversion",
        "//litert/cc:litert_expected",
    ],
)

cc_library(
    name = "options_helper",
    hdrs = ["options_helper.h"],
    deps = [
        "//litert/c:litert_environment_options",
        "//litert/c:litert_options",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_handle",
        "//litert/cc:litert_opaque_options",
        "//litert/cc:litert_options",
    ],
)

cc_test(
    name = "partition_with_capabilities_test",
    srcs = ["partition_with_capabilities_test.cc"],
    deps = [
        ":partition_with_capabilities",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_model",
        "//litert/core/model:model_graph_testonly",
        "//litert/core/model:model_testonly",
        "//litert/core/util:flatbuffer_tools_testonly",
        "//litert/vendors/examples:example_conversion_impl",
        "//litert/vendors/examples:example_ir",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "convert_graph_test",
    srcs = ["convert_graph_test.cc"],
    deps = [
        ":backend_ir",
        ":convert_graph",
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_model",
        "//litert/core/model:model_graph_testonly",
        "//litert/core/model:model_testonly",
        "//litert/core/util:flatbuffer_tools_testonly",
        "//litert/test:matchers",
        "//litert/vendors/examples:example_conversion_impl",
        "//litert/vendors/examples:example_ir",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/schema:schema_fbs",
    ],
)
