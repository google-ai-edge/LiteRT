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
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

cc_library(
    name = "litert_compiler_plugin",
    hdrs = ["litert_compiler_plugin.h"],
    deps = [
        "//litert/cc:litert_macros",
        "//litert/vendors/c:litert_compiler_plugin",
    ],
)

cc_library(
    name = "conversion",
    hdrs = ["conversion.h"],
    deps = [
        ":backend_ir",
        "//third_party/absl/container:flat_hash_map",
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_model",
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

cc_test(
    name = "partition_with_capabilities_test",
    srcs = ["partition_with_capabilities_test.cc"],
    deps = [
        ":partition_with_capabilities",
        "//testing/base/public:gunit_main",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_model",
        "//litert/core/model",
        "//litert/core/model:model_graph",
        "//litert/core/util:flatbuffer_tools",
        "//litert/vendors/examples:example_conversion_impl",
        "//litert/vendors/examples:example_ir",
        "//third_party/tensorflow/compiler/mlir/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "convert_graph_test",
    srcs = ["convert_graph_test.cc"],
    deps = [
        ":backend_ir",
        ":convert_graph",
        "//testing/base/public:gunit_main",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_model",
        "//litert/core/model",
        "//litert/core/model:model_graph",
        "//litert/core/util:flatbuffer_tools",
        "//litert/test:matchers",
        "//litert/vendors/examples:example_conversion_impl",
        "//litert/vendors/examples:example_ir",
        "//third_party/tensorflow/compiler/mlir/lite/schema:schema_fbs",
    ],
)
