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
    default_visibility = ["//tflite/experimental/litert:__subpackages__"],
)

cc_library(
    name = "litert_compiler_plugin",
    hdrs = ["litert_compiler_plugin.h"],
    deps = [
        "//tflite/experimental/litert/cc:litert_macros",
        "//tflite/experimental/litert/vendors/c:litert_compiler_plugin",
    ],
)

cc_library(
    name = "conversion",
    hdrs = ["conversion.h"],
    deps = [
        ":backend_ir",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_op_code",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/cc:litert_model",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "backend_ir",
    hdrs = ["backend_ir.h"],
    deps = ["//tflite/experimental/litert/c:litert_common"],
)

cc_library(
    name = "partition_with_capabilities",
    hdrs = ["partition_with_capabilities.h"],
    deps = [
        ":backend_ir",
        ":conversion",
        "//tflite/experimental/litert/c:litert_logging",
        "//tflite/experimental/litert/c:litert_model",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/cc:litert_model",
    ],
)

cc_test(
    name = "partition_with_capabilities_test",
    srcs = ["partition_with_capabilities_test.cc"],
    deps = [
        ":conversion",
        ":partition_with_capabilities",
        "//tflite/experimental/litert/c:litert_model",
        "//tflite/experimental/litert/c:litert_op_code",
        "//tflite/experimental/litert/cc:litert_model",
        "//tflite/experimental/litert/core/model",
        "//tflite/experimental/litert/core/model:model_graph",
        "//tflite/experimental/litert/core/util:flatbuffer_tools",
        "//tflite/experimental/litert/vendors/examples:example_conversion_impl",
        "//tflite/experimental/litert/vendors/examples:example_ir",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/schema:schema_fbs",
    ],
)
