# Copyright 2025 Google LLC.
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

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

# Testing ComplianceLint.
cc_library(
    name = "arithmetic_tflite",
    srcs = ["arithmetic_tflite.cc"],
    hdrs = ["arithmetic_tflite.h"],
    deps = [
        "//tensor:arithmetic_graph",
        "//tensor:datatypes",
        "//tensor/internal:graph",
        "//tensor/internal:mixin",
        "//tensor/utils:macros",
        "//tflite/schema:schema_fbs_with_mutable",
        "//tflite/types:half",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "linked_flat_hash_map",
    hdrs = ["linked_flat_hash_map.h"],
    deps = ["@com_google_absl//absl/container:flat_hash_map"],
)

cc_library(
    name = "tflite_flatbuffer_conversion",
    srcs = ["tflite_flatbuffer_conversion.cc"],
    hdrs = ["tflite_flatbuffer_conversion.h"],
    deps = [
        ":arithmetic_tflite",
        ":linked_flat_hash_map",
        "//tensor",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/internal:graph",
        "//tensor/internal:graph_traversal",
        "//tensor/utils:macros",
        "//tflite:framework_stable",
        "//tflite/c:c_api_types",
        "//tflite/core:framework",
        "//tflite/kernels:builtin_ops",
        "//tflite/schema:schema_fbs_with_mutable",
        "@XNNPACK",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "tflite_flatbuffer_conversion_test",
    srcs = ["tflite_flatbuffer_conversion_test.cc"],
    deps = [
        ":arithmetic_tflite",
        ":tflite_flatbuffer_conversion",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_options",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/internal:graph",
        "//tensor/internal:matchers",
        "//tensor/utils:matchers",
        "//tflite:framework",
        "//tflite:test_util",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core:framework",
        "//tflite/kernels:builtin_ops",
        "//tflite/schema:schema_fbs_with_mutable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
