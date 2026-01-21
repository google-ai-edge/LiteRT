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

cc_library(
    name = "external_weight_loader",
    srcs = ["external_weight_loader_litert.cc"],
    hdrs = [
        "external_weight_loader_litert.h",
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_layout",
        "//litert/c:litert_model_types",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:scoped_weight_source",
        "//litert/core:environment",
        "//litert/core/model:flatbuffer_to_litert",
        "//litert/runtime:open_cl_sync",
        "//tflite:framework",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@opencl_headers",
    ],
)

cc_test(
    name = "external_weight_loader_test",
    srcs = ["external_weight_loader_litert_test.cc"],
    deps = [
        ":external_weight_loader",
        "//litert/c:litert_common",
        "//litert/c:litert_layout",
        "//litert/c:litert_model_types",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc/internal:scoped_file",
        "//litert/cc/internal:scoped_weight_source",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)
