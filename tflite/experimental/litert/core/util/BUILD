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
    default_visibility = [
        # copybara:uncomment "@org_tensorflow//third_party/mediapipe/calculators/tensor:__subpackages__",
        "//tflite/experimental/litert:__subpackages__",
    ],
)

cc_library(
    name = "flatbuffer_tools",
    srcs = ["flatbuffer_tools.cc"],
    hdrs = [
        "flatbuffer_tools.h",
        "//tflite/experimental/litert/cc:litert_consts.h",
    ],
    deps = [
        "//tflite:model_builder",
        "//tflite:stderr_reporter",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/cc:litert_buffer_ref",
        "//tflite/experimental/litert/cc:litert_detail",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/core:filesystem",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@flatbuffers//:runtime_cc",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:allocation",
    ],
)

cc_test(
    name = "flatbuffer_tools_test",
    srcs = ["flatbuffer_tools_test.cc"],
    data = [
        "//tflite/experimental/litert/test:mlir_test_data",
        "//tflite/experimental/litert/test:tflite_test_data",
    ],
    deps = [
        ":flatbuffer_tools",
        "//tflite/experimental/litert/cc:litert_buffer_ref",
        "//tflite/experimental/litert/test:common",
        "//tflite/experimental/litert/test:test_macros",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tensor_type_util",
    srcs = [
        "tensor_type_util.cc",
    ],
    hdrs = [
        "tensor_type_util.h",
    ],
    deps = [
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_model",
        "//tflite/experimental/litert/cc:litert_expected",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "tensor_type_util_test",
    srcs = ["tensor_type_util_test.cc"],
    deps = [
        ":tensor_type_util",
        "//tflite/experimental/litert/c:litert_model",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
