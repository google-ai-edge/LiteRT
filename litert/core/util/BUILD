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
        # copybara:uncomment "@mediapipe//mediapipe/calculators/tensor:__subpackages__",
        "//litert:__subpackages__",
    ],
)

cc_library(
    name = "flatbuffer_tools",
    srcs = ["flatbuffer_tools.cc"],
    hdrs = [
        "flatbuffer_tools.h",
    ],
    deps = [
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@flatbuffers//:runtime_cc",
        "//litert/c:litert_common",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_consts",
        "//litert/cc:litert_expected",
        "//litert/core:filesystem",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:allocation",
        "@org_tensorflow//tensorflow/lite:model_builder",
        "@org_tensorflow//tensorflow/lite:stderr_reporter",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "flatbuffer_tools_test",
    srcs = ["flatbuffer_tools_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    deps = [
        ":flatbuffer_tools",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings:string_view",
        "//litert/cc:litert_buffer_ref",
        "//litert/test:common",
        "//litert/test:matchers",
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
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/cc:litert_expected",
    ],
)

cc_test(
    name = "tensor_type_util_test",
    srcs = ["tensor_type_util_test.cc"],
    deps = [
        ":tensor_type_util",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/types:span",
        "//litert/c:litert_model",
    ],
)
