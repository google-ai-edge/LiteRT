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
    default_visibility = [
        # copybara:uncomment "//third_party/mediapipe/calculators/tensor:__subpackages__",
        "//litert:__subpackages__",
    ],
)

cc_library(
    name = "flatbuffer_tools",
    srcs = ["flatbuffer_tools.cc"],
    hdrs = [
        "flatbuffer_tools.h",
        "//litert/cc:litert_consts.h",
    ],
    deps = [
        "//third_party/absl/container:inlined_vector",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
        "//third_party/flatbuffers:runtime_cc",
        "//litert/c:litert_common",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_detail",
        "//litert/cc:litert_expected",
        "//litert/core:filesystem",
        "//third_party/tensorflow/compiler/mlir/lite:allocation",
        "//third_party/tensorflow/lite:model_builder",
        "//third_party/tensorflow/lite:stderr_reporter",
        "//third_party/tensorflow/lite/schema:schema_fbs",
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
        "//testing/base/public:gunit_main",
        "//third_party/absl/strings:string_view",
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
        "//third_party/absl/strings",
        "//third_party/absl/types:span",
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
        "//testing/base/public:gunit_main",
        "//third_party/absl/types:span",
        "//litert/c:litert_model",
    ],
)
