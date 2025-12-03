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

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_public",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
)

cc_library(
    name = "irfft_kernel",
    srcs = ["irfft_kernel.cc"],
    hdrs = ["irfft_kernel.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_custom_op_kernel",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_tensor_buffer",
        "//tflite/c:c_api_types",
        "//tflite/schema:schema_fbs",
        "//third_party/tflite_micro/tensorflow/lite/micro/audio_frontend/src:complex",
        "//third_party/tflite_micro/tensorflow/lite/micro/audio_frontend/src:irfft",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@flatbuffers",
    ],
    alwayslink = 1,
)

cc_library(
    name = "overlap_add_kernel",
    srcs = ["overlap_add_kernel.cc"],
    hdrs = ["overlap_add_kernel.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_custom_op_kernel",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_tensor_buffer",
        "//tflite/c:c_api_types",
        "//tflite/schema:schema_fbs",
        "//third_party/tflite_micro/tensorflow/lite/micro/audio_frontend/src:overlap_add",
        "@flatbuffers",
    ],
    alwayslink = 1,
)

cc_library(
    name = "audio_frontend_ops",
    srcs = ["audio_frontend_ops.cc"],
    hdrs = ["audio_frontend_ops.h"],
    deps = [
        ":irfft_kernel",
        ":overlap_add_kernel",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_options",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "irfft_kernel_test",
    srcs = ["irfft_kernel_test.cc"],
    deps = [
        ":irfft_kernel",
        "//litert/c:litert_common",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc:litert_tensor_buffer",
        "//tflite/c:c_api_types",
        "//tflite/schema:schema_fbs",
        "//third_party/tflite_micro/tensorflow/lite/micro/audio_frontend/testdata:fft_test_data",
        "@com_google_absl//absl/cleanup",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_test(
    name = "overlap_add_kernel_test",
    srcs = ["overlap_add_kernel_test.cc"],
    deps = [
        ":overlap_add_kernel",
        "//litert/c:litert_common",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc:litert_tensor_buffer",
        "//tflite/c:c_api_types",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)
