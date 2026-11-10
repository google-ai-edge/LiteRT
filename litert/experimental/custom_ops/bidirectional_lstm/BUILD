# Copyright 2026 Google LLC.
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
    default_visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_public",
        # "//third_party/py/litert_torch:__subpackages__",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
)

cc_library(
    name = "bidirectional_lstm_impl",
    srcs = ["bidirectional_lstm_impl.cc"],
    hdrs = ["bidirectional_lstm_impl.h"],
    deps = [
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "bidirectional_lstm_custom_op",
    srcs = ["bidirectional_lstm_custom_op.cc"],
    hdrs = ["bidirectional_lstm_custom_op.h"],
    deps = [
        ":bidirectional_lstm_impl",
        "//litert/cc:litert_common",
        "//litert/cc:litert_custom_op_kernel",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)

cc_test(
    name = "bidirectional_lstm_custom_op_test",
    srcs = ["bidirectional_lstm_custom_op_test.cc"],
    deps = [
        ":bidirectional_lstm_custom_op",
        ":bidirectional_lstm_impl",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_options",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc:litert_tensor_buffer",
        "//litert/cc:litert_tensor_buffer_types",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)
