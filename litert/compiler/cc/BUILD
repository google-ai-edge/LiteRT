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
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors:__subpackages__"],
)

cc_library(
    name = "litert_model",
    hdrs = ["litert_model.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/c/internal:litert_compiler_context",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_ranked_tensor_type",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_model_test",
    srcs = ["litert_model_test.cc"],
    data = [
        "//litert/test:testdata/simple_multi_op.tflite",
    ],
    deps = [
        ":litert_model",
        "//litert/c:litert_common",
        "//litert/c/internal:litert_compiler_context",
        "//litert/test:common",
        "@com_google_googletest//:gtest_main",
    ],
)
