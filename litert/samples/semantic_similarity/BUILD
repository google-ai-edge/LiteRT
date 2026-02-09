# Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_binary(
    name = "tokenizer_client",
    srcs = ["tokenizer_client.cc"],
    deps = [
        # copybara:uncomment "//base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@sentencepiece//:sentencepiece_processor",
    ],
)

cc_binary(
    name = "semantic_similarity",
    srcs = ["main.cc"],
    # copybara:uncomment data = ["//litert/runtime/accelerators/gpu:ml_drift_cl_accelerator_so"],
    deps = [
        # copybara:uncomment "//base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "//litert/cc/options:litert_gpu_options",
        "@sentencepiece//:sentencepiece_processor",
    ],
)

cc_binary(
    name = "embedding_client",
    srcs = ["embedding_client.cc"],
    deps = [
        # copybara:uncomment "//base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//litert/cc:litert_api_with_dynamic_runtime",
    ],
)
