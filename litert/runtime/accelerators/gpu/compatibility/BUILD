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
load("//litert/build_common:special_rule.bzl", "gles_linkopts")

# copybara:uncomment_begin(google-only)
# package(
#     default_applicable_licenses = ["//third_party/odml:license"],
# )
# copybara:uncomment_end

cc_library(
    name = "ml_drift_compatibility",
    srcs = ["ml_drift_compatibility.cc"],
    hdrs = ["ml_drift_compatibility.h"],
    linkopts = gles_linkopts(),
    visibility = ["//visibility:public"],
    deps = [
        "//litert/c/internal:litert_logging",
        "//tflite/experimental/acceleration/compatibility:android_info",
        "//tflite/experimental/acceleration/compatibility:canonicalize_value",
        "//tflite/experimental/acceleration/compatibility:database_fbs",
        "//tflite/experimental/acceleration/compatibility:devicedb",
        "//tflite/experimental/acceleration/compatibility:mldrift_compatibility_binary_embed",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
    ],
)

cc_test(
    name = "ml_drift_compatibility_test",
    srcs = ["ml_drift_compatibility_test.cc"],
    deps = [
        ":ml_drift_compatibility",
        "//tflite/experimental/acceleration/compatibility:android_info",
        "//tflite/experimental/acceleration/compatibility:devicedb_sample",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
