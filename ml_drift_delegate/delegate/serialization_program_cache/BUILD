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

load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

# copybara:uncomment package(default_applicable_licenses = ["//third_party/odml:license"])

flatbuffer_cc_library(
    name = "serialization_program_cache_schema",
    srcs = ["serialization_program_cache_schema.fbs"],
    flatc_args = [
        "--gen-object-api",
    ],
)

cc_library(
    name = "serialization_program_cache",
    srcs = ["serialization_program_cache.cc"],
    hdrs = ["serialization_program_cache.h"],
    visibility = ["//ml_drift_delegate:__subpackages__"],
    deps = [
        ":serialization_program_cache_schema",
        "//ml_drift_delegate/delegate/serialization_weight_cache:file_util",
        "//ml_drift_delegate/delegate/serialization_weight_cache:mmap_handle",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "serialization_program_cache_test",
    srcs = ["serialization_program_cache_test.cc"],
    deps = [
        ":serialization_program_cache",
        "//ml_drift_delegate/delegate/serialization_weight_cache:file_util",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
