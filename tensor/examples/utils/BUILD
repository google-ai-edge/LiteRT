# Copyright 2026 Google LLC.
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
# ==============================================================================

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "initialization",
    srcs = ["initialization.cc"],
    hdrs = ["initialization.h"],
    deps = [
        "@com_google_absl//absl/flags:config",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:usage",
    ],
)

cc_library(
    name = "minijson",
    srcs = ["minijson.cc"],
    hdrs = ["minijson.h"],
)

cc_library(
    name = "safetensors",
    srcs = ["safetensors.cc"],
    hdrs = ["safetensors.h"],
    deps = [
        ":minijson",
        "//tensor:datatypes",
        "//tensor/internal:fp16",
    ],
)

cc_library(
    name = "perfetto_session",
    srcs = ["perfetto_session.cc"],
    hdrs = ["perfetto_session.h"],
    deps = [
        "//tensor/utils:macros",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@perfetto//:libperfetto_client_experimental",
    ],
)

cc_test(
    name = "perfetto_session_test",
    srcs = ["perfetto_session_test.cc"],
    deps = [
        ":perfetto_session",
        "//tensor/utils:matchers",
        "@com_google_googletest//:gtest_main",
        "@perfetto//:libperfetto_client_experimental",
    ],
)
