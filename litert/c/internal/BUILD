# Copyright 2025 Google LLC.
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

# copybara:uncomment_begin(google-only)
# load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
# load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")
# copybara:uncomment_end

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert/c:__subpackages__",
        # "//litert/cc:__subpackages__",  # only for litert_external_litert_buffer_context
        # "//litert/test:__subpackages__",
        # "//litert/runtime:__pkg__",
        # "//litert/runtime/accelerators:__subpackages__",
        # "//litert/runtime/dispatch:__subpackages__",
        # "//litert/vendors/c:__subpackages__",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
)

cc_library(
    name = "litert_accelerator",
    srcs = ["litert_accelerator.cc"],
    hdrs = ["litert_accelerator.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "//litert/core:environment",
        "//litert/runtime:accelerator",
    ],
)

cc_test(
    name = "litert_accelerator_test",
    srcs = ["litert_accelerator_test.cc"],
    deps = [
        ":litert_accelerator",
        ":litert_accelerator_registration",
        "//litert/c:litert_common",
        "//litert/c:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/runtime:accelerator",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_accelerator_registration",
    srcs = ["litert_accelerator_registration.cc"],
    hdrs = ["litert_accelerator_registration.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_metrics",
        "//litert/c:litert_opaque_options",
        "//litert/cc:litert_expected",
        "//litert/core:environment",
        "//litert/runtime:accelerator",
        "//litert/runtime:accelerator_registry",
    ],
)

cc_test(
    name = "litert_accelerator_registration_test",
    srcs = ["litert_accelerator_registration_test.cc"],
    deps = [
        ":litert_accelerator_registration",
        "//litert/c:litert_common",
        "//litert/c:litert_environment",
        "//litert/c:litert_opaque_options",
        "//litert/runtime:accelerator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_tensor_buffer_registry",
    srcs = ["litert_tensor_buffer_registry.cc"],
    hdrs = ["litert_tensor_buffer_registry.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_macros",
        "//litert/runtime:tensor_buffer_registry",
    ],
)

cc_library(
    name = "litert_external_litert_buffer_context",
    srcs = ["litert_external_litert_buffer_context.cc"],
    hdrs = ["litert_external_litert_buffer_context.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_dispatch_headers",
        "//litert/cc:litert_macros",
        "//litert/runtime:external_litert_buffer_context",
        "//tflite/c:common",
    ],
)

exports_files(srcs = glob(["litert_*.h"]))

filegroup(
    name = "internal_headers",
    srcs = glob(["litert_*.h"]),
    visibility = ["//litert/c:__pkg__"],
)
