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
    default_visibility = ["//litert:litert_internal_users"],
)

cc_library(
    name = "ynnpack_accelerator",
    srcs = ["ynnpack_accelerator.cc"],
    hdrs = ["ynnpack_accelerator.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options",
        "//litert/c/internal:litert_accelerator_def",
        "//litert/c/internal:litert_runtime_context_header",
        "//litert/c/options:litert_cpu_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/runtime:accelerator",
        "//litert/runtime:litert_cpu_options",
        "//litert/runtime/accelerators:accelerator_implementation_helper",
        "//tflite/c:c_api_types",
        "//tflite/core/c:private_common",
        "//tflite/delegates/ynnpack:ynnpack_delegate",
    ],
    alwayslink = 1,
)

cc_test(
    name = "ynnpack_accelerator_test",
    srcs = ["ynnpack_accelerator_test.cc"],
    copts = select({
        "//litert/build_common:litert_enable_ynnpack": [
            "-DLITERT_TEST_EXPECT_YNNPACK",
        ],
        "//conditions:default": [],
    }),
    data = [
        "//litert/test:testdata/simple_model.tflite",
    ],
    target_compatible_with = select({
        "//litert/build_common:litert_disable_cpu": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_environment",
        "//litert/c:litert_model",
        "//litert/c:litert_opaque_options_header",
        "//litert/c:litert_options",
        "//litert/c/options:litert_cpu_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/options:litert_cpu_options",
        "//litert/runtime:compiled_model",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model",
        "//tflite:framework",
        "//tflite:schema_fbs_version",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)
