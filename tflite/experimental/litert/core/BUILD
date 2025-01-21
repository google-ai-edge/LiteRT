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
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//tflite/experimental/litert:__subpackages__"],
)

cc_library(
    name = "byte_code_util",
    srcs = ["byte_code_util.cc"],
    hdrs = ["byte_code_util.h"],
    deps = [
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_logging",
        "//tflite/experimental/litert/cc:litert_buffer_ref",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/cc:litert_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "byte_code_util_test",
    srcs = ["byte_code_util_test.cc"],
    data = [
        "//tflite/experimental/litert/test:tflite_test_data",
    ],
    deps = [
        ":byte_code_util",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/cc:litert_buffer_ref",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/test:common",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "dynamic_loading",
    srcs = ["dynamic_loading.cc"],
    hdrs = ["dynamic_loading.h"],
    linkopts = ["-ldl"],
    deps = [
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_logging",  # buildcleaner: keep
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "environment",
    srcs = ["environment.cc"],
    hdrs = [
        "environment.h",
        "//tflite/experimental/litert/c:litert_environment.h",
    ],
    deps = [
        ":accelerator",
        "//tflite/experimental/litert/c:litert_any",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_logging",
        "//tflite/experimental/litert/cc:litert_any",
        "//tflite/experimental/litert/cc:litert_expected",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "environment_test",
    srcs = ["environment_test.cc"],
    deps = [
        ":environment",
        "//tflite/experimental/litert/c:litert_any",
        "//tflite/experimental/litert/cc:litert_any",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "accelerator",
    srcs = ["accelerator.cc"],
    hdrs = ["accelerator.h"],
    deps = [
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/cc:litert_expected",
    ],
)

cc_library(
    name = "filesystem",
    srcs = ["filesystem.cc"],
    hdrs = ["filesystem.h"],
    deps = [
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/cc:litert_buffer_ref",
        "//tflite/experimental/litert/cc:litert_detail",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/cc:litert_macros",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "dispatch_op_schema",
    srcs = ["dispatch_op_schema.cc"],
    hdrs = ["dispatch_op_schema.h"],
    copts = ["-DFLATBUFFERS_LOCALE_INDEPENDENT=0"],
    deps = [
        "//tflite/experimental/litert/cc:litert_buffer_ref",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "filesystem_test",
    srcs = ["filesystem_test.cc"],
    deps = [
        ":filesystem",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

# copybara:uncomment_begin(no OSS for unique-test-directory)
# cc_test(
#     name = "dynamic_loading_test",
#     srcs = ["dynamic_loading_test.cc"],
#     tags = [
#         # Sanitizer runtimes are incompatible with RTLD_DEEPBIND.
#         "noasan",
#         "nomsan",
#         "nosan",
#     ],
#     deps = [
#         ":dynamic_loading",
#         ":filesystem",
#         "@com_google_googletest//:gtest_main",
#         "@com_google_absl//absl/strings:string_view",
#         "//tflite/experimental/litert/c:litert_logging",  # buildcleaner: keep
#         "//tflite/experimental/litert/test:common",
#         "//tflite/experimental/litert/test:test_macros",
#     ],
# )
# copybara:uncomment_end

cc_test(
    name = "dispatch_op_schema_test",
    srcs = ["dispatch_op_schema_test.cc"],
    deps = [
        ":dispatch_op_schema",
        "@com_google_googletest//:gtest_main",
    ],
)
