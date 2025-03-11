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
    default_visibility = [
        "//third_party/odml/infra/ml_drift_delegate/litert:__subpackages__",
        "//litert:__subpackages__",
    ],
)

cc_library(
    name = "build_stamp",
    srcs = ["build_stamp.cc"],
    hdrs = ["build_stamp.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
    ],
)

cc_test(
    name = "build_stamp_test",
    srcs = ["build_stamp_test.cc"],
    data = [
        "//litert/test:tflite_test_data",
    ],
    deps = [
        ":build_stamp",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/test:matchers",
    ],
)

cc_library(
    name = "dynamic_loading",
    srcs = ["dynamic_loading.cc"],
    hdrs = ["dynamic_loading.h"],
    linkopts = ["-ldl"],
    deps = [
        ":filesystem",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",  # buildcleaner: keep
    ],
)

cc_library(
    name = "insert_order_map",
    hdrs = ["insert_order_map.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_test(
    name = "insert_order_map_test",
    srcs = ["insert_order_map_test.cc"],
    deps = [
        ":insert_order_map",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "environment",
    srcs = ["environment.cc"],
    hdrs = [
        "environment.h",
        "//litert/c:litert_environment.h",
    ],
    deps = [
        ":accelerator",
        "@com_google_absl//absl/types:span",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_expected",
        "@org_tensorflow//tensorflow/lite:shared_library",
    ],
)

cc_test(
    name = "environment_test",
    srcs = ["environment_test.cc"],
    deps = [
        ":environment",
        "@com_google_googletest//:gtest_main",
        "//litert/c:litert_any",
        "//litert/cc:litert_any",
    ],
)

cc_library(
    name = "accelerator",
    srcs = ["accelerator.cc"],
    hdrs = ["accelerator.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_shared_library",
    ],
)

cc_library(
    name = "accelerator_model_compilation_data",
    hdrs = ["accelerator_model_compilation_data.h"],
    deps = [
        ":accelerator",
        "@com_google_absl//absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
    ],
)

cc_test(
    name = "accelerator_model_compilation_data_test",
    srcs = ["accelerator_model_compilation_data_test.cc"],
    deps = [
        ":accelerator_model_compilation_data",
        "@com_google_googletest//:gtest_main",
        "//litert/c:litert_accelerator_options",
        "//litert/c:litert_common",
        "//litert/test:matchers",
    ],
)

cc_library(
    name = "filesystem",
    srcs = ["filesystem.cc"],
    hdrs = ["filesystem.h"],
    deps = [
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_detail",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
    ],
)

cc_library(
    name = "dispatch_op_schema",
    srcs = ["dispatch_op_schema.cc"],
    hdrs = ["dispatch_op_schema.h"],
    copts = ["-DFLATBUFFERS_LOCALE_INDEPENDENT=0"],
    deps = [
        "@flatbuffers//:runtime_cc",
        "//litert/cc:litert_buffer_ref",
    ],
)

cc_test(
    name = "filesystem_test",
    srcs = ["filesystem_test.cc"],
    deps = [
        ":filesystem",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
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
#         "//litert/c:litert_logging",  # buildcleaner: keep
#         "//litert/test:common",
#         "//litert/test:matchers",
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
