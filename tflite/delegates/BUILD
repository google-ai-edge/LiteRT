# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite:build_def.bzl", "tflite_copts", "tflite_linkopts")
load("//tflite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

config_setting(
    name = "tflite_debug_delegate",
    define_values = {"tflite_debug_delegate": "true"},
)

cc_library(
    name = "telemetry",
    srcs = ["telemetry.cc"],
    hdrs = ["telemetry.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//tflite/acceleration/configuration:configuration_fbs",
        "//tflite/core/api",
        "//tflite/core/c:common",
    ],
)

cc_test(
    name = "telemetry_test",
    srcs = ["telemetry_test.cc"],
    linkopts = tflite_linkopts(),
    linkstatic = 1,
    deps = [
        ":telemetry",
        "//tflite/acceleration/configuration:configuration_fbs",
        "//tflite/core/api",
        "//tflite/core/c:common",
        "//tflite/profiling:profile_buffer",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//tflite:kernel_api",
        "//tflite:util",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core:subgraph",
        "//tflite/core/c:common",
        "//tflite/kernels:kernel_util",
    ],
)

cc_library(
    name = "interpreter_utils",
    srcs = ["interpreter_utils.cc"],
    hdrs = ["interpreter_utils.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//tflite:framework",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/api:error_reporter",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    linkopts = tflite_linkopts(),
    linkstatic = 1,
    deps = [
        ":utils",
        "//tflite/core:subgraph",
        "//tflite/core/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "interpreter_utils_test",
    size = "small",
    srcs = ["interpreter_utils_test.cc"],
    features = ["-dynamic_link_test_srcs"],  # see go/dynamic_link_test_srcs
    deps = [
        ":delegate_test_util",
        ":interpreter_utils",
        "//tflite:framework",
        "//tflite/core/c:common",
        "//tflite/kernels:builtin_ops",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "delegate_test",
    size = "small",
    srcs = ["delegate_test.cc"],
    data = ["//tflite:testdata/add.bin"],
    features = ["-dynamic_link_test_srcs"],  # see go/dynamic_link_test_srcs.
    deps = [
        ":delegate_test_util",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite:version",
        "//tflite/converter/experimental/remat:metadata_util",
        "//tflite/converter/schema:schema_conversion_utils",
        "//tflite/core:framework",
        "//tflite/core/c:c_api_experimental",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/kernels:builtin_ops",
        "//tflite/kernels:kernel_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_test(
    name = "opaque_delegate_test",
    size = "small",
    srcs = ["opaque_delegate_test.cc"],
    data = [
        "//tflite:testdata/add.bin",
        "//tflite:testdata/test_custom_node_with_init_data.bin",
    ],
    defines = [
        "TFLITE_USE_OPAQUE_DELEGATE",
    ],
    deps = [
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api_experimental",
        "//tflite/c:common",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "//tflite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "opaque_delegate_strip_error_strings_test",
    size = "small",
    srcs = ["opaque_delegate_test.cc"],
    data = [
        "//tflite:testdata/add.bin",
        "//tflite:testdata/test_custom_node_with_init_data.bin",
    ],
    defines = [
        "TFLITE_USE_OPAQUE_DELEGATE",
        "TF_LITE_STRIP_ERROR_STRINGS",
    ],
    deps = [
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api_experimental",
        "//tflite/c:common",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "//tflite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "delegate_test_util",
    testonly = True,
    srcs = ["delegate_test_util.cc"],
    hdrs = ["delegate_test_util.h"],
    deps = [
        ":utils",
        "//tflite:builtin_ops",
        "//tflite:util",
        "//tflite/core:framework",
        "//tflite/core/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:compatibility",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "serialization",
    srcs = ["serialization.cc"],
    hdrs = ["serialization.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//tflite:minimal_logging",
        "//tflite/core/c:common",
        "@farmhash_archive//:farmhash",
    ],
)

cc_test(
    name = "serialization_test",
    srcs = ["serialization_test.cc"],
    linkopts = tflite_linkopts(),
    linkstatic = 1,
    deps = [
        ":serialization",
        "//tflite:util",
        "//tflite/core/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_portable_test_suite()
