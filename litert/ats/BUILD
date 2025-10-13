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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//litert/ats:ats.bzl", "litert_define_ats")
load("//litert/integration_test:litert_device.bzl", "litert_device_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",  # copybara:comment_replace "//litert:litert_public",
    ],
)

# LiteRt's compliance test suite (CTS).

cc_test(
    name = "ats",
    srcs = ["ats.cc"],
    # TODO: Figure out data seeding and test re-running.
    args = [
        # copybara:uncomment "--gunit_random_seed=42",
        # copybara:uncomment "--gunit_color=yes",
    ],
    copts = ["-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1"],
    # TODO: Copybara rewrite the gunit flags and remove this tag.
    tags = ["no_oss"],
    deps = [
        ":capture",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/c:litert_logging",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_c_types_printing",
        "//litert/cc:litert_detail",
        "//litert/test/generators",
        "//litert/test/generators:common",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "configure",
    testonly = True,
    srcs = ["configure.cc"],
    hdrs = ["configure.h"],
    deps = [
        ":common",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_rng",
        "//litert/core:filesystem_testonly",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "executor",
    testonly = True,
    hdrs = ["executor.h"],
    deps = [
        ":capture",
        "//litert/c:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_options",
        "//litert/test:simple_buffer",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "register",
    testonly = True,
    hdrs = ["register.h"],
    deps = [
        ":capture",
        ":configure",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_detail",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_rng",
        "//litert/core/model:model_load",
        "//litert/test:simple_buffer",
        "//litert/test/generators",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "executor_test",
    srcs = ["executor_test.cc"],
    deps = [
        ":executor",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/core/model",
        "//litert/test:matchers",
        "//litert/test:simple_buffer",
        "//litert/test/generators:graph_helpers",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "inference_fixture",
    testonly = True,
    hdrs = ["inference_fixture.h"],
    deps = [
        ":capture",
        ":common",
        ":configure",
        ":executor",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_c_types_printing",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core/model",
        "//litert/test:matchers",
        "//litert/test:rng_fixture",
        "//litert/test:simple_buffer",
        "//litert/test/generators:common",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

litert_device_test(
    name = "executor_test_npu",
    srcs = ["executor_test.cc"],
    backend_id = "qualcomm",
    data = ["//litert/test:testdata/simple_model.tflite"],
    defines = ["_TEST_NPU"],
    deps = [
        ":executor",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/core/model",
        "//litert/test:matchers",
        "//litert/test:simple_buffer",
        "//litert/test/generators:graph_helpers",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "check_ats",
    srcs = ["check_ats.cc"],
    data = [
        "//litert/test:testdata/simple_model.tflite",
        "//litert/vendors/examples:example_dispatch_so",
        "//litert/vendors/examples:example_plugin_so",
    ],
    deps = [
        ":capture",
        ":common",
        ":configure",
        ":inference_fixture",
        ":register",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_detail",
        "//litert/cc:litert_expected",
        "//litert/test:common",
        "//litert/test/generators",
        "//litert/test/generators:common",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "capture",
    testonly = True,
    hdrs = ["capture.h"],
    deps = [
        ":common",
        ":print",
        "//litert/cc:litert_detail",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "capture_test",
    srcs = ["capture_test.cc"],
    deps = [
        ":capture",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "common",
    testonly = True,
    hdrs = ["common.h"],
    deps = ["@com_google_absl//absl/strings:str_format"],
)

cc_library(
    name = "print",
    testonly = True,
    hdrs = ["print.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "print_test",
    srcs = ["print_test.cc"],
    deps = [
        ":print",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

# PRE-CONFIGURED CTS SUITES ########################################################################

litert_define_ats(
    name = "sample_cpu_ats",
    backend = "cpu",
)

litert_define_ats(
    name = "example_npu_ats",
    backend = "example",
    do_register = [
        "sub.*f32",
        "mul.*f32",
    ],
)

litert_define_ats(
    name = "qualcomm_ats",
    backend = "qualcomm",
    dont_register = [
        ".*5d.*",  # Filter out 5d tests.
        ".*6d.*",  # Filter out 6d tests.
        ".*f32.*",  # Filter out f32 tests. TODO: Debug float numerics issue.
        # NOTE: Comment above and uncomment to run one of the fp tests with numerics issues.
        # "^(?:(?!npu_ats_87).)*$$",
    ],
    extra_flags = [
        "--f16_range_for_f32=true",
        "--data_seed=42",
    ],
)
