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

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",  # copybara:comment_replace "//litert:litert_public",
    ],
)

# Imperitive logic for generating test graphs in memory.

cc_library(
    name = "common",
    hdrs = ["common.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_expected",
        "//litert/cc/internal:litert_detail",
        "//litert/cc/internal:litert_rng",
        "//litert/core/model",
        "//litert/test:simple_buffer",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "no_op",
    testonly = True,
    hdrs = ["no_op.h"],
    deps = [
        ":common",
        ":graph_helpers",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_c_types_printing",
        "//litert/cc/internal:litert_detail",
        "//litert/cc/internal:litert_rng",
        "//litert/core/model",
        "//litert/test:simple_buffer",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "no_op_test",
    srcs = ["no_op_test.cc"],
    deps = [
        ":common",
        ":no_op",
        "//litert/cc/internal:litert_rng",
        "//litert/test:matchers",
        "//litert/test:rng_fixture",
        "//litert/test:simple_buffer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "binary_no_bcast",
    testonly = True,
    hdrs = ["binary_no_bcast.h"],
    deps = [
        ":common",
        ":graph_helpers",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_c_types_printing",
        "//litert/cc/internal:litert_detail",
        "//litert/cc/internal:litert_rng",
        "//litert/core/model",
        "//litert/core/model:model_load",
        "//litert/core/model:model_serialize",
        "//litert/core/util:flatbuffer_tools_testonly",
        "//litert/test:simple_buffer",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "binary_no_bcast_test",
    srcs = ["binary_no_bcast_test.cc"],
    deps = [
        ":binary_no_bcast",
        ":common",
        "//litert/c:litert_op_code",
        "//litert/cc/internal:litert_rng",
        "//litert/test:matchers",
        "//litert/test:rng_fixture",
        "//litert/test:simple_buffer",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "extra_model",
    testonly = True,
    hdrs = ["extra_model.h"],
    deps = [
        ":common",
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc/internal:litert_rng",
        "//litert/core/model:model_load",
        "//litert/test:simple_buffer",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "graph_helpers",
    testonly = True,
    hdrs = ["graph_helpers.h"],
    deps = [
        ":common",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core/model",
        "//litert/core/model:model_load",
        "//litert/core/model:model_serialize",
        "//litert/core/util:flatbuffer_tools",
    ],
)

cc_test(
    name = "graph_helpers_test",
    srcs = ["graph_helpers_test.cc"],
    deps = [
        ":graph_helpers",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/core/model",
        "//litert/test:matchers",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "generators",
    testonly = True,
    hdrs = ["generators.h"],
    deps = [
        ":binary_no_bcast",
        ":common",
        ":extra_model",
        ":no_op",
    ],
)
