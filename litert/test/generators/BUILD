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
    testonly = True,
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
    name = "unary",
    testonly = True,
    hdrs = ["unary.h"],
    deps = [
        ":common",
        ":graph_helpers",
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_c_types_printing",
        "//litert/cc/internal:litert_detail",
        "//litert/cc/internal:litert_rng",
        "//litert/core/model",
        "//litert/test:simple_buffer",
        "//tflite/converter/schema:schema_fbs",
        "//tflite/types:half",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "unary_test",
    srcs = ["unary_test.cc"],
    deps = [
        ":common",
        ":unary",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_element_type",
        "//litert/cc/internal:litert_rng",
        "//litert/core/model",
        "//litert/test:matchers",
        "//litert/test:rng_fixture",
        "//litert/test:simple_buffer",
        "//tflite/converter/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "transpose",
    testonly = True,
    hdrs = ["transpose.h"],
    deps = [
        ":common",
        ":graph_helpers",
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc/internal:litert_detail",
        "//litert/cc/internal:litert_rng",
        "//litert/core/model",
        "//litert/test:simple_buffer",
        "//tflite/c:common",
        "//tflite/kernels/internal:tensor_utils_no_eigen",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "transpose_test",
    srcs = ["transpose_test.cc"],
    deps = [
        ":common",
        ":transpose",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc/internal:litert_rng",
        "//litert/core/model",
        "//litert/test:matchers",
        "//litert/test:rng_fixture",
        "//litert/test:simple_buffer",
        "//tflite/converter/schema:schema_fbs",
        "//tflite/kernels/internal:tensor_utils_no_eigen",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/types:span",
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
    name = "fully_connected",
    testonly = True,
    hdrs = ["fully_connected.h"],
    deps = [
        ":common",
        ":graph_helpers",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc/internal:litert_detail",
        "//litert/cc/internal:litert_rng",
        "//litert/core/model",
        "//litert/test:simple_buffer",
        "//tflite/converter/schema:schema_fbs",
        "//tflite/kernels/internal:cppmath",
        "//tflite/kernels/internal:quantization_util",
        "//tflite/kernels/internal:reference_base",
        "//tflite/kernels/internal:runtime_shape",
        "//tflite/kernels/internal:tensor_utils",
        "//tflite/kernels/internal:types",
        "//tflite/schema:schema_fbs",
        "//tflite/types:half",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "fully_connected_test",
    srcs = ["fully_connected_test.cc"],
    deps = [
        ":common",
        ":fully_connected",
        "//litert/ats:executor",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/c/options:litert_cpu_options",
        "//litert/cc:litert_common",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_options",
        "//litert/cc/internal:litert_rng",
        "//litert/core/model",
        "//litert/test:matchers",
        "//litert/test:rng_fixture",
        "//litert/test:simple_buffer",
        "//tflite/converter/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
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
        "//tflite/converter/schema:schema_fbs",
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
        ":fully_connected",
        ":no_op",
        ":transpose",
        ":unary",
    ],
)
