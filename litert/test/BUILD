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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//litert/build_common:symlink_files.bzl", "symlink_files")  # @unused
load("//litert/build_common:tfl_model_gen.bzl", "tfl_model_gen")  # @unused

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",  # copybara:comment_replace "//litert:litert_public",
    ],
)

# copybara:uncomment_begin(internal)
# tfl_model_gen(
#     name = "mlir_test_data",
#     srcs = glob(["testdata/*.mlir"]),
# )
# copybara:uncomment_end

# copybara:comment_begin(external)
symlink_files(
    name = "mlir_test_data",
    srcs = ["@models"],
    declared_outputs = [
        f.replace(".mlir", ".tflite")
        for f in glob(["testdata/*.mlir"])
        if not glob([f.replace(".mlir", ".tflite")])
    ],
    dst = "testdata",
    flatten = True,
)
# copybara:comment_end

# copybara:uncomment_begin(internal)
# filegroup(name = "ats_test_data")
# copybara:uncomment_end

# copybara:comment_begin(external)
symlink_files(
    name = "ats_test_data",
    srcs = ["@ats_models"],
    dst = "testdata",
    exclude_files = glob(["testdata/*.tflite"]),
    flatten = True,
)
# copybara:comment_end

filegroup(
    name = "tflite_test_data",
    srcs = glob(["testdata/*.tflite"]),
)

filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/*.tflite",
        "testdata/*.bin",
    ]) + [":mlir_test_data"],
)

cc_library(
    name = "common",
    testonly = 1,
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    copts = select({
        # TODO (b/449178556): Add different build configuration in TAP for continuous integration test.
        "//litert:without_builtin_ops": ["-DLITERT_NO_BUILTIN_OPS"],
        "//conditions:default": [],
    }),
    deps = [
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc/internal:litert_extended_model",
        "//litert/core:filesystem",
        "//litert/core/model:model_buffer",
        "//litert/core/util:flatbuffer_tools",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
        "//tflite/c:common",
        "//tflite/core:private_cc_api_stable",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ] + select({
        "//litert:without_builtin_ops": [],
        "//conditions:default": ["//tflite/kernels:builtin_ops"],
    }) + [
        "@local_tsl//tsl/platform",
    ],
)

cc_library(
    name = "simple_model",
    testonly = 1,
    hdrs = ["testdata/simple_model_test_vectors.h"],
    data = ["testdata/simple_model.tflite"],
    deps = [
        "//litert/c:litert_model",
        "//litert/cc:litert_layout",
    ],
)

cc_library(
    name = "simple_model_custom_op",
    testonly = 1,
    srcs = [],
    hdrs = ["testdata/simple_model_custom_op_test_vectors.h"],
    data = ["testdata/simple_model_custom_op.tflite"],
    deps = [
        "//litert/c:litert_model",
        "//litert/cc:litert_layout",
    ],
)

cc_library(
    name = "simple_model_npu",
    testonly = 1,
    srcs = [],
    hdrs = ["testdata/simple_model_test_vectors.h"],
    data = [
        "testdata/simple_add_uint16_qualcomm.bin",
        "testdata/simple_model_google_tensor.bin",
        "testdata/simple_model_mtk.bin",
        "testdata/simple_model_npu.tflite",
        "testdata/simple_model_qualcomm.bin",
    ],
    deps = [
        "//litert/c:litert_model",
        "//litert/cc:litert_layout",
    ],
)

cc_library(
    name = "simple_cascade_model_npu",
    testonly = 1,
    srcs = [],
    hdrs = ["testdata/simple_cascade_model_test_vectors.h"],
    data = [
        "testdata/simple_cascade_model_npu.tflite",
        "testdata/simple_model_google_tensor.bin",
        "testdata/simple_model_mtk.bin",
    ],
    deps = [
        "//litert/c:litert_model",
        "//litert/cc:litert_layout",
    ],
)

cc_library(
    name = "simple_mixed_cascade_model_npu",
    testonly = 1,
    srcs = [],
    hdrs = ["testdata/simple_mixed_cascade_model_test_vectors.h"],
    data = [
        "testdata/simple_mixed_cascade_model_npu.tflite",
        "testdata/simple_model_google_tensor.bin",
        "testdata/simple_model_mtk.bin",
    ],
    deps = [
        "//litert/c:litert_model",
        "//litert/cc:litert_layout",
    ],
)

cc_library(
    name = "test_models",
    hdrs = ["test_models.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "matchers",
    testonly = True,
    hdrs = ["matchers.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/cc:litert_common",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc/internal:litert_c_types_printing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "rng_fixture",
    testonly = True,
    hdrs = ["rng_fixture.h"],
    deps = [
        ":fuzz",
        "//litert/c/internal:litert_logging",
        "//litert/cc/internal:litert_rng",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "rng_fixture_oss",
    testonly = True,
    hdrs = ["rng_fixture.h"],
    deps = [
        ":fuzz",
        "//litert/c/internal:litert_logging",
        "//litert/cc/internal:litert_rng",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "matchers_test",
    srcs = ["matchers_test.cc"],
    deps = [
        ":matchers",
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_model",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

# Use this library if you want to enforce an OSS environment for your test.
cc_library(
    name = "matchers_oss",
    testonly = True,
    hdrs = ["matchers.h"],
    defines = ["LITERT_DEFINE_GTEST_STATUS_PRINTER"],
    tags = ["avoid_dep"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/cc:litert_common",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc/dynamic_runtime:litert_model",
        "//litert/cc/internal:litert_c_types_printing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "fuzz",
    hdrs = ["fuzz.h"],
)

cc_library(
    name = "simple_buffer",
    hdrs = ["simple_buffer.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_tensor_buffer",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_tensor_buffer",
        "//litert/cc/internal:litert_extended_model",
        "//litert/cc/internal:litert_rng",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "simple_buffer_test",
    srcs = ["simple_buffer_test.cc"],
    deps = [
        ":matchers",
        ":rng_fixture",
        ":simple_buffer",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_model",
        "//litert/cc/internal:litert_rng",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

exports_files(
    srcs = [
        "testdata/mobilenet_v2_1.0_224.tflite",
        "testdata/model_magic_test.tflite",
        "testdata/model_with_external_weights.tflite",
        "testdata/shared_input_cpu_npu_google_tensor_precompiled.tflite",
        "testdata/simple_add_op_google_tensor_p25_precompiled.tflite",
        "testdata/simple_add_op_qc_v75_precompiled.tflite",
        "testdata/simple_add_uint16_qualcomm.bin",
        "testdata/simple_model_google_tensor.bin",
        "testdata/simple_model_mtk.bin",
        "testdata/simple_model_npu_google_tensor_precompiled.tflite",
        "testdata/simple_model_npu_mediatek_mt6989_precompiled.tflite",
        "testdata/simple_model_qualcomm.bin",
        "testdata/simple_model_qualcomm_sm8650_precompiled.tflite",
        "testdata/simple_quantized_ops.tflite",
        "testdata/torchaudio_conformer.tflite",
    ],
)
