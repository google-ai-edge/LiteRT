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
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib")
load(":tcft_release_transition.bzl", "arm_vulkan_ml_llvm_compat_file", "tcft_release_file")

package(
    default_visibility = ["//litert:__subpackages__"],
)

TCFT_VISIBILITY = ["@tosa-converter-for-tflite//tosa_converter_for_tflite:__subpackages__"]

alias(
    name = "tflite_flatbuffer_import",
    actual = "//tflite/converter:flatbuffer_import",
    visibility = TCFT_VISIBILITY,
)

alias(
    name = "tflite_lift_tflite_flex_ops",
    actual = "//tflite/converter:lift_tflite_flex_ops",
    visibility = TCFT_VISIBILITY,
)

alias(
    name = "tflite_tf_tfl_passes",
    actual = "//tflite/converter:tf_tfl_passes",
    visibility = TCFT_VISIBILITY,
)

alias(
    name = "tflite_tensorflow_lite",
    actual = "//tflite/converter:tensorflow_lite",
    visibility = TCFT_VISIBILITY,
)

alias(
    name = "tflite_tensorflow_lite_ops",
    actual = "//tflite/converter:tensorflow_lite_ops",
    visibility = TCFT_VISIBILITY,
)

alias(
    name = "tflite_tensorflow_lite_ops_td_files",
    actual = "//tflite/converter:tensorflow_lite_ops_td_files",
    visibility = TCFT_VISIBILITY,
)

alias(
    name = "tflite_tensorflow_lite_quantize",
    actual = "//tflite/converter:tensorflow_lite_quantize",
    visibility = TCFT_VISIBILITY,
)

ARM_VULKAN_ML_COMPILER_TARGET_COMPATIBLE = select({
    "//litert:android": [],
    "//litert:linux": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

tcft_release_file(
    name = "tcft_release_so",
    src = "@tosa-converter-for-tflite//tosa_converter_for_tflite:libtcft.so",
    target_compatible_with = ARM_VULKAN_ML_COMPILER_TARGET_COMPATIBLE,
)

cc_import(
    name = "tcft_c_api",
    shared_library = ":tcft_release_so",
    target_compatible_with = ARM_VULKAN_ML_COMPILER_TARGET_COMPATIBLE,
)

alias(
    name = "tcft_c_api_headers",
    actual = "@tosa-converter-for-tflite//tosa_converter_for_tflite:tcft_c_api_headers",
)

cc_library(
    name = "model_converter_c_api",
    deps = [
        ":model_converter_c_api_link",
        "@model_converter//:model_converter_c_api_headers",
    ],
    target_compatible_with = ARM_VULKAN_ML_COMPILER_TARGET_COMPATIBLE,
)

arm_vulkan_ml_llvm_compat_file(
    name = "model_converter_llvm_compat_so",
    src = "@model_converter//:libmodel_converter.so",
    target_compatible_with = ARM_VULKAN_ML_COMPILER_TARGET_COMPATIBLE,
)

copy_file(
    name = "model_converter_runtime_so",
    src = ":model_converter_llvm_compat_so",
    out = "libmodel_converter.so",
    target_compatible_with = ARM_VULKAN_ML_COMPILER_TARGET_COMPATIBLE,
)

cc_import(
    name = "model_converter_c_api_link",
    shared_library = ":model_converter_runtime_so",
    target_compatible_with = ARM_VULKAN_ML_COMPILER_TARGET_COMPATIBLE,
)

copy_file(
    name = "tcft_runtime_so",
    src = ":tcft_release_so",
    out = "libtcft.so",
    target_compatible_with = ARM_VULKAN_ML_COMPILER_TARGET_COMPATIBLE,
)

filegroup(
    name = "on_device_compiler_libraries",
    srcs = [
        ":model_converter_runtime_so",
        ":tcft_runtime_so",
    ],
    target_compatible_with = ARM_VULKAN_ML_COMPILER_TARGET_COMPATIBLE,
)

cc_library(
    name = "tcft_model_converter",
    srcs = ["tcft_model_converter.cc"],
    hdrs = ["tcft_model_converter.h"],
    deps = [
        ":model_converter_c_api",
        ":model_converter_c_api_link",
        "//litert/vendors/arm_vulkan_ml:capabilities",
        ":tcft_c_api",
        ":tcft_c_api_headers",
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/c:litert_model",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_extended_model",
        "//litert/core/model",
    ],
    target_compatible_with = ARM_VULKAN_ML_COMPILER_TARGET_COMPATIBLE,
)

cc_library(
    name = "jit_compiler",
    srcs = ["jit/jit_compiler.cc"],
    hdrs = ["jit/jit_compiler.h"],
    deps = [
        ":tcft_model_converter",
        "//litert/c:litert_common",
        "//litert/vendors/arm_vulkan_ml/common:graph_desc",
        "//litert/vendors/arm_vulkan_ml/common:vgf_loader",
    ],
)

litert_dynamic_lib(
    name = "compiler_plugin",
    srcs = ["compiler_plugin.cc"],
    hdrs = ["//litert/vendors/c:litert_compiler_plugin.h"],
    export_litert_only = True,
    linkopts = ["-Wl,-rpath,$$ORIGIN"],
    linkstatic = 1,
    shared_lib_name = "compiler_plugin_so",
    so_name = "libLiteRtCompilerPlugin_ArmVulkanML.so",
    target_compatible_with = ARM_VULKAN_ML_COMPILER_TARGET_COMPATIBLE,
    visibility = ["//litert:litert_public"],
    deps = [
        ":jit_compiler",
        "//litert/vendors/arm_vulkan_ml/common:graph_desc",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/c:litert_model",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_context_wrapper",
        "//litert/cc/internal:litert_handle",
        "//litert/cc/internal:litert_opaque_options_wrapper",
        "//litert/cc/internal:litert_options_wrapper",
        "//litert/cc/options:litert_arm_vulkan_ml_options",
        "//litert/vendors/c:litert_compiler_plugin",
    ],
)

copy_file(
    name = "compiler_plugin_runtime_so",
    src = "//litert/c:litert_runtime_c_api_so",
    out = "libLiteRt.so",
    target_compatible_with = ARM_VULKAN_ML_COMPILER_TARGET_COMPATIBLE,
    visibility = ["//litert:litert_public"],
)

filegroup(
    name = "compiler_plugin_so_data",
    srcs = [
        ":compiler_plugin_so",
        ":compiler_plugin_runtime_so",
        ":on_device_compiler_libraries",
    ],
    target_compatible_with = ARM_VULKAN_ML_COMPILER_TARGET_COMPATIBLE,
    visibility = ["//litert:litert_public"],
)

cc_test(
    name = "compiler_plugin_test",
    srcs = ["compiler_plugin_test.cc"],
    data = ["//litert/test:tflite_test_data"],
    deps = [
        ":compiler_plugin",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/c/internal:litert_compiler_context_testonly",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_options",
        "//litert/cc/options:litert_arm_vulkan_ml_options",
        "//litert/core/model",
        "//litert/test:load_test_model",
        "//litert/test:matchers",
        "//litert/vendors/c:litert_compiler_plugin",
        "//litert/vendors/cc:litert_compiler_plugin",
        "@com_google_googletest//:gtest_main",
    ],
    target_compatible_with = ARM_VULKAN_ML_COMPILER_TARGET_COMPATIBLE,
)
