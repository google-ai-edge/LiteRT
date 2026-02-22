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
load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib", "litert_test", "make_rpaths")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors/mediatek/compiler:__subpackages__"],
)

litert_dynamic_lib(
    name = "compiler_plugin",
    srcs = ["compiler_plugin.cc"],
    hdrs = ["//litert/vendors/c:litert_compiler_plugin.h"],
    export_litert_only = True,
    linkopts = [
        # Wheel package places the shared library in the root directory.
        "-Wl,-rpath,$$ORIGIN/../../..",
    ],
    shared_lib_name = "compiler_plugin_so",
    so_name = "libLiteRtCompilerPlugin_MediaTek.so",
    ungrte = True,
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
        "//litert:litert_friends",
    ],
    deps = [
        ":compile_model",
        ":create_model",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_opaque_options",
        "//litert/cc/dynamic_runtime/options:litert_mediatek_options",
        "//litert/vendors/cc:options_helper",
        "//litert/vendors/mediatek:neuron_adapter_api",
        "//litert/vendors/mediatek/compiler/legalizations:common_op_legalization",
        "//litert/vendors/mediatek/compiler/legalizations:operand_map",
        "//litert/vendors/mediatek/schema:mediatek_litert_schema",
        "//litert/vendors/mediatek/schema:neuron_litert_schema",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "create_model",
    srcs = ["create_model.cc"],
    hdrs = ["create_model.h"],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
        "//litert/vendors/mediatek/compiler/legalizations:add_op_legalization",
        "//litert/vendors/mediatek/compiler/legalizations:batch_matmul_op_legalization",
        "//litert/vendors/mediatek/compiler/legalizations:common_op_legalization",
        "//litert/vendors/mediatek/compiler/legalizations:concat_op_legalization",
        "//litert/vendors/mediatek/compiler/legalizations:fully_connected_op_legalization",
        "//litert/vendors/mediatek/compiler/legalizations:gelu_op_legalization",
        "//litert/vendors/mediatek/compiler/legalizations:legalize_helper",
        "//litert/vendors/mediatek/compiler/legalizations:mean_op_legalization",
        "//litert/vendors/mediatek/compiler/legalizations:mul_op_legalization",
        "//litert/vendors/mediatek/compiler/legalizations:neuron_utils",
        "//litert/vendors/mediatek/compiler/legalizations:operand_map",
        "//litert/vendors/mediatek/compiler/legalizations:reshape_op_legalization",
        "//litert/vendors/mediatek/compiler/legalizations:resize_bilinear_op_legalization",
        "//litert/vendors/mediatek/compiler/legalizations:resize_nearest_neighbor_op_legalization",
        "//litert/vendors/mediatek/compiler/legalizations:rms_norm_op_legalization",
        "//litert/vendors/mediatek/compiler/legalizations:rsqrt_op_legalization",
        "//litert/vendors/mediatek/compiler/legalizations:softmax_op_legalization",
        "//litert/vendors/mediatek/compiler/legalizations:split_op_legalization",
        "//litert/vendors/mediatek/compiler/legalizations:squared_difference_op_legalization",
        "//litert/vendors/mediatek/compiler/legalizations:strided_slice_op_legalization",
        "//litert/vendors/mediatek/compiler/legalizations:sub_op_legalization",
        "//litert/vendors/mediatek/compiler/legalizations:transpose_conv_op_legalization",
        "//litert/vendors/mediatek/compiler/legalizations:transpose_op_legalization",
        "//litert/vendors/mediatek/schema:mediatek_litert_schema",
        "@neuro_pilot//:v8_latest_host_headers",
    ],
)

cc_library(
    name = "compile_model",
    srcs = ["compile_model.cc"],
    hdrs = ["compile_model.h"],
    local_defines = select({
        "@org_tensorflow//tensorflow:android": ["LITERT_USE_JIT=1"],
        "@org_tensorflow//tensorflow:chromiumos": ["LITERT_USE_JIT=1"],
        "//conditions:default": [],
    }),
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime/options:litert_mediatek_options",
        "//litert/vendors/mediatek:neuron_adapter_api",
        "@neuro_pilot//:v8_latest_host_headers",
    ],
)

litert_test(
    name = "compiler_plugin_test",
    srcs = [
        "compiler_plugin_test.cc",
    ],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": [],
        "//conditions:default": [
            make_rpaths([
                "//litert/c:litert_runtime_c_api_so",
            ]),
        ],
    }),
    linkstatic = True,
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        "nosan",
    ],
    # Currently this test can only be run on Android because we don't have x86 shared libraries for
    # MTK.
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        ":compiler_plugin",  # buildcleaner: keep
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_model_predicates",
        "//litert/core/model:model_testonly",
        "//litert/test:common",
        "//litert/test:matchers_oss",
        "//litert/test:test_models",
        "//litert/vendors/cc:litert_compiler_plugin",
        "//litert/vendors/mediatek:neuron_adapter_api",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
    ],
)
