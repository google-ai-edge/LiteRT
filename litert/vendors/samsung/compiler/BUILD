# Copyright (C) 2026 Samsung Electronics Co. LTD.
# SPDX-License-Identifier: Apache-2.0
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
load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors/samsung/compiler:__subpackages__"],
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
    so_name = "libLiteRtCompilerPlugin_Samsung.so",
    ungrte = True,
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
    deps = [
        ":compile_model",
        ":create_model",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_common",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_options",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_opaque_options",
        "//litert/cc/dynamic_runtime/options:litert_samsung_options",
        "//litert/vendors/cc:options_helper",
        "//litert/vendors/samsung:ai_litecore_api",
        "//litert/vendors/samsung:soc_model",
    ],
)

cc_library(
    name = "create_model",
    srcs = ["create_model.cc"],
    hdrs = ["create_model.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_common",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_consts",
        "//litert/cc/internal:litert_extended_model",
        "//litert/vendors/samsung/compiler/builders:op_wrapper",
        "//litert/vendors/samsung/compiler/builders:batch_matmul_op_builder",
        "//litert/vendors/samsung/compiler/builders:cast_op_builder",
        "//litert/vendors/samsung/compiler/builders:concat_op_builder",
        "//litert/vendors/samsung/compiler/builders:conv2d_op_builder",
        "//litert/vendors/samsung/compiler/builders:elementwise_op_builder",
        "//litert/vendors/samsung/compiler/builders:logistic_op_builder",
        "//litert/vendors/samsung/compiler/builders:pad_op_builder",
        "//litert/vendors/samsung/compiler/builders:reshape_op_builder",
        "//litert/vendors/samsung/compiler/builders:softmax_op_builder",
        "//litert/vendors/samsung/compiler/builders:transpose_op_builder",
        "//litert/vendors/samsung:ai_litecore_api",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@exynos_ai_litecore//:ai_litecore_headers",
    ],
)

cc_library(
    name = "compile_model",
    srcs = ["compile_model.cc"],
    hdrs = ["compile_model.h"],
    deps = [
        "//litert/cc:litert_common",
        "//litert/cc:litert_expected",
        "//litert/vendors/samsung:ai_litecore_api",
        "@exynos_ai_litecore//:ai_litecore_headers",
    ],
)
