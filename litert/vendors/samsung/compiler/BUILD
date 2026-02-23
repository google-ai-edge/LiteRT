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
load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib", "litert_test", "make_rpaths")

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
        ":create_model",
        ":compile_model",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
	"//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_opaque_options",
        "//litert/cc/dynamic_runtime/options:litert_samsung_options",
        "//litert/vendors/cc:options_helper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@exynos_ai_litecore//:ai_litecore_headers",
    ],
)


cc_library(
    name = "create_model",
    srcs = ["create_model.cc"],
    hdrs = ["create_model.h"],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_model",
        "//litert/vendors/samsung/compiler/builders:add_op_builder",
        "//litert/vendors/samsung/compiler/builders:mul_op_builder",
        "//litert/vendors/samsung/compiler/builders:reshape_op_builder",
        "//litert/vendors/samsung:ai_litecore_api",
	"@com_google_absl//absl/container:inlined_vector",
	"@com_google_absl//absl/container:flat_hash_map",
	"@exynos_ai_litecore//:ai_litecore_headers"
    ],
)


cc_library(
    name = "compile_model",
    srcs = ["compile_model.cc"],
    hdrs = ["compile_model.h"],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_model",
        "//litert/cc/dynamic_runtime:litert_op_options",
        "//litert/vendors/samsung:ai_litecore_api",
        "@exynos_ai_litecore//:ai_litecore_headers",
    ],
)

