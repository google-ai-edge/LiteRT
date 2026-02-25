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
#
# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib")
load("//litert/build_common:special_rule.bzl", "litert_android_linkopts")
load("//litert/integration_test:litert_device.bzl", "litert_device_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:private"],
)

litert_dynamic_lib(
    name = "dispatch_api",
    srcs = [
        "dispatch_api.cc",
        "litert_dispatch_device_context.cc",
        "litert_dispatch_invocation_context.cc",
    ],
    hdrs = [
        "litert_dispatch_device_context.h",
        "litert_dispatch_invocation_context.h",
        "registry.h",
    ],
    copts = [
        "-Os",
        "-fno-exceptions",
        "-fno-unwind-tables",
        "-fno-asynchronous-unwind-tables",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    export_litert_only = True,
    linkopts = litert_android_linkopts() + [
        "-Wl,-soname=libLiteRtDispatch_Qualcomm.so",
        "-Wl,-lc++abi",
    ],
    shared_lib_name = "dispatch_api_so",
    so_name = "libLiteRtDispatch_Qualcomm.so",
    visibility = ["//litert:litert_public"],
    deps = [
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//litert/c/internal:litert_logging",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc/dynamic_runtime/options:litert_qualcomm_options",
        "//litert/vendors/cc:options_helper",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_element_type",
        # TODO: Remove this dependency.
        "//litert/core/util:tensor_type_util",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//litert/vendors/qualcomm:common",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm:context_binary_info",
        "//litert/vendors/qualcomm:qnn_manager",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "//litert/c/internal:litert_scheduling_info",
        "//litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        "@qairt//:qnn_lib_headers",
    ],
)

litert_device_test(
    name = "dispatch_api_qualcomm_test",
    srcs = [
        "dispatch_api_qualcomm_test.cc",
    ],
    backend_id = "qualcomm",
    data = [
        "//litert/test:testdata/simple_add_uint16_qualcomm.bin",
        "//litert/test:testdata/simple_model_npu.tflite",
        "//litert/test:testdata/simple_model_qualcomm.bin",
    ],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_any",
        "//litert/cc:litert_environment_options",
        "//litert/cc/dynamic_runtime:litert_environment",
        "//litert/cc/dynamic_runtime:litert_options",
        "//litert/core:filesystem_testonly",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model_npu",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
