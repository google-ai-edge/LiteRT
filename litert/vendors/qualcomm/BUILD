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

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//litert/build_common:litert_build_defs.bzl", "litert_test", "make_rpaths")
load("//litert/integration_test:litert_device.bzl", "litert_device_exec")
load("//litert/vendors/qualcomm:qualcomm_build_defs.bzl", "litert_cc_lib_with_qnn")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

cc_library(
    name = "common",
    hdrs = ["common.h"],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_element_type",
        "//litert/cc/dynamic_runtime/options:litert_qualcomm_options",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/utils:log",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_library(
    name = "qnn_manager_hdr",
    hdrs = ["qnn_manager.h"],
    deps = [
        ":common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_shared_library",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/backends:qnn_backend",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@qairt//:qnn_lib_headers",
    ],
)

litert_cc_lib_with_qnn(
    name = "qnn_manager",
    srcs = [
        "qnn_manager.cc",
    ],
    hdrs = ["qnn_manager.h"],
    include_system = True,
    ungrte = True,
    deps = [
        ":common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_shared_library",
        "//litert/core:dynamic_loading",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/backends:htp_backend",
        "//litert/vendors/qualcomm/core/backends:ir_backend",
        "//litert/vendors/qualcomm/core/backends:qnn_backend",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@qairt//:qnn_lib_headers",
    ],
)

litert_test(
    name = "qnn_manager_test",
    srcs = ["qnn_manager_test.cc"],
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
        "no-remote-exec",
        "nosan",
        "notap",
    ],
    # This test can be run only on Android and Linux.
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        ":qnn_manager",
        "//litert/test:common",
        "//litert/test:matchers_oss",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/tools:dump",
    ],
)

litert_device_exec(
    name = "qnn_manager_device_test",
    backend_id = "qualcomm",
    target = ":qnn_manager_test",
)

cc_library(
    name = "context_binary_info",
    srcs = ["context_binary_info.cc"],
    hdrs = ["context_binary_info.h"],
    deps = [
        ":qnn_manager",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        "@qairt//:qnn_lib_headers",
    ],
)

build_test(
    name = "build_so_test",
    targets = [
        "//litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so",
        "//litert/vendors/qualcomm/dispatch:dispatch_api_so",
    ],
)
