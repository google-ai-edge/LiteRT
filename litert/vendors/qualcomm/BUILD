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

load("//litert/build_common:litert_build_defs.bzl", "litert_lib", "litert_test")
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
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/cc:litert_model",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
    ],
)

litert_lib(
    name = "qnn_log",
    srcs = ["qnn_log.cc"],
    hdrs = ["qnn_log.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
    ],
)

cc_library(
    name = "qnn_manager_hdr",
    hdrs = ["qnn_manager.h"],
    deps = [
        ":common",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_shared_library",
        "//litert/vendors/qualcomm/core/backends:htp_device_config",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
    ],
)

litert_cc_lib_with_qnn(
    name = "qnn_manager",
    srcs = [
        "qnn_manager.cc",
    ],
    hdrs = ["qnn_manager.h"],
    include_system = True,
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    ungrte = True,
    deps = [
        ":common",
        ":qnn_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_shared_library",
        "//litert/core:dynamic_loading",
        "//litert/vendors/qualcomm/core/backends:htp_device_config",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/backends:htp_perf_control",
        "//litert/vendors/qualcomm/core:common",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
    ],
)

litert_test(
    name = "qnn_manager_test",
    srcs = ["qnn_manager_test.cc"],
    linkstatic = True,
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        # Don't build/test in OS until qnn is available.
        "nobuilder",
        "no_oss",
        # Sanitizer runtime doesn't work with anything that loads libQnnHtp.so.
        "nosan",
    ],
    # This test can be run only on Android and Linux.
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":qnn_manager",
        "//litert/test:common",
        "//litert/test:matchers_oss",
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
        "@com_google_absl//absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
    ],
)

cc_library(
    name = "qnn_tensor",
    srcs = ["qnn_tensor.cc"],
    hdrs = ["qnn_tensor.h"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_expected",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
    ],
)
