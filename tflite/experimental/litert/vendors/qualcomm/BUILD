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

load("//tflite/experimental/litert/build_common:litert_build_defs.bzl", "litert_lib", "litert_test")
load("//tflite/experimental/litert/vendors/qualcomm:qualcomm_build_defs.bzl", "litert_cc_lib_with_qnn")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//tflite/experimental/litert:__subpackages__"],
)

cc_library(
    name = "common",
    hdrs = ["common.h"],
    deps = [
        # copybara:uncomment "@org_tensorflow//third_party/qairt/latest:qnn_lib_headers",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_model",
        "//tflite/experimental/litert/cc:litert_model",
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
        # copybara:uncomment "@org_tensorflow//third_party/qairt/latest:qnn_lib_headers",
    ],
)

cc_library(
    name = "qnn_manager_hdr",
    hdrs = ["qnn_manager.h"],
    deps = [
        ":common",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "@org_tensorflow//third_party/qairt/latest:qnn_lib_headers",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/cc:litert_macros",
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
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "@org_tensorflow//third_party/qairt/latest:qnn_lib_headers",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_logging",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/cc:litert_macros",
        "//tflite/experimental/litert/core:dynamic_loading",
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
    deps = [
        ":qnn_manager",
        "//tflite/experimental/litert/test:common",
        "//tflite/experimental/litert/vendors/qualcomm/tools:dump",
    ],
)

cc_library(
    name = "context_binary_info",
    srcs = ["context_binary_info.cc"],
    hdrs = ["context_binary_info.h"],
    deps = [
        ":qnn_manager",
        ":qnn_tensor",
        "@com_google_absl//absl/strings:string_view",
        # copybara:uncomment "@org_tensorflow//third_party/qairt/latest:qnn_lib_headers",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_logging",
        "//tflite/experimental/litert/cc:litert_expected",
    ],
)

cc_library(
    name = "qnn_tensor",
    srcs = ["qnn_tensor.cc"],
    hdrs = ["qnn_tensor.h"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        # copybara:uncomment "@org_tensorflow//third_party/qairt/latest:qnn_lib_headers",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_logging",
        "//tflite/experimental/litert/cc:litert_expected",
    ],
)
