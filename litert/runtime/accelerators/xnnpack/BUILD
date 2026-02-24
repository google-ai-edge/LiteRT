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

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

cc_library(
    name = "xnnpack_accelerator",
    srcs = ["xnnpack_accelerator.cc"],
    hdrs = ["xnnpack_accelerator.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_opaque_options",
        "//litert/c:litert_options",
        "//litert/c/internal:litert_accelerator_def",
        "//litert/c/internal:litert_delegate_wrapper",
        "//litert/c/options:litert_cpu_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/runtime:accelerator",
        "//litert/runtime/accelerators:accelerator_implementation_helper",
        "//tflite/c:c_api_types",
        "//tflite/delegates/xnnpack:xnnpack_delegate",
    ],
    alwayslink = 1,  # Keep the symbols in Windows and macOS.
)
