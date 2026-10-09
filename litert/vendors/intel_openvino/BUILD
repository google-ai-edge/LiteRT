# Copyright (C) 2025 Intel Corporation
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

# copybara:uncomment_begin(google-only)
# load("//tools/build_defs/build_test:build_test.bzl", "build_test")
# copybara:uncomment_end

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//litert:__subpackages__"],
)

# copybara:uncomment_begin(google-only)
# # Mirrors the android_x86_64 step of the Kokoro prebuilt_binaries job
# # (third_party/odml/kokoro/litert/generic/linux_x86_cpu/prebuilt_binaries/common.gcl).
# # Referenced from the `third_party.openvino.android_x86_64` TAP buildable unit so
# # that OpenVINO re-import CLs fail presubmit if these plugins stop building, instead
# # of only being caught post-submit by the litert Kokoro job (see cl/991463461).
# # Google-only: //tools/build_defs/build_test has no OSS equivalent. See
# # //third_party/openvino/BUILD for the flags to run this locally.
# build_test(
#     name = "prebuilt_plugins_build_test",
#     allow_empty_target = False,
#     targets = [
#         "//litert/vendors/intel_openvino/compiler:intel_openvino_compiler_plugin_so",
#         "//litert/vendors/intel_openvino/dispatch:dispatch_api_so",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "ov_utils",
    hdrs = ["utils.h"],
    deps = [
        "//litert/c:litert_model_types",
        "@intel_openvino//:openvino",
    ],
)

cc_library(
    name = "openvino_version_info",
    hdrs = ["openvino_version_info.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "@intel_openvino//:openvino",
    ],
)

cc_library(
    name = "bytecode_header",
    hdrs = ["bytecode_header.h"],
    deps = [
        "//litert/c/options:litert_intel_openvino_options",
    ],
)
