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
#
# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

load(
    "//litert/build_common:litert_build_defs.bzl",
    "litert_lib",
)

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:private",
    ],
    features = ["-layering_check"],  # buildifier: disable=no-layering-check
)

litert_lib(
    name = "fold_const_dequantize",
    srcs = ["fold_const_dequantize.cc"],
    hdrs = ["fold_const_dequantize.h"],
    visibility = ["//litert/vendors/qualcomm:__subpackages__"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/c/internal:litert_compiler_context_header",
        "//litert/cc:litert_macros",
        "//litert/compiler/cc:litert_builder",
        "//litert/compiler/cc:litert_matchers",
        "//litert/compiler/cc:litert_model",
        "//litert/compiler/cc:litert_op_options",
    ],
)
