# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tflite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")
load("//tflite:special_rules.bzl", "nnapi_sl_visibility_allowlist")

_DEFAULT_VISIBILITY = ["//tflite:__subpackages__"]

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = _DEFAULT_VISIBILITY,
    licenses = ["notice"],
)

cc_library(
    name = "nnapi_support_library_headers",
    hdrs = [
        "include/SupportLibrary.h",
        "include/SupportLibrarySymbols.h",
        "public/NeuralNetworksSupportLibraryImpl.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = _DEFAULT_VISIBILITY + nnapi_sl_visibility_allowlist(),
    deps = [
        "//tflite/kernels/internal:compatibility",
        "//tflite/nnapi:nnapi_lib",
    ],
)

cc_library(
    name = "nnapi_support_library",
    srcs = [
        "SupportLibrary.cc",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = _DEFAULT_VISIBILITY + nnapi_sl_visibility_allowlist(),
    deps = [
        ":nnapi_support_library_headers",
        "//tflite:minimal_logging",
        "//tflite/nnapi:nnapi_lib",
    ],
)
