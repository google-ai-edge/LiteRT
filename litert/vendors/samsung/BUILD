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

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors/samsung:__subpackages__"],
)


cc_library(
    name = "ai_litecore_api",
    srcs = ["ai_litecore_manager.cc"],
    hdrs = ["ai_litecore_manager.h"],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_shared_library",
        "@com_google_absl//absl/strings:string_view",
        "@exynos_ai_litecore//:ai_litecore_headers",
    ],
)

cc_library(
    name = "soc_model",
    hdrs = ["soc_model.h"],
    srcs = ["soc_model.cc"],
    deps = [
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
    ]
)

