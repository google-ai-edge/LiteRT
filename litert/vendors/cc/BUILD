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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_public"],
)

cc_library(
    name = "litert_compiler_plugin",
    hdrs = ["litert_compiler_plugin.h"],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_macros",
        "//litert/vendors/c:litert_compiler_plugin",
    ],
)

cc_library(
    name = "options_helper",
    hdrs = ["options_helper.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_opaque_options",
        "//litert/cc/dynamic_runtime:litert_options",
        "//litert/cc/internal:litert_handle",
    ],
)

cc_library(
    name = "namespace_heuristics",
    srcs = ["namespace_heuristics.cc"],
    hdrs = ["namespace_heuristics.h"],
    deps = [
        "//litert/c:litert_op_code",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "options_helper_test",
    srcs = ["options_helper_test.cc"],
    deps = [
        ":options_helper",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_opaque_options",
        "//litert/cc/dynamic_runtime:litert_options",
        "//litert/cc/internal:litert_handle",
        "@com_google_googletest//:gtest_main",
    ],
)
