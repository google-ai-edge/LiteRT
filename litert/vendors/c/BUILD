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
    default_visibility = ["//litert:__subpackages__"],
)

cc_library(
    name = "litert_compiler_plugin",
    hdrs = ["litert_compiler_plugin.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options",
        "//litert/c:litert_model",
        "//litert/c:litert_options",
        "//litert/c:litert_rewriter",
    ],
)

cc_library(
    name = "litert_compiler_plugin_api",
    hdrs = ["litert_compiler_plugin_api.h"],
    deps = [
        ":litert_compiler_plugin",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options",
        "//litert/c:litert_model",
        "//litert/c:litert_options",
        "//litert/c:litert_rewriter",
        "@com_google_absl//absl/strings:string_view",
    ],
)

# This library is used to build the C API header files for the vendor dispatch API.
# All the vendor dispatch .so tragets should depend on this library.
cc_library(
    name = "litert_dispatch_c_api",
    hdrs = [
        "litert_dispatch.h",
        "litert_dispatch_api.h",
    ],
    deps = [
        # only depend on the headers, not the implementation.
        "//litert/c/internal:litert_dispatch_headers",
    ],
)

# This test verifies that the C API header files can build via C compiler.
cc_test(
    name = "litert_vendor_c_api_common_test",
    srcs = ["litert_vendor_c_api_common_test.c"],
    copts = ["--std=c11"],
    linkopts = ["-ldl"],
    deps = [
        ":litert_compiler_plugin",
        ":litert_compiler_plugin_api",
        ":litert_dispatch_c_api",
    ],
)

exports_files(srcs = glob(["litert_*.h"]))
