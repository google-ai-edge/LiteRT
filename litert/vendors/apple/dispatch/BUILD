# Copyright 2026 Google LLC.
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

load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//litert/build_common:litert_build_defs.bzl", "litert_dispatch_api")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//litert:__subpackages__"],
)

litert_dispatch_api(
    name = "dispatch_api",
    srcs = ["dispatch_api.cc"],
    hdrs = ["//litert/vendors/c:litert_dispatch.h"],
    export_litert_only = True,
    linkopts = select({
        "//litert:macos_x86_64": ["-lmlx"],
        "//litert:macos_arm64": ["-lmlx"],
        "//conditions:default": [],
    }),
    shared_lib_name = "dispatch_api_so",
    so_name = "libLiteRtDispatch_Apple.so",
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
        "//litert:linux_aarch64": [],
        "//litert:macos_x86_64": [],
        "//litert:macos_arm64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = [
        "//litert:__subpackages__",
        "//litert:litert_friends",
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_metrics",
        "//litert/c:litert_model_types",
        "//litert/c:litert_tensor_buffer",
        "//litert/c/internal:litert_logging",
        "//litert/c/internal:litert_runtime_context",
        "//litert/c/internal:litert_scheduling_info",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/vendors/apple:bytecode",
        "//litert/vendors/c:litert_dispatch_c_api",
    ],
)

cc_test(
    name = "dispatch_api_test",
    srcs = ["dispatch_api_test.cc"],
    data = [
        "//litert/test:testdata",
    ],
    deps = [
        ":dispatch_api",
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/c:litert_tensor_buffer",
        "//litert/c/internal:litert_compiler_context",
        "//litert/c/internal:litert_runtime_context_testonly",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model",
        "//litert/test:load_test_model_runtimecapi",
        "//litert/test:matchers",
        "//litert/test:simple_buffer_runtimecapi",
        "//litert/vendors/apple/compiler:compiler_plugin",
        "//litert/vendors/c:litert_compiler_plugin",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//litert/vendors/cc:litert_compiler_plugin",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
