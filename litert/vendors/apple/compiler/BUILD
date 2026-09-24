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
load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//litert:__subpackages__"],
)

litert_dynamic_lib(
    name = "compiler_plugin",
    srcs = ["compiler_plugin.cc"],
    hdrs = ["//litert/vendors/c:litert_compiler_plugin.h"],
    export_litert_only = True,
    shared_lib_name = "compiler_plugin_so",
    so_name = "libLiteRtCompilerPlugin_Apple.so",
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
        "//litert/c:litert_builder",
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/c:litert_op_options",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_tfl_types",
        "//litert/compiler/cc:litert_model",
        "//litert/vendors/apple:bytecode",
        "//litert/vendors/c:litert_compiler_plugin",
    ],
)

cc_test(
    name = "compiler_plugin_test",
    srcs = ["compiler_plugin_test.cc"],
    data = [
        "//litert/test:testdata",
    ],
    deps = [
        ":compiler_plugin",
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/c/internal:litert_compiler_context",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_model",
        "//litert/cc/internal:litert_extended_model",
        "//litert/test:common",
        "//litert/test:load_test_model",
        "//litert/test:matchers",
        "//litert/vendors/apple:bytecode",
        "//litert/vendors/c:litert_compiler_plugin",
        "//litert/vendors/cc:litert_compiler_plugin",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
