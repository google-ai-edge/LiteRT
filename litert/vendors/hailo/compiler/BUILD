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

load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib", "litert_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

litert_dynamic_lib(
    name = "compiler_plugin",
    srcs = ["hailo_compiler_plugin.cc"],
    hdrs = [],
    copts = ["-fexceptions"],
    export_litert_only = True,
    features = ["-use_header_modules"],
    shared_lib_name = "hailo_compiler_plugin_so",
    so_name = "libLiteRtCompilerPlugin_Hailo.so",
    ungrte = True,
    deps = [
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_context_wrapper",
        "//litert/cc/internal:litert_handle",
        "//litert/compiler/cc:litert_model",
        "//litert/vendors/c:litert_compiler_plugin",
        "@com_google_absl//absl/strings:str_format",
    ],
)

litert_test(
    name = "hailo_compiler_plugin_test",
    srcs = [
        "hailo_compiler_plugin_test.cc",
    ],
    copts = ["-fexceptions"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    features = ["-use_header_modules"],
    linkstatic = True,
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        ":compiler_plugin",
        "//litert/c:litert_common",
        "//litert/c/internal:litert_compiler_context",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/test:common",
        "//litert/test:matchers_oss",
        "//litert/test:test_models",
        "//litert/vendors/cc:litert_compiler_plugin",
        "@com_google_absl//absl/strings",
    ],
)
