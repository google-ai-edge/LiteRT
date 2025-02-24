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

load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//visibility:private"],
)

litert_dynamic_lib(
    name = "google_tensor_compiler_plugin",
    srcs = ["google_tensor_compiler_plugin.cc"],
    hdrs = ["//litert/vendors/c:litert_compiler_plugin.h"],
    export_litert_only = True,
    linkstatic = 1,
    shared_lib_name = "google_tensor_compiler_plugin_so",
    so_name = "libLiteRtCompilerPlugin_google_tensor.so",
    visibility = ["//litert:__subpackages__"],
    deps = [
        "//third_party/absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/core/model",
    ],
)

cc_test(
    name = "google_tensor_compiler_plugin_test",
    srcs = [
        "google_tensor_compiler_plugin_test.cc",
    ],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        # Don't build/test in OS until google tensor is available.
        "nobuilder",
        "no_oss",
        # Sanatizer runtime doesn't work with anything that loads a shared library.
        "nosan",
    ],
    deps = [
        ":google_tensor_compiler_plugin",  # buildcleaner: keep
        "//testing/base/public:gunit_main",
        "//third_party/absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_model",
        "//litert/core/model",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/vendors/c:litert_compiler_plugin",
        "//litert/vendors/cc:litert_compiler_plugin",
    ],
)
