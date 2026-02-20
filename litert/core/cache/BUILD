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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//litert:__subpackages__",
    ],
)

cc_library(
    name = "hash_util",
    hdrs = ["hash_util.h"],
)

cc_library(
    name = "compilation_cache",
    srcs = ["compilation_cache.cc"],
    hdrs = ["compilation_cache.h"],
    deps = [
        ":hash_util",
        "//litert/c:litert_common",
        "//litert/c:litert_opaque_options",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/compiler/plugin:compiler_plugin",
        "//litert/core:filesystem",
        "//litert/core:options",
        "//litert/core/model",
        "//litert/core/model:model_load",
        "//litert/core/util:flatbuffer_tools",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "compilation_cache_test",
    srcs = ["compilation_cache_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    deps = [
        ":compilation_cache",
        "//litert/c:litert_common",
        "//litert/c:litert_environment",
        "//litert/c:litert_opaque_options",
        "//litert/c:litert_options",
        "//litert/c/options:litert_google_tensor_options",
        "//litert/cc:litert_macros",
        "//litert/core:environment",
        "//litert/core:options",
        "//litert/core/model",
        "//litert/core/model:model_load",
        "//litert/runtime:compiled_model",
        "//litert/test:common",
        "//litert/test:simple_model",
        "@com_google_googletest//:gtest_main",
    ],
)
