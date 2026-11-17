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

# Model tools that are kept with the NVIDIA backend but do not depend on it:
# nothing here needs CUDA, TensorRT or a GPU.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//litert:__subpackages__"],
)

# Region patterns and their matcher: no file I/O, usable from a compiler
# plugin.
cc_library(
    name = "carve_pattern",
    srcs = [
        "carve_builtin_patterns.cc",
        "carve_pattern.cc",
    ],
    hdrs = ["carve_pattern.h"],
    deps = [
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/core/model",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
    ],
)

cc_library(
    name = "carve_model",
    srcs = ["carve_model.cc"],
    hdrs = ["carve_model.h"],
    deps = [
        ":carve_pattern",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/core:build_stamp",
        "//litert/core/model",
        "//litert/core/model:model_load",
        "//litert/core/util:flatbuffer_tools",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
    ],
)

# The checks on a carved model: structure from the two files, and a run of
# both on the CPU.
cc_library(
    name = "carve_verify",
    srcs = ["carve_verify.cc"],
    hdrs = ["carve_verify.h"],
    deps = [
        ":carve_model",
        "//litert/cc:litert_buffer_ref",
        "//litert/core/util:flatbuffer_tools",
        "//tflite:framework",
        "//tflite/c:common",
        "//tflite/delegates/xnnpack:xnnpack_delegate",
        "//tflite/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@flatbuffers",
    ],
)

# Reading and writing .tflite files and LiteRT-LM containers.
cc_library(
    name = "carve_container",
    srcs = ["carve_container.cc"],
    hdrs = ["carve_container.h"],
    deps = [
        ":carve_model",
        "//litert/cc:litert_buffer_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

# The tool: carve --model=IN --output=OUT (the guide is in carve_main.cc).
cc_binary(
    name = "carve",
    srcs = ["carve_main.cc"],
    deps = [
        ":carve_container",
        ":carve_model",
        ":carve_pattern",
        ":carve_verify",
        "//litert/cc:litert_buffer_ref",
        "//tflite:framework",
        "//tflite:stderr_reporter",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:usage",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "carve_test_util",
    testonly = True,
    srcs = ["carve_test_util.cc"],
    hdrs = ["carve_test_util.h"],
    deps = [
        ":carve_model",
        "//litert/cc:litert_buffer_ref",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)

cc_test(
    name = "carve_pattern_test",
    srcs = ["carve_pattern_test.cc"],
    deps = [
        ":carve_pattern",
        ":carve_test_util",
        "//litert/c:litert_op_code",
        "//litert/core/model",
        "//litert/core/model:model_load",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_test(
    name = "carve_model_test",
    srcs = ["carve_model_test.cc"],
    deps = [
        ":carve_model",
        ":carve_pattern",
        ":carve_test_util",
        ":carve_verify",
        "//litert/core/model",
        "//litert/core/model:model_load",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_test(
    name = "carve_container_test",
    srcs = ["carve_container_test.cc"],
    deps = [
        ":carve_container",
        ":carve_model",
        ":carve_pattern",
        ":carve_test_util",
        "//litert/cc:litert_buffer_ref",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)
