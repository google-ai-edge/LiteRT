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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

cc_library(
    name = "outliner_util",
    srcs = ["outliner_util.cc"],
    hdrs = ["outliner_util.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_expected",
        "//litert/compiler/plugin:algo",
        "//litert/core/model",
        "//tflite/converter/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@flatbuffers//include/flatbuffers:flexbuffers",
    ],
)

cc_binary(
    name = "outliner",
    srcs = ["outliner_main.cc"],
    deps = [
        ":outliner_util",
        "//litert/c:litert_op_code",
        "//litert/core/model",
        "//litert/core/model:model_load",
        "//litert/core/model:model_serialize",
        "//litert/tools:dump",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
)
