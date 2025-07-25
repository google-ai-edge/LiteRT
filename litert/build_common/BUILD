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

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

bool_flag(
    name = "ungrte",
    build_setting_default = True,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_grte",
    flag_values = {
        ":ungrte": "false",
    },
)

config_setting(
    name = "without_grte",
    flag_values = {
        ":ungrte": "true",
    },
)

selects.config_setting_group(
    name = "linux_x86_64_grte",
    match_all = [
        "@org_tensorflow//tensorflow:linux_x86_64",
        ":with_grte",
    ],
)

selects.config_setting_group(
    name = "linux_x86_64_ungrte",
    match_all = [
        "@org_tensorflow//tensorflow:linux_x86_64",
        ":without_grte",
    ],
)

exports_files(srcs = [
    "export_litert_only_darwin.lds",
    "export_litert_only_linux.lds",
    "export_litert_runtime_only_darwin.lds",
    "export_litert_runtime_only_linux.lds",
    "export_litert_tflite_runtime_darwin.lds",
    "export_litert_tflite_runtime_linux.lds",
])

bzl_library(
    name = "special_rule_bzl",
    srcs = ["special_rule.bzl"],
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "litert_build_defs_bzl",
    srcs = ["litert_build_defs.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        "//devtools/build_cleaner/skylark:build_defs_lib",
        "@bazel_skylib//lib:selects",
    ],
)

bzl_library(
    name = "tfl_model_gen_bzl",
    srcs = ["tfl_model_gen.bzl"],
    visibility = ["//visibility:private"],
)
