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

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag", "string_flag")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

bool_flag(
    name = "ungrte",
    build_setting_default = True,
    visibility = ["//visibility:public"],
)

string_flag(
    name = "build_include",
    build_setting_default = "gpu,npu",
    values = [
        "gpu",
        "npu",
        "gpu,npu",
        "cpu_only",
    ],
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

config_setting(
    name = "build_include_gpu",
    flag_values = {":build_include": "gpu"},
)

config_setting(
    name = "build_include_npu",
    flag_values = {":build_include": "npu"},
)

config_setting(
    name = "build_include_gpu_npu",
    flag_values = {":build_include": "gpu,npu"},
)

config_setting(
    name = "build_include_cpu_only",
    flag_values = {":build_include": "cpu_only"},
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

selects.config_setting_group(
    name = "build_include_gpu_enabled",
    match_any = [
        ":build_include_gpu",
        ":build_include_gpu_npu",
    ],
)

selects.config_setting_group(
    name = "build_include_npu_enabled",
    match_any = [
        ":build_include_npu",
        ":build_include_gpu_npu",
    ],
)

selects.config_setting_group(
    name = "android_gpu_enabled",
    match_all = [
        "@platforms//os:android",
        ":build_include_gpu_enabled",
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

genrule(
    name = "build_config_header",
    srcs = select({
        ":build_include_gpu": ["config/build_config_gpu.h"],
        ":build_include_npu": ["config/build_config_npu.h"],
        ":build_include_gpu_npu": ["config/build_config_gpu_npu.h"],
        ":build_include_cpu_only": ["config/build_config_cpu_only.h"],
        "//conditions:default": ["config/build_config_gpu_npu.h"],
    }),
    outs = ["build_config.h"],
    cmd = "cp $< $@",
)

cc_library(
    name = "build_config",
    hdrs = [":build_config_header"],
    include_prefix = "third_party/odml/litert/litert/build_common",
    strip_include_prefix = "",
    visibility = ["//visibility:public"],
)
