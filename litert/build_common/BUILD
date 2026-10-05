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
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//litert/build_common:special_rule.bzl", "get_compatible_with_portable", "litert_platform_defines")

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
    # LINT.IfChange(string_flag_default)
    build_setting_default = "gpu,npu",
    # LINT.ThenChange(./BUILD:build_config_header_default)
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

# Selects the kernels used by the CPU backend.
#   xnnpack: XNNPACK accelerator, with TFLite builtin kernels as fallback.
#   builtin: TFLite builtin kernels only. XNNPACK is not linked.
#   selective: Application-selected kernels only. Neither XNNPACK nor the full
#              TFLite builtin kernel library is linked.
# Usage: --//litert/build_common:cpu_backend=builtin
string_flag(
    name = "cpu_backend",
    build_setting_default = "xnnpack",
    values = [
        "xnnpack",
        "builtin",
        "selective",
    ],
    visibility = ["//visibility:public"],
)

# Replace the default all-ops resolver at link time for a fixed set of models.
# See litert_selected_op_resolver() in selective_ops.bzl.
label_flag(
    name = "op_resolver",
    build_setting_default = "//litert/runtime:default_op_resolver",
    visibility = ["//visibility:public"],
)

# Allow unused C APIs to be stripped from statically linked C++ applications.
# Explicitly supplied system runtimes still use their own function table.
bool_flag(
    name = "cc_static_dispatch",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "cc_static_dispatch_enabled",
    flag_values = {":cc_static_dispatch": "true"},
)

config_setting(
    name = "cc_static_dispatch_dynamic",
    define_values = {"litert_runtime_link_mode": "dynamic"},
    flag_values = {":cc_static_dispatch": "true"},
)

config_setting(
    name = "cc_static_dispatch_none",
    define_values = {"litert_runtime_link_mode": "none"},
    flag_values = {":cc_static_dispatch": "true"},
)

config_setting(
    name = "cpu_backend_builtin",
    flag_values = {":cpu_backend": "builtin"},
)

# Selective op registration mode: omits both the XNNPACK CPU accelerator and the
# default full TFLite builtin kernel library (`builtin_ops` / `reference_ops`).
# The application is responsible for linking and registering only the ops
# needed by its model (e.g. via TFLite's `gen_selected_ops`).
config_setting(
    name = "cpu_backend_selective",
    flag_values = {":cpu_backend": "selective"},
)

# Legacy way to leave out XNNPACK. Relies on the same definition with TFLite to
# disable the XNNPACK CPU backend on the TFLite Interpreter.
config_setting(
    name = "tflite_without_xnnpack",
    define_values = {"tflite_with_xnnpack": "false"},
)

# Leaves out the XNNPACK CPU accelerator (defines LITERT_DISABLE_CPU). CPU
# execution still works through the TFLite builtin kernels (or selected ops).
selects.config_setting_group(
    name = "litert_disable_cpu",
    match_any = [
        ":cpu_backend_builtin",
        ":cpu_backend_selective",
        ":tflite_without_xnnpack",
    ],
)

# Enables YNNPACK. Equivalent to `--define=litert_enable_ynnpack=true`, but can also be set from a
# Starlark transition (transitions on `--define` are not supported).
bool_flag(
    name = "enable_ynnpack",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "litert_enable_ynnpack_define",
    define_values = {"litert_enable_ynnpack": "true"},
)

config_setting(
    name = "litert_enable_ynnpack_flag",
    flag_values = {":enable_ynnpack": "true"},
)

# Matches if YNNPACK is enabled with either `--define=litert_enable_ynnpack=true` or
# `--//litert/build_common:enable_ynnpack`.
selects.config_setting_group(
    name = "litert_enable_ynnpack",
    match_any = [
        ":litert_enable_ynnpack_define",
        ":litert_enable_ynnpack_flag",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "build_include_gpu",
    flag_values = {":build_include": "gpu"},
)

config_setting(
    name = "build_include_gpu_define",
    define_values = {"litert_build_include": "gpu"},
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
        "//litert:linux_x86_64",
        ":with_grte",
    ],
)

selects.config_setting_group(
    name = "linux_x86_64_ungrte",
    match_all = [
        "//litert:linux_x86_64",
        ":without_grte",
    ],
)

selects.config_setting_group(
    name = "build_include_gpu_enabled",
    match_any = [
        ":build_include_gpu",
        ":build_include_gpu_define",
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
        "//litert:android",
        ":build_include_gpu_enabled",
    ],
)

selects.config_setting_group(
    name = "emscripten_gpu_enabled",
    match_all = [
        "//litert:emscripten",
        ":build_include_gpu_enabled",
    ],
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "linux_gpu_enabled",
    match_all = [
        "//litert:linux",
        ":build_include_gpu_enabled",
    ],
)

selects.config_setting_group(
    name = "macos_gpu_enabled",
    match_all = [
        "//litert:macos",
        ":build_include_gpu_enabled",
    ],
)

selects.config_setting_group(
    name = "ios_gpu_enabled",
    match_all = [
        "//litert:ios",
        ":build_include_gpu_enabled",
    ],
)

selects.config_setting_group(
    name = "windows_gpu_enabled",
    match_all = [
        "//litert:windows",
        ":build_include_gpu_enabled",
    ],
)

exports_files(srcs = [
    "export_litert_gpu_accelerator_darwin.lds",
    "export_litert_gpu_accelerator_linux.lds",
    "export_litert_only_darwin.lds",
    "export_litert_only_linux.lds",
    "export_litert_runtime_only_darwin.lds",
    "export_litert_runtime_only_linux.lds",
    "export_litert_tflite_runtime_darwin.lds",
    "export_litert_tflite_runtime_linux.lds",
])

copy_file(
    name = "build_config_header",
    src = select({
        ":build_include_gpu": "config/build_config_gpu.h",
        ":build_include_gpu_define": "config/build_config_gpu.h",
        ":build_include_npu": "config/build_config_npu.h",
        ":build_include_cpu_only": "config/build_config_cpu_only.h",
        # Note: build_include_gpu_npu is removed to avoid illegal ambiguous match error b/w
        # build_include_gpu_define (define flag) and build_include_gpu_npu (string flag w/ default).
        # LINT.IfChange(build_config_header_default).
        "//conditions:default": "config/build_config_gpu_npu.h",
        # LINT.ThenChange(./BUILD:string_flag_default)
    }),
    out = "build_config.h",
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "build_config",
    hdrs = [":build_config_header"],
    compatible_with = get_compatible_with_portable(),
    defines = litert_platform_defines(),
    include_prefix = "third_party/odml/litert/litert/build_common",
    strip_include_prefix = "",
    visibility = ["//visibility:public"],
)

# Files that make up the LiteRT C++ SDK.
#
# lint IfChange(litert_cc_sdk_files)
filegroup(
    name = "litert_cc_sdk_files",
    srcs = [
        "build_config.h.in",
    ],
    visibility = ["//litert/cc_sdk:__pkg__"],
)
# lint ThenChange(litert/cc_sdk/litert_cc_sdk_hdrs.txt, litert/cc_sdk/litert_cc_sdk_srcs.txt)
