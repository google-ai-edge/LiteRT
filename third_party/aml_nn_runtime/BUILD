# Copyright 2026 Amlogic, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# ABI selectors for @aml_npu_nn_runtime (used from aml_nn_runtime.BUILD).

load("@bazel_skylib//lib:selects.bzl", "selects")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

config_setting(
    name = "aml_flag_armeabi_v7a",
    values = {"cpu": "armeabi-v7a"},
)

config_setting(
    name = "aml_flag_arm64_v8a",
    values = {"cpu": "arm64-v8a"},
)

config_setting(
    name = "aml_flag_x86_64",
    values = {"cpu": "x86_64"},
)

config_setting(
    name = "aml_flag_k8",
    values = {"cpu": "k8"},
)

config_setting(
    name = "aml_cpu_armv7",
    constraint_values = ["@platforms//cpu:armv7"],
)

config_setting(
    name = "aml_cpu_arm64",
    constraint_values = ["@platforms//cpu:arm64"],
)

selects.config_setting_group(
    name = "aml_abi_armv7",
    match_any = [
        ":aml_flag_armeabi_v7a",
        ":aml_cpu_armv7",
    ],
)

selects.config_setting_group(
    name = "aml_abi_arm64",
    match_any = [
        ":aml_flag_arm64_v8a",
        ":aml_cpu_arm64",
    ],
)

selects.config_setting_group(
    name = "aml_abi_x86_64",
    match_any = [
        ":aml_flag_x86_64",
        ":aml_flag_k8",
    ],
)

# Compatibility aliases (old //third_party/aml_compiler_core:aml_abi_*).
alias(
    name = "aml_abi_arm64_compat",
    actual = ":aml_abi_arm64",
)
