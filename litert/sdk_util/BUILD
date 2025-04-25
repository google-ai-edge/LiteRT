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
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = [
        "//litert:litert_stable_abi_users",
    ],
)

# BUILD CONFIGURATIONS #############################################################################

# MEDIATEK SDK -------------------------------------------------------------------------------------

string_flag(
    name = "mtk_sdk_version",
    build_setting_default = "latest",
    values = [
        "v8",
        "v7",
        "latest",
    ],
)

config_setting(
    name = "mtk_sdk_v7",
    flag_values = {
        ":mtk_sdk_version": "v7",
    },
)

config_setting(
    name = "_mtk_sdk_latest_flag",
    flag_values = {
        ":mtk_sdk_version": "latest",
    },
    visibility = ["//visibility:private"],
)

config_setting(
    name = "_mtk_sdk_v8_flag",
    flag_values = {
        ":mtk_sdk_version": "v8",
    },
    visibility = ["//visibility:private"],
)

selects.config_setting_group(
    name = "mtk_sdk_v8",
    match_any = [
        ":_mtk_sdk_v8_flag",
        ":_mtk_sdk_latest_flag",
    ],
)

selects.config_setting_group(
    name = "mtk_sdk_v8_host",
    match_all = [
        "@org_tensorflow//tensorflow:linux_x86_64",
        ":mtk_sdk_v8",
    ],
)

selects.config_setting_group(
    name = "mtk_sdk_v7_host",
    match_all = [
        "@org_tensorflow//tensorflow:linux_x86_64",
        ":mtk_sdk_v7",
    ],
)

selects.config_setting_group(
    name = "mtk_sdk_v8_android",
    match_all = [
        "@org_tensorflow//tensorflow:android",
        ":mtk_sdk_v8",
    ],
)

selects.config_setting_group(
    name = "mtk_sdk_v7_android",
    match_all = [
        "@org_tensorflow//tensorflow:android",
        ":mtk_sdk_v7",
    ],
)
