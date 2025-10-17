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

"""
BUILD file for LiteRt SDK release artifacts.
"""

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("//litert/build_common:litert_build_defs.bzl", "litert_test", "make_rpaths")

# copybara:uncomment_begin(google-only)
# load("//litert/sdk_util:sdk_release.bzl", "sdk_release_tar")
# copybara:uncomment_end

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = [
        "//litert:litert_public",
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

# SDK ARTIFACT PREPERATION #########################################################################

litert_test(
    name = "check_sdk_deps_test",
    srcs = ["check_sdk_deps_test.cc"],
    data = select({
        "@org_tensorflow//tensorflow:linux_x86_64": [
            # copybara:uncomment "@neuro_pilot//:v8_latest/host/lib/libneuron_adapter.so",
            "@qairt//:lib/x86_64-linux-clang/libQnnHtp.so",
            "@qairt//:lib/x86_64-linux-clang/libQnnSystem.so",
        ],
        "//conditions:default": [],
    }),
    linkopts = ["-ldl"] + select({
        "@org_tensorflow//tensorflow:linux_x86_64": [
            make_rpaths(["@qairt//:lib/x86_64-linux-clang/libQnnHtp.so"]),
            # copybara:uncomment make_rpaths(["@neuro_pilot//:v8_latest/host/lib/libneuron_adapter.so"]),
        ],
        "//conditions:default": [],
    }) + ["-Wl,--disable-new-dtags"],
    linkstatic = True,
    tags = [
        "no-remote-exec",
        "notap",
    ],
    ungrte = True,
    deps = [
        "//litert/cc/internal:litert_logging",
        "//litert/test:common",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@neuro_pilot//:v8_latest_host_headers",
        "@qairt//:qnn_lib_headers",
    ],
)

# copybara:uncomment_begin(google-only)
#
# # QUALCOMM SDK RELEASE -----------------------------------------------------------------------------
#
# # Build the tar, copy to /tmp, check sizes, etc.
# # buildifier: disable=no-effect
# """
# bzl build -c opt //litert/sdk_util:litert_qualcomm_sdk_release && \
#    cp -f $(bzl info -c opt bzl-genfiles)/third_party/odml/litert/litert/sdk_util/litert_qualcomm_sdk_release.tar.gz /tmp && \
#    tar -xzvf /tmp/litert_qualcomm_sdk_release.tar.gz -C /tmp/ --one-top-level && \
#    echo "---------- Total Size ----------" && \
#    du -h /tmp/litert_qualcomm_sdk_release -d 0 && \
#    du -h /tmp/litert_qualcomm_sdk_release.tar.gz && \
#    echo "---------- Breakdown ----------" && \
#    du -h $(find /tmp/litert_qualcomm_sdk_release -type d -links 2) && \
#    rm -rf /tmp/litert_qualcomm_sdk_release/
# """
#
# sdk_release_tar(
#     name = "litert_qualcomm_sdk_release",
#     srcs = [
#         "@qairt//:qnn_htp_latest_files",
#     ],
#     strip_prefix = "third_party/qairt",
# )
#
# copybara:uncomment_end
