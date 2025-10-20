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

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//litert/vendors/mediatek:mediatek_build_defs.bzl", "litert_cc_lib_with_mtk")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

litert_cc_lib_with_mtk(
    name = "neuron_adapter_api",
    srcs = [
        "neuron_adapter_api.cc",
    ],
    hdrs = [
        "neuron_adapter_api.h",
    ],
    ungrte = True,
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime/options:litert_mediatek_options",
        "//litert/cc/internal:litert_shared_library",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@neuro_pilot//:v8_latest_host_headers",
    ],
)

build_test(
    name = "build_so_test",
    targets = [
        "//litert/vendors/mediatek/dispatch:dispatch_api_so",
        "//litert/vendors/mediatek/compiler:compiler_plugin_so",
    ],
)
