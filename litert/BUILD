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
        "//litert:__subpackages__",
    ],
)

# VISIBILITY #######################################################################################

package_group(
    name = "litert_internal_users",
    packages = [
        # Self packages.
        "//litert/c/...",
        "//litert/cc/...",
        "//litert/compiler/...",
        "//litert/core/...",
        "//litert/google/...",
        "//litert/integration_test/...",
        "//litert/kotlin/...",
        "//litert/python/...",
        "//litert/runtime/...",
        "//litert/test/...",
        "//litert/tools/...",
        # Users with access to core internal apis.
        # copybara:uncomment "//third_party/odml/infra/ml_drift_delegate/...",
        # copybara:uncomment "//third_party/odml/infra/genai/conversion/...",
        # copybara:uncomment "//third_party/darwinn/tflite/...",
        # copybara:uncomment "//platforms/darwinn/devtools/google_tensor_tpu_delegate_executor/...",
        # copybara:uncomment "//third_party/darwinn/litert/...",
    ],
)

package_group(
    name = "litert_stable_abi_users",
    includes = [
        ":litert_internal_users",
    ],
    packages = [
        # copybara:uncomment "INTERNAL_IMAGE_COLLECTION_INTEGRATION/...",
        # copybara:uncomment "@mediapipe//mediapipe/calculators/tensor/...",
        # copybara:uncomment "//third_party/odml/infra/genai/inference/...",
        "//litert/vendors/...",
        # copybara:uncomment "//third_party/odml/litert/llm/...",
        # copybara:uncomment "//third_party/odml/litert_lm/...",
    ],
)

package_group(
    name = "litert_python_internal_users",
    packages = [
        "//litert/python/...",
        # copybara:uncomment "//third_party/py/ai_edge_torch/aot/...",
    ],
)

package_group(
    name = "litert_python_users",
    includes = [
        ":litert_internal_users",
    ],
    packages = [],
)

# BUILD CONFIGURATIONS #############################################################################

# MEDIATEK SDK -------------------------------------------------------------------------------------

# copybara:uncomment_begin(google-only)
# # Reccomend putting the following on your .bazelrc:
# # build --flag_alias=mtk_sdk_version=//litert:mtk_sdk_version
# copybara:uncomment_end
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
