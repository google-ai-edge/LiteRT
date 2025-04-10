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

load(
    "//tools/build_defs/android:rules.bzl",
    "android_feature_module",
    "android_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//third_party/odml/litert:__subpackages__"],
)

android_library(
    name = "simple_model_qualcomm_v75",
    assets = [
        "simple_model_npu.tflite",
    ],
    assets_dir = "",
    custom_package = "com.google.ai.edge.litert.sample.dummy.npu_models",
)

android_library(
    name = "simple_model",
    assets = [
        "model#group_other/simple_model.tflite",
        "model#group_qti_v73/simple_model.tflite",
        "model#group_qti_v75/simple_model.tflite",
        "model#group_qti_v79/simple_model.tflite",
    ],
    assets_dir = "",
    custom_package = "com.google.ai.edge.litert.sample.dummy.npu_models",
)

android_feature_module(
    name = "simple_model_ai_pack",
    custom_package = "com.google.ai.edge.litert.sample.dummy.npu_models",
    feature_name = "simple_model_ai_pack",
    is_asset_pack = True,
    library = ":simple_model",
    manifest = "AiPackManifest.xml",
    title = "Simple model AI Model",
)
