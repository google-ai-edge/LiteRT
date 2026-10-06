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

load("//tools/build_defs/android:rules.bzl", "android_local_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//third_party/odml/litert:__subpackages__",
    ],
)

android_local_test(
    name = "compiled_model_int16_test",
    size = "small",
    srcs = ["java/com/google/ai/edge/litert/CompiledModelInt16Test.java"],
    data = ["//litert/test:testdata/single_add_default_a16w8_recipe_quantized.tflite"],
    manifest = "AndroidManifest.xml",
    test_class = "com.google.ai.edge.litert.CompiledModelInt16Test",
    deps = [
        "//litert/kotlin:litert_compiled_model_api",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

android_local_test(
    name = "compiled_model_resize_test",
    size = "small",
    srcs = ["java/com/google/ai/edge/litert/CompiledModelResizeTest.java"],
    data = [
        "//litert/test:testdata/dynamic_add_model.tflite",
        "//litert/test:testdata/reverse_signature_model.tflite",
        "//litert/test:testdata/simple_model.tflite",
    ],
    manifest = "AndroidManifest.xml",
    test_class = "com.google.ai.edge.litert.CompiledModelResizeTest",
    deps = [
        "//litert/kotlin:litert_compiled_model_api",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)
