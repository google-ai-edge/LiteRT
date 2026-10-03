# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//tflite/core/shims:cc_library_with_tflite.bzl", "java_test_with_tflite")

java_test_with_tflite(
    name = "compiled_model_resize_test",
    size = "small",
    srcs = ["java/com/google/ai/edge/litert/CompiledModelResizeTest.java"],
    data = [
        "//litert/test:testdata/dynamic_add_model.tflite",
        "//litert/test:testdata/reverse_signature_model.tflite",
        "//litert/test:testdata/simple_model.tflite",
    ],
    test_class = "com.google.ai.edge.litert.CompiledModelResizeTest",
    tflite_jni_binaries = ["//litert/kotlin:litert_jni"],
    deps = [
        "//litert/kotlin:litert_compiled_model_api_no_jni",
        "@com_google_truth",
        "@junit",
    ],
)
