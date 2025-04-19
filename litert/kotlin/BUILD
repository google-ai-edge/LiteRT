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

load("@rules_kotlin//kotlin:android.bzl", "kt_android_library")
load(
    "//tflite:build_def.bzl",
    "tflite_jni_binary",
    "tflite_jni_linkopts",
    "tflite_linkopts_no_undefined",
)
load("//tflite/java:aar_with_jni.bzl", "aar_with_jni")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//third_party/odml/litert:__subpackages__",
    ],
)

cc_library(
    name = "litert_jni_common",
    hdrs = ["src/main/jni/litert_jni_common.h"],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "@com_google_absl//absl/cleanup",  # buildcleaner: keep
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
    ] + select({
        "@org_tensorflow//tensorflow:android": [],
        "//conditions:default": ["//third_party/java/jdk:jni"],
    }),
)

cc_library(
    name = "litert_environment_jni",
    srcs = ["src/main/jni/litert_environment_jni.cc"],
    hdrs = ["src/main/jni/litert_environment_jni.h"],
    deps = [
        ":litert_jni_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_environment",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:span",
    ] + select({
        "@org_tensorflow//tensorflow:android": [],
        "//conditions:default": ["//third_party/java/jdk:jni"],
    }),
)

cc_library(
    name = "litert_model_jni",
    srcs = ["src/main/jni/litert_model_jni.cc"],
    hdrs = ["src/main/jni/litert_model_jni.h"],
    deps = [
        ":litert_jni_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_buffer_ref",
    ] + select({
        "@org_tensorflow//tensorflow:android": [],
        "//conditions:default": ["//third_party/java/jdk:jni"],
    }),
)

cc_library(
    name = "litert_compiled_model_jni",
    srcs = ["src/main/jni/litert_compiled_model_jni.cc"],
    hdrs = ["src/main/jni/litert_compiled_model_jni.h"],
    deps = [
        ":litert_jni_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_handle",
        "//litert/cc:litert_tensor_buffer",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
    ] + select({
        "@org_tensorflow//tensorflow:android": [],
        "//conditions:default": ["//third_party/java/jdk:jni"],
    }),
)

cc_library(
    name = "litert_tensor_buffer_jni",
    srcs = ["src/main/jni/litert_tensor_buffer_jni.cc"],
    hdrs = ["src/main/jni/litert_tensor_buffer_jni.h"],
    deps = [
        ":litert_jni_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_handle",
        "//litert/cc:litert_tensor_buffer",
        "@com_google_absl//absl/types:span",
    ] + select({
        "@org_tensorflow//tensorflow:android": [],
        "//conditions:default": ["//third_party/java/jdk:jni"],
    }),
)

tflite_jni_binary(
    name = "litert_jni",
    linkopts = tflite_jni_linkopts() + tflite_linkopts_no_undefined(),
    linkscript = ":litert_version_script.lds",
    # Set stamp = 0 to avoid including internal info to the binary for the public release.
    stamp = 0,
    deps = [
        ":litert_compiled_model_jni",
        ":litert_environment_jni",
        ":litert_model_jni",
        ":litert_tensor_buffer_jni",
    ],
)

cc_library(
    name = "litert_jni_cc",
    srcs = [":litert_jni"],
)

# It's needed to include all the files in one target to be able to pack them in the aar.
kt_android_library(
    name = "litert_kotlin_api",
    srcs = [
        "src/main/kotlin/com/google/ai/edge/litert/AcceleratorProvider.kt",
        "src/main/kotlin/com/google/ai/edge/litert/Common.kt",
        "src/main/kotlin/com/google/ai/edge/litert/Environment.kt",
        "src/main/kotlin/com/google/ai/edge/litert/LiteRtException.kt",
        "src/main/kotlin/com/google/ai/edge/litert/Model.kt",
        "src/main/kotlin/com/google/ai/edge/litert/ModelProvider.kt",
        "src/main/kotlin/com/google/ai/edge/litert/TensorBuffer.kt",
        "src/main/kotlin/com/google/ai/edge/litert/deployment/AiPackModelProvider.kt",
    ],
    exports_manifest = 1,
    manifest = "src/main/AndroidManifest.xml",
    visibility = ["//visibility:public"],
    deps = [
        ":litert_jni_cc",  # buildcleaner: keep
        "@maven//:com_google_android_play_ai_delivery",
        "@maven//:com_google_guava_guava",
        "@maven//:androidx_lifecycle_lifecycle_common",
        "@maven//:org_jetbrains_kotlinx_kotlinx_coroutines_android",
        "@maven//:org_jetbrains_kotlinx_kotlinx_coroutines_guava",
        "@maven//:org_jetbrains_kotlinx_kotlinx_coroutines_play_services",
        # copybara:uncomment_begin(google-only)
        # "//third_party/odml/infra/ml_drift_delegate/litert:ml_drift_cl_accelerator_shared_lib",  # buildcleaner: keep
        # copybara:uncomment_end
        "//litert/c:litert_runtime_c_api_shared_lib",  # buildcleaner: keep
    ],
)

# bazel build --config=android_arm64 --android_ndk_min_sdk_version=26 \
#   //litert/kotlin:litert_kotlin_api_aar
aar_with_jni(
    name = "litert_kotlin_api_aar",
    android_library = ":litert_kotlin_api",
)
