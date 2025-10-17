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

load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_kotlin//kotlin:android.bzl", "kt_android_library")
load("//litert/build_common:special_rule.bzl", "litert_android_linkopts")
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
        "//tflite/java/jni",
        "@com_google_absl//absl/cleanup",  # buildcleaner: keep
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "litert_environment_jni",
    srcs = ["src/main/jni/litert_environment_jni.cc"],
    hdrs = ["src/main/jni/litert_environment_jni.h"],
    deps = [
        ":litert_jni_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_any",
        "//litert/cc/dynamic_runtime:litert_environment",
        "//tflite/java/jni",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "litert_model_jni",
    srcs = ["src/main/jni/litert_model_jni.cc"],
    hdrs = [
        "src/main/jni/litert_model_jni.h",
        "src/main/jni/litert_model_wrapper.h",
    ],
    deps = [
        ":litert_jni_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_layout",
        "//litert/cc/dynamic_runtime:litert_model",
        "//tflite/java/jni",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "litert_compiled_model_jni",
    srcs = ["src/main/jni/litert_compiled_model_jni.cc"],
    hdrs = ["src/main/jni/litert_compiled_model_jni.h"],
    deps = [
        ":litert_jni_common",
        ":litert_model_jni",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc/dynamic_runtime:litert_compiled_model",
        "//litert/cc/dynamic_runtime:litert_tensor_buffer",
        "//litert/cc/internal:litert_handle",
        "//tflite/java/jni",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "litert_tensor_buffer_jni",
    srcs = ["src/main/jni/litert_tensor_buffer_jni.cc"],
    hdrs = ["src/main/jni/litert_tensor_buffer_jni.h"],
    deps = [
        ":litert_jni_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc/dynamic_runtime:litert_tensor_buffer",
        "//litert/cc/internal:litert_handle",
        "//tflite/java/jni",
        "@com_google_absl//absl/types:span",
    ],
)

# TODO(b/409850505): Unify the JNI build target with the OSS version.
# copybara:uncomment_begin(google-only)
# android_jni_library(
#     name = "litert_jni_lib",
#     binary_name = "liblitert_jni.so",
#     force_android = True,
#     linkopts = tflite_jni_linkopts() + tflite_linkopts_no_undefined() + litert_android_linkopts(),
#     deps = [
#         ":litert_compiled_model_jni",
#         ":litert_environment_jni",
#         ":litert_model_jni",
#         ":litert_tensor_buffer_jni",
#     ],
# )
# copybara:uncomment_end

# copybara:comment_begin(oss-only)
tflite_jni_binary(
    name = "litert_jni",
    linkopts = tflite_jni_linkopts() + tflite_linkopts_no_undefined() + litert_android_linkopts(),
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
    name = "litert_jni_lib",
    srcs = [":litert_jni"],
)
# copybara:comment_end

# Bazel kt_android_library does not support cc_library as a dependency, so android_library is used
# as a workaround.
android_library(
    name = "litert_kotlin_api_native_deps",
    exports = [
        ":litert_jni_lib",
        "//litert/c:litert_runtime_c_api_shared_lib",
        # copybara:uncomment "//litert/runtime/accelerators/gpu:ml_drift_cl_accelerator_shared_lib",
    ],
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
        ":litert_kotlin_api_native_deps",  # buildcleaner: keep
        "@litert_maven//:androidx_lifecycle_lifecycle_common",
        "@litert_maven//:com_google_android_play_ai_delivery",
        "@litert_maven//:com_google_guava_guava",
        "@litert_maven//:org_jetbrains_kotlinx_kotlinx_coroutines_android",
        "@litert_maven//:org_jetbrains_kotlinx_kotlinx_coroutines_guava",
        "@litert_maven//:org_jetbrains_kotlinx_kotlinx_coroutines_play_services",
    ],
)

# bazel build --config=android_arm64 --android_ndk_min_sdk_version=26 \
#   //litert/kotlin:litert
aar_with_jni(
    name = "litert",
    android_library = ":litert_kotlin_api",
)

# copybara:uncomment_begin(google-only)
# android_library(
#     name = "litert_kotlin_api_native_gpu_deps",
#     exports = [
#         "//litert/runtime/accelerators/gpu:ml_drift_gpu_accelerator_shared_lib",
#     ],
# )
#
# # Marvel target for GPU Accelerator which supports both WebGPU and OpenCL.
# aar_with_jni(
#     name = "litert_gpu_accelerator",
#     android_library = ":litert_kotlin_api_native_gpu_deps",
# )
# copybara:uncomment_end
