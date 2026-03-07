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
        "//litert/c:litert_common",
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
        "//litert/c:litert_common",
        "//litert/c/internal:litert_accelerator",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_any",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_environment_options",
        "//tflite/java/jni",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "litert_model_wrapper",
    hdrs = ["src/main/jni/litert_model_wrapper.h"],
    deps = [
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_compiled_model",
    ],
)

cc_library(
    name = "litert_compiled_model_jni",
    srcs = ["src/main/jni/litert_compiled_model_jni.cc"],
    hdrs = ["src/main/jni/litert_compiled_model_jni.h"],
    deps = [
        ":litert_jni_common",
        ":litert_model_wrapper",
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_options",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc:litert_tensor_buffer",
        "//litert/cc:litert_tensor_buffer_requirements",
        "//litert/cc/internal:litert_handle",
        "//litert/cc/options:litert_cpu_options",
        "//litert/cc/options:litert_gpu_options",
        "//litert/cc/options:litert_qualcomm_options",
        "//tflite/java/jni",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "litert_tensor_buffer_jni",
    srcs = ["src/main/jni/litert_tensor_buffer_jni.cc"],
    hdrs = ["src/main/jni/litert_tensor_buffer_jni.h"],
    deps = [
        ":litert_jni_common",
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_tensor_buffer",
        "//tflite/java/jni",
        "@com_google_absl//absl/types:span",
    ],
)

# TODO(b/409850505): Unify the JNI build target with the OSS version.
# copybara:uncomment_begin(google-only)
# android_jni_library(
#     name = "litert_jni_lib",
#     binary_name = "libLiteRt.so",
#     force_android = True,
#     lds = ":litert_version_script.lds",
#     linkopts = tflite_jni_linkopts() + tflite_linkopts_no_undefined() + litert_android_linkopts(),
#     deps = [
#         ":litert_compiled_model_jni",
#         ":litert_environment_jni",
#         ":litert_tensor_buffer_jni",
#         "//litert/c:litert_tflite_runtime_c_api_so_shim",  # runtime
#         "//tflite/java/src/main/native",  # TFLite JNI
#     ],
# )
# copybara:uncomment_end

# copybara:comment_begin(oss-only)
tflite_jni_binary(
    name = "LiteRt",
    linkopts = tflite_jni_linkopts() + tflite_linkopts_no_undefined() + litert_android_linkopts(),
    linkscript = ":litert_version_script.lds",
    # Set stamp = 0 to avoid including internal info to the binary for the public release.
    stamp = 0,
    deps = [
        ":litert_compiled_model_jni",
        ":litert_environment_jni",
        ":litert_tensor_buffer_jni",
        "//litert/c:litert_tflite_runtime_c_api_so_shim",  # runtime
        "//tflite/java/src/main/native",  # TFLite JNI
    ],
)

cc_library(
    name = "litert_jni_lib",
    srcs = [":LiteRt"],
)
# copybara:comment_end

# Bazel kt_android_library does not support cc_library as a dependency, so android_library is used
# as a workaround.
android_library(
    name = "litert_kotlin_api_native_deps",
    exports = [
        ":litert_jni_lib",
        # copybara:uncomment "//litert/runtime/accelerators/gpu:ml_drift_cl_gl_accelerator_shared_lib",
    ],
)

# TFLite Java sources w/o any changes.
TFLITE_JAVA_SRCS_COPIED = [
    # go/keep-sorted start
    "src/copied_from_tflite/java/org/tensorflow/lite/DataType.java",
    "src/copied_from_tflite/java/org/tensorflow/lite/DataTypeUtils.java",
    "src/copied_from_tflite/java/org/tensorflow/lite/InterpreterFactory.java",
    "src/copied_from_tflite/java/org/tensorflow/lite/NativeInterpreterWrapperExperimental.java",
    "src/copied_from_tflite/java/org/tensorflow/lite/NativeSignatureRunnerWrapper.java",
    "src/copied_from_tflite/java/org/tensorflow/lite/RuntimeFlavor.java",
    "src/copied_from_tflite/java/org/tensorflow/lite/Tensor.java",
    "src/copied_from_tflite/java/org/tensorflow/lite/TensorImpl.java",
    "src/copied_from_tflite/java/org/tensorflow/lite/annotations/UsedByReflection.java",
    "src/copied_from_tflite/java/org/tensorflow/lite/package-info.java",
    # go/keep-sorted end
]

[genrule(
    name = "copy_%s" % f.replace("/", "_").replace(".", "_"),
    srcs = ["//tflite/java:%s" % f.replace(
        "src/copied_from_tflite/java",
        "src/main/java",
    )],
    outs = [f],
    cmd = "cp $< $@",
    visibility = ["//visibility:private"],
) for f in TFLITE_JAVA_SRCS_COPIED]

# TFLite Java sources w/ changes.
TFLITE_JAVA_SRCS_COPIED_WITH_CHANGES = [
    # go/keep-sorted start
    "src/copied_from_tflite/java/org/tensorflow/lite/Interpreter.java",
    "src/copied_from_tflite/java/org/tensorflow/lite/InterpreterApi.java",
    "src/copied_from_tflite/java/org/tensorflow/lite/InterpreterFactoryApi.java",
    "src/copied_from_tflite/java/org/tensorflow/lite/InterpreterFactoryImpl.java",
    "src/copied_from_tflite/java/org/tensorflow/lite/InterpreterImpl.java",
    "src/copied_from_tflite/java/org/tensorflow/lite/NativeInterpreterWrapper.java",
    "src/copied_from_tflite/java/org/tensorflow/lite/TensorFlowLite.java",
    # go/keep-sorted end
]

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
    ] + TFLITE_JAVA_SRCS_COPIED + TFLITE_JAVA_SRCS_COPIED_WITH_CHANGES,
    exports_manifest = 1,
    manifest = "src/main/AndroidManifest.xml",
    proguard_specs = ["//tflite/java:proguard.flags"],
    visibility = ["//visibility:public"],
    deps = [
        ":litert_kotlin_api_native_deps",  # unuseddeps: keep
        "@litert_maven//:androidx_lifecycle_lifecycle_common",
        "@litert_maven//:com_google_android_play_ai_delivery",
        "@litert_maven//:com_google_guava_guava",
        "@litert_maven//:org_jetbrains_kotlinx_kotlinx_coroutines_android",
        "@litert_maven//:org_jetbrains_kotlinx_kotlinx_coroutines_guava",
        "@litert_maven//:org_jetbrains_kotlinx_kotlinx_coroutines_play_services",
        "@org_checkerframework_qual",  # TFLite JAVA deps.
    ],
)

# bazel build --config=android_arm64 --android_ndk_min_sdk_version=26 \
#   //litert/kotlin:litert
aar_with_jni(
    name = "litert",
    android_library = ":litert_kotlin_api",
)
