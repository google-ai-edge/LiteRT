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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

# copybara:uncomment(g3 only) load("@rules_cc//cc:cc_test.bzl", "cc_test")
load(
    "//litert/build_common:special_rule.bzl",
    "litert_android_linkopts",
    "litert_gpu_accelerator_deps",
    "litert_gpu_accelerator_prebuilts",
)

package(
    # copybara:uncomment(g3 only) default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "image_utils",
    srcs = ["image_utils.cc"],
    hdrs = ["image_utils.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        # copybara:uncomment(g3 only) "@stblib",
        "@stblib//:stb_image_hdrs",  # copybara:comment
    ],
)

cc_binary(
    name = "segmentation_example",
    srcs = ["segmentation_example.cc"],
    data = [
        "image.jpeg",
        "selfie_multiclass_256x256.tflite",
    ] + litert_gpu_accelerator_prebuilts(),
    linkopts = litert_android_linkopts(),
    tags = [],
    deps = [
        ":image_utils",
        "//litert/cc:litert_common",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_options",
        "//third_party/odml/litert/tensor",
        "//third_party/odml/litert/tensor:arithmetic",
        "//third_party/odml/litert/tensor:buffer",
        "//third_party/odml/litert/tensor:datatypes",
        "//third_party/odml/litert/tensor/backends/tflite:arithmetic_tflite",
        "//third_party/odml/litert/tensor/runners/litert:lambda_model_runner",
        "//third_party/odml/litert/tensor/runners/litert:litert_dynamic_runner",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ] + litert_gpu_accelerator_deps(),
)

# copybara:uncomment_begin(g3 only)
# cc_binary(
#     name = "segmentation_example_webgpu",
#     srcs = ["segmentation_example_webgpu.cc"],
#     data = [
#         "image.jpeg",
#         "selfie_multiclass_256x256.tflite",
#     ],
#     linkopts = litert_android_linkopts(),
#     deps = [
#         ":image_utils",
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/log:absl_check",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/types:span",
#         "@dawn//dawn:dawn_headers",
#         "@ml_drift//ml_drift/webgpu:execution_environment",
#         "@ml_drift//ml_drift/webgpu:webgpu_headers",
#         "//litert/cc:litert_common",
#         "//litert/cc:litert_environment",
#         "//litert/cc:litert_options",
#         "//third_party/odml/litert/tensor",
#         "//third_party/odml/litert/tensor:arithmetic",
#         "//third_party/odml/litert/tensor:buffer",
#         "//third_party/odml/litert/tensor:datatypes",
#         "//third_party/odml/litert/tensor/backends/webgpu:arithmetic_webgpu",
#         "//third_party/odml/litert/tensor/backends/webgpu:webgpu_conversion",
#         "//third_party/odml/litert/tensor/runners/litert:litert_dynamic_runner",
#         "//third_party/odml/litert/tensor/runners/webgpu:webgpu_runner",
#     ],
# )
#
# cc_test(
#     name = "segmentation_example_test",
#     srcs = ["segmentation_example.cc"],
#     data = [
#         "image.jpeg",
#         "segmented_output.png",
#         "selfie_multiclass_256x256.tflite",
#         "//litert/runtime/accelerators/gpu:ml_drift_cl_accelerator_so",
#     ],
#     linkopts = litert_android_linkopts(),
#     tags = ["requires-gpu-nvidia"],
#     deps = [
#         ":image_utils",
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/log:absl_check",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/time",
#         "@com_google_absl//absl/types:span",
#         "//litert/cc:litert_common",
#         "//litert/cc:litert_environment",
#         "//litert/cc:litert_options",
#         "//litert/runtime/accelerators/gpu:ml_drift_gpu_accelerator",
#         "//third_party/odml/litert/tensor",
#         "//third_party/odml/litert/tensor:arithmetic",
#         "//third_party/odml/litert/tensor:buffer",
#         "//third_party/odml/litert/tensor:datatypes",
#         "//third_party/odml/litert/tensor/backends/tflite:arithmetic_tflite",
#         "//third_party/odml/litert/tensor/runners/litert:lambda_model_runner",
#         "//third_party/odml/litert/tensor/runners/litert:litert_dynamic_runner",
#     ],
# )
# copybara:uncomment_end
