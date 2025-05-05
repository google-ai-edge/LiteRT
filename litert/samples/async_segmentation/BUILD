# Copyright 2025 Google LLC.
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

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "shader_files",
    srcs = [
        "shaders/mask_blend_compute.glsl",
        "shaders/passthrough_shader.vert",
        "shaders/preprocess_compute.glsl",
        "shaders/resize_compute.glsl",
    ],
)

filegroup(
    name = "test_images",
    srcs = [
        "test_images/image.jpeg",
    ],
)

filegroup(
    name = "model_files",
    srcs = [
        "models/selfie_multiclass_256x256.tflite",
    ],
)

cc_binary(
    name = "async_segmentation",
    srcs = [
        "main.cc",
    ],
    copts = [
        "-I.",
    ],
    data = [
        ":model_files",
        ":shader_files",
        ":test_images",
        "//litert/c:litert_runtime_c_api_shared_lib",  # buildcleaner: keep
        "//litert/vendors/qualcomm/dispatch:dispatch_api_so",  # buildcleaner: keep
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        ":image_processor",
        ":image_utils",
        ":segmentation_model",
        "//litert/runtime/accelerators/gpu:ml_drift_cl_accelerator",  # buildcleaner: keep
        "//litert/vendors/qualcomm/dispatch:dispatch_api",  # buildcleaner: keep
        "//third_party/GL:GLES2_headers",
    ],
)

cc_library(
    name = "image_utils",
    srcs = ["image_utils.cc"],
    hdrs = ["image_utils.h"],
    deps = [
        "//third_party/stblib",
    ],
)

cc_library(
    name = "image_processor",
    srcs = ["image_processor.cc"],
    hdrs = ["image_processor.h"],
    deps = [
        "//third_party/GL:EGL_headers",
        "//third_party/GL:GLES3_headers",
        "//third_party/GL/native:EGL",
        "//third_party/GL/native:GLESv3",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_library(
    name = "segmentation_model",
    srcs = ["segmentation_model.cc"],
    hdrs = ["segmentation_model.h"],
    deps = [
        ":image_processor",
        "//litert/c:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_tensor_buffer",
        "//third_party/GL:EGL_headers",
        "//third_party/GL:GLES2_headers",
        "//third_party/GL:GLES3_headers",
        "//third_party/GL/native:EGL",
        "//third_party/GL/native:GLESv3",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)
