load("@bazel_skylib//lib:selects.bzl", "selects")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_static_framework")
load("@build_bazel_rules_apple//apple:macos.bzl", "macos_dylib")
load(
    "@org_tensorflow//tensorflow/core/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load("//tflite:build_def.bzl", "CXX17_BAZEL_ONLY_COPTS", "tflite_pagesize_linkopts")
load("//tflite:special_rules.bzl", "tflite_extra_gles_deps", "tflite_portable_test_suite")
load("//tflite/delegates/gpu:build_defs.bzl", "gpu_delegate_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files([
    "delegate.h",
    "delegate_options.h",
    "metal_delegate.h",
])

_DELEGATE_NO_GL_DEPS = select({
    "@org_tensorflow//tensorflow:android": [
        ":async_buffers",
        "//tflite/core/async/c:task",
        "//tflite/core/async/interop/c:attribute_map",
        "//tflite/core/async/interop/c:constants",
        "//tflite/delegates/gpu/gl:android_sync",
        "//tflite/delegates/gpu/gl:egl_environment",
        "//tflite/delegates/utils",
        "//tflite/delegates/utils:async_type_helpers",
        "//tflite/delegates/utils:ret_macros",
        "//tflite/delegates/utils:sync_fence",
    ],
    "//conditions:default": [],
}) + [
    ":android_hardware_buffer",
    ":api",
    ":delegate_options",
    ":tflite_profile",
    #"@org_tensorflow//third_party/GL:EGL_headers",
    #"@org_tensorflow//third_party/GL:GLES3_headers",
    # go/keep-sorted start
    "@com_google_absl//absl/container:flat_hash_map",
    "@com_google_absl//absl/container:flat_hash_set",
    "@com_google_absl//absl/memory",
    "@com_google_absl//absl/strings",
    "@com_google_absl//absl/types:span",
    "//tflite/async:backend_async_kernel_interface",
    "//tflite/core/async/interop/c:types",
    "//tflite/core/c:common",
    "//tflite/delegates/gpu/cl:util",
    "//tflite/delegates/gpu/common:data_type",
    "//tflite/delegates/gpu/common:model",
    "//tflite/delegates/gpu/common:model_builder",
    "//tflite/delegates/gpu/common:model_builder_helper",
    "//tflite/delegates/gpu/common:quantization_util",
    "//tflite/delegates/gpu/common:status",
    "//tflite/delegates:serialization",
    "//tflite/kernels:kernel_util",
    "//tflite/profiling/telemetry",
    "//tflite/profiling/telemetry/c:telemetry_setting",
    "//tflite/profiling/telemetry/c:telemetry_setting_internal",
    "//tflite/profiling/telemetry:telemetry_status",
    "//tflite:kernel_api",
    "//tflite:minimal_logging",
    # go/keep-sorted end
]

config_setting(
    name = "tflite_gpu_binary_release",
    values = {"copt": "-DTFLITE_GPU_BINARY_RELEASE"},
)

config_setting(
    name = "tflite_gpu_extra_gles_deps",
    constraint_values = [
        "@org_tensorflow//third_party/bazel_platforms/cpu:x86_64",
        "@org_tensorflow//third_party/bazel_platforms/os:linux",
    ],
    values = {
        "copt": "-DTFLITE_GPU_EXTRA_GLES_DEPS",
    },
)

cc_library(
    name = "gl_delegate",
    srcs = ["gl_delegate.cc"],
    hdrs = ["gl_delegate.h"],
    linkopts = gpu_delegate_linkopts(),
    deps = [
        "//tflite:kernel_api",
        "//tflite:minimal_logging",
        "//tflite/core/c:common",
        "//tflite/delegates/gpu/common:convert",
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_builder",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:tensor",
        "//tflite/delegates/gpu/common/transformations:model_transformations",
        "//tflite/delegates/gpu/gl:api",
        "//tflite/delegates/gpu/gl:command_queue",
        "//tflite/delegates/gpu/gl:compiler",
        "//tflite/delegates/gpu/gl:compiler_options",
        "//tflite/delegates/gpu/gl:egl_environment",
        "//tflite/delegates/gpu/gl:gl_buffer",
        "//tflite/delegates/gpu/gl:gl_call",
        "//tflite/delegates/gpu/gl:object",
        "//tflite/delegates/gpu/gl:object_manager",
        "//tflite/delegates/gpu/gl:request_gpu_info",
        "//tflite/delegates/gpu/gl:runtime_options",
        "//tflite/delegates/gpu/gl/converters:bhwc_to_phwc4",
        "//tflite/delegates/gpu/gl/converters:phwc4_to_bhwc",
        "//tflite/delegates/gpu/gl/kernels:registry",
        "//tflite/delegates/gpu/gl/workgroups:best_effort_calculator",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/types:span",
    ] + select({
        "//conditions:default": [
            "//tflite/delegates/gpu/gl:common_cc_fbs",
            "//tflite/delegates/gpu/gl:metadata_cc_fbs",
            "//tflite/delegates/gpu/gl:workgroups_cc_fbs",
            "//tflite/schema:schema_fbs",
            "@flatbuffers",
        ],
        ":tflite_gpu_binary_release": [],
    }) + tflite_extra_gles_deps(),
)

objc_library(
    name = "metal_delegate",
    srcs = ["metal_delegate.mm"],
    hdrs = ["metal_delegate.h"],
    copts = CXX17_BAZEL_ONLY_COPTS,
    module_name = "TensorFlowLiteCMetal",
    sdk_frameworks = ["Metal"],
    deps = [
        "//tflite:kernel_api",
        "//tflite:minimal_logging",
        "//tflite/core/c:common",
        "//tflite/delegates/gpu/common:convert",
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_builder",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:precision",
        "//tflite/delegates/gpu/common:quantization_util",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:types",
        "//tflite/delegates/gpu/metal:buffer_convert",
        "//tflite/delegates/gpu/metal:common",
        "//tflite/delegates/gpu/metal:inference_context",
        "//tflite/delegates/gpu/metal:metal_spatial_tensor",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
    ],
)

objc_library(
    name = "metal_delegate_internal",
    hdrs = ["metal_delegate_internal.h"],
    copts = CXX17_BAZEL_ONLY_COPTS,
    sdk_frameworks = ["Metal"],
    deps = ["//tflite/delegates/gpu:metal_delegate"],
)

# build -c opt --config android_arm64 --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --linkopt -s --strip always :libtensorflowlite_gpu_gl.so
cc_binary(
    name = "libtensorflowlite_gpu_gl.so",
    linkopts = [
        "-Wl,-soname=libtensorflowlite_gpu_gl.so",
    ] + gpu_delegate_linkopts() + select({
        "@org_tensorflow//tensorflow:windows": [],
        "//conditions:default": [
            "-fvisibility=hidden",
        ],
    }) + tflite_pagesize_linkopts(),
    linkshared = 1,
    linkstatic = 1,
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [":gl_delegate"],
)

# build -c opt --config android_arm64 --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --linkopt -s --strip always :libtensorflowlite_gpu_delegate.so
cc_binary(
    name = "libtensorflowlite_gpu_delegate.so",
    linkopts = [
        "-Wl,-soname=libtensorflowlite_gpu_delegate.so",
    ] + gpu_delegate_linkopts() + select({
        "@org_tensorflow//tensorflow:windows": [],
        "//conditions:default": [
            "-fvisibility=hidden",
        ],
    }),
    linkshared = 1,
    linkstatic = 1,
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [":delegate"],
)

# bazel build -c opt --cpu ios_arm64 --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --linkopt -s --strip always --cxxopt=-std=c++14 :libtensorflowlite_gpu_metal --apple_platform_type=ios
ios_static_framework(
    name = "tensorflow_lite_gpu_framework",
    hdrs = [
        "metal_delegate.h",
        "metal_delegate_internal.h",
    ],
    minimum_os_version = "12.0",
    deps = [":metal_delegate"],
)

# Note: Support for MacOS is best-effort at the moment.
# bazel build -c opt --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --linkopt -s --strip always --cxxopt=-std=c++14 :tensorflow_lite_gpu_dylib --apple_platform_type=macos
macos_dylib(
    name = "tensorflow_lite_gpu_dylib",
    linkopts = [
        "-all_load",
        "-dead_strip",
    ],
    minimum_os_version = "12.0",
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        ":metal_delegate",
        ":metal_delegate_internal",
    ],
)

cc_library(
    name = "api",
    srcs = ["api.cc"],
    hdrs = ["api.h"],
    deps = [
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:util",
        "//tflite/delegates/gpu/gl:portable",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@opencl_headers",
        "@vulkan_headers//:vulkan_headers_no_prototypes",
    ],
)

cc_library(
    name = "spi",
    hdrs = ["spi.h"],
    deps = [
        ":api",
        "//tflite/delegates/gpu/common:access_type",
        "//tflite/delegates/gpu/common:status",
    ],
)

# Currently the GPU delegate needs to be built on Android (due to EGL dependency),
# or built with -DCL_DELEGATE_NO_GL (disabling OpenGL backend fallback), or both.
selects.config_setting_group(
    name = "supports_gpu_delegate",
    match_any = [
        "@org_tensorflow//tensorflow:android",
        "//tflite/delegates/gpu/cl:opencl_delegate_no_gl",
    ],
)

cc_library(
    name = "delegate_options",
    srcs = ["delegate_options.cc"],
    hdrs = ["delegate_options.h"],
    deps = ["//tflite/core/c:common"],
)

# copybara:uncomment_begin(google-only)
# cc_library(
#     name = "delegate_no_gl",
#     srcs = [
#         # copybara:comment_begin(oss-only)
# "android_version.cc",
#         # copybara:comment_end
#         "delegate.cc",
#     ],
#     hdrs = ["delegate.h"],
#     defines = ["CL_DELEGATE_NO_GL"],
#     linkopts = gpu_delegate_linkopts(),
#     deps = _DELEGATE_NO_GL_DEPS + [
#         "//tflite/delegates/gpu/cl:api_no_gl",
#         "//tflite/delegates/gpu/gl:api2",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "delegate",
    srcs = [
        # copybara:comment_begin(oss-only)
        "android_version.cc",
        # copybara:comment_end
        "delegate.cc",
    ],
    hdrs = ["delegate.h"],
    linkopts = gpu_delegate_linkopts(),
    deps = select({
        "//tflite/delegates/gpu/cl:opencl_delegate_no_gl": [],
        "//conditions:default": [
            "//tflite/delegates/gpu/gl:api2",
        ],
    }) + _DELEGATE_NO_GL_DEPS + ["//tflite/delegates/gpu/cl:api"],
)

cc_library(
    name = "tflite_profile",
    srcs = ["tflite_profile.cc"],
    hdrs = ["tflite_profile.h"],
    deps = [
        "//tflite/core/api",
        "//tflite/delegates/gpu/common/task:profiling_info",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "android_hardware_buffer",
    srcs = ["android_hardware_buffer.cc"],
    hdrs = ["android_hardware_buffer.h"],
)

cc_test(
    name = "android_hardware_buffer_test",
    srcs = ["android_hardware_buffer_test.cc"],
    deps = [
        ":android_hardware_buffer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "async_buffers",
    srcs = ["async_buffers.cc"],
    hdrs = ["async_buffers.h"],
    deps = [
        ":android_hardware_buffer",
        ":api",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/gl:gl_errors",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "async_buffers_test",
    srcs = ["async_buffers_test.cc"],
    tags = tf_gpu_tests_tags() + [
        "local",
        "nobuilder",
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":android_hardware_buffer",
        ":api",
        ":async_buffers",
        ":delegate",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/gl:egl_environment",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_portable_test_suite()
