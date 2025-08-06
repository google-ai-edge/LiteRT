load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load(
    "@org_tensorflow//tensorflow/core/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite:special_rules.bzl", "tflite_portable_test_suite")
load("//tflite/delegates/gpu:build_defs.bzl", "gpu_delegate_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "android_sync",
    srcs = ["android_sync.cc"],
    hdrs = ["android_sync.h"],
    linkopts = gpu_delegate_linkopts(),
    deps = [
        ":portable",
        "//tflite/delegates/gpu/common:status",
    ],
)

cc_test(
    name = "android_sync_test",
    srcs = ["android_sync_test.cc"],
    linkopts = gpu_delegate_linkopts(),
    tags = tf_gpu_tests_tags() + [
        "local",
        "nobuilder",
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":android_sync",
        ":egl_environment",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "api",
    srcs = ["api.cc"],
    hdrs = ["api.h"],
    deps = [
        ":command_queue",
        ":common_cc_fbs",
        ":compiler",
        ":compiler_options",
        ":gl_call",
        ":node_shader",
        ":object",
        ":object_manager",
        ":portable",
        ":request_gpu_info",
        ":runtime",
        ":runtime_options",
        ":stats",
        ":variable",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:types",
        "//tflite/delegates/gpu/common:util",
        "//tflite/delegates/gpu/gl/workgroups:calculator",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ] + select({
        "//tflite/delegates/gpu:tflite_gpu_binary_release": [],
        "//conditions:default": [
            ":serialization",
        ],
    }),
)

cc_library(
    name = "api2",
    srcs = ["api2.cc"],
    hdrs = ["api2.h"],
    deps = [
        ":command_queue",
        ":compiler",
        ":egl_environment",
        ":gl_call",
        ":object",
        ":portable",
        ":request_gpu_info",
        ":runtime",
        ":variable",
        "//tflite/delegates/gpu:api",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:tensor",
        "//tflite/delegates/gpu/gl/kernels:converter",
        "//tflite/delegates/gpu/gl/kernels:registry",
        "//tflite/delegates/gpu/gl/workgroups:default_calculator",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "command_queue",
    srcs = ["command_queue.cc"],
    hdrs = ["command_queue.h"],
    deps = [
        ":gl_call",
        ":gl_program",
        ":gl_sync",
        ":portable",
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:types",
        "@com_google_absl//absl/memory",
    ],
)

flatbuffer_cc_library(
    name = "common_cc_fbs",
    srcs = ["common.fbs"],
)

# Generic schema for inference on GPU device.
flatbuffer_cc_library(
    name = "compiled_model_cc_fbs",
    srcs = ["compiled_model.fbs"],
    flatc_args = [
        "--scoped-enums",
    ],
    includes = [
        ":common_cc_fbs_includes",
    ],
)

cc_library(
    name = "compiler",
    srcs = ["compiler.cc"],
    hdrs = ["compiler.h"],
    deps = [
        ":compiler_options",
        ":float16_conversions",
        ":node_shader",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:types",
        "//tflite/delegates/gpu/gl/compiler:compiled_node",
        "//tflite/delegates/gpu/gl/compiler:fuse_auto_input",
        "//tflite/delegates/gpu/gl/compiler:fuse_inline",
        "//tflite/delegates/gpu/gl/compiler:fuse_inplace",
        "//tflite/delegates/gpu/gl/compiler:shader_code",
        "//tflite/delegates/gpu/gl/compiler:shader_codegen",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:any",
    ],
)

cc_library(
    name = "compiler_options",
    hdrs = ["compiler_options.h"],
    deps = [
        ":object",
    ],
)

cc_library(
    name = "egl_context",
    srcs = ["egl_context.cc"],
    hdrs = ["egl_context.h"],
    deps = [
        ":gl_call",
        ":gl_errors",
        ":portable",
        "//tflite/delegates/gpu/common:status",
    ],
)

cc_library(
    name = "egl_environment",
    srcs = ["egl_environment.cc"],
    hdrs = ["egl_environment.h"],
    deps = [
        ":egl_context",
        ":egl_surface",
        ":gl_call",
        ":portable",
        ":request_gpu_info",
        "//tflite/delegates/gpu/common:status",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "egl_surface",
    srcs = ["egl_surface.cc"],
    hdrs = ["egl_surface.h"],
    deps = [
        ":gl_call",
        ":gl_errors",
        ":portable",
        "//tflite/delegates/gpu/common:status",
    ],
)

cc_library(
    name = "float16_conversions",
    srcs = ["float16_conversions.cc"],
    hdrs = ["float16_conversions.h"],
    deps = [
        ":object",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:tensor",
        "@FP16",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "gl_buffer",
    srcs = ["gl_buffer.cc"],
    hdrs = ["gl_buffer.h"],
    deps = [
        ":gl_call",
        ":gl_errors",
        ":portable",
        "//tflite/delegates/gpu/common:status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "gl_buffer_test",
    srcs = ["gl_buffer_test.cc"],
    linkopts = [
        "-lEGL",
        "-lGLESv2",
        "-lm",
    ],
    tags = tf_gpu_tests_tags() + [
        "local",
        "nobuilder",
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":egl_environment",
        ":gl_buffer",
        "//tflite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gl_call",
    hdrs = ["gl_call.h"],
    deps = [
        ":gl_errors",
        "//tflite/delegates/gpu/common:status",
    ],
)

cc_library(
    name = "gl_errors",
    srcs = ["gl_errors.cc"],
    hdrs = ["gl_errors.h"],
    deps = [
        ":portable",
        "//tflite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gl_program",
    srcs = ["gl_program.cc"],
    hdrs = ["gl_program.h"],
    deps = [
        ":gl_call",
        ":gl_errors",
        ":gl_shader",
        ":portable",
        ":variable",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:types",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "gl_shader",
    srcs = ["gl_shader.cc"],
    hdrs = ["gl_shader.h"],
    deps = [
        ":gl_call",
        ":gl_errors",
        ":portable",
        "//tflite/delegates/gpu/common:status",
    ],
)

cc_library(
    name = "gl_texture",
    srcs = ["gl_texture.cc"],
    hdrs = ["gl_texture.h"],
    deps = [
        ":gl_call",
        ":gl_errors",
        ":gl_texture_helper",
        ":portable",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:tensor",
        "//tflite/delegates/gpu/common:types",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "gl_texture_helper",
    srcs = ["gl_texture_helper.cc"],
    hdrs = ["gl_texture_helper.h"],
    deps = [
        ":portable",
        "//tflite/delegates/gpu/common:data_type",
    ],
)

cc_library(
    name = "gl_sync",
    srcs = ["gl_sync.cc"],
    hdrs = ["gl_sync.h"],
    deps = [
        ":gl_buffer",
        ":gl_call",
        ":gl_errors",
        ":gl_program",
        ":portable",
        "//tflite/delegates/gpu/common:status",
    ],
)

flatbuffer_cc_library(
    name = "metadata_cc_fbs",
    srcs = ["metadata.fbs"],
    includes = [
        ":common_cc_fbs_includes",
        ":workgroups_cc_fbs_includes",
    ],
)

cc_library(
    name = "node_shader",
    hdrs = ["node_shader.h"],
    deps = [
        ":compiler_options",
        ":object",
        ":variable",
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:types",
        "@com_google_absl//absl/types:any",
    ],
)

cc_library(
    name = "object",
    hdrs = ["object.h"],
    deps = [
        "//tflite/delegates/gpu/common:access_type",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:types",
        "//tflite/delegates/gpu/common:util",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "object_manager",
    srcs = ["object_manager.cc"],
    hdrs = ["object_manager.h"],
    deps = [
        ":gl_buffer",
        ":gl_texture",
        ":stats",
        "//tflite/delegates/gpu/common:convert",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "portable",
    hdrs = [
        "portable_egl.h",
        "portable_gl31.h",
    ],
)

cc_library(
    name = "request_gpu_info",
    srcs = ["request_gpu_info.cc"],
    hdrs = ["request_gpu_info.h"],
    deps = [
        ":gl_errors",
        ":portable",
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "runtime",
    srcs = ["runtime.cc"],
    hdrs = ["runtime.h"],
    deps = [
        ":command_queue",
        ":gl_buffer",
        ":gl_call",
        ":gl_errors",
        ":gl_program",
        ":gl_shader",
        ":gl_texture",
        ":object",
        ":object_manager",
        ":portable",
        ":runtime_options",
        ":stats",
        ":variable",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:memory_management",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:types",
        "//tflite/delegates/gpu/gl/runtime:shared_buffer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "runtime_options",
    hdrs = ["runtime_options.h"],
)

cc_library(
    name = "serialization",
    srcs = ["serialization.cc"],
    hdrs = ["serialization.h"],
    deps = [
        ":common_cc_fbs",
        ":compiled_model_cc_fbs",
        ":object",
        ":variable",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:types",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@flatbuffers",
    ],
)

cc_test(
    name = "serialization_test",
    srcs = ["serialization_test.cc"],
    linkopts = [
        "-lm",
    ],
    tags = [
        "local",
        "nobuilder",
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":object",
        ":serialization",
        ":variable",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:types",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "stats",
    hdrs = ["stats.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "variable",
    hdrs = ["variable.h"],
    deps = [
        "//tflite/delegates/gpu/common:types",
        "@com_google_absl//absl/types:variant",
    ],
)

flatbuffer_cc_library(
    name = "workgroups_cc_fbs",
    srcs = ["workgroups.fbs"],
    includes = [
        ":common_cc_fbs_includes",
    ],
)

tflite_portable_test_suite()
