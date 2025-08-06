load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite/delegates/gpu:build_defs.bzl", "gtest_main_no_heapcheck_deps")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_binary(
    name = "delegate_testing",
    srcs = ["delegate_testing.cc"],
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [
        "//tflite/delegates/gpu:delegate",
        "//tflite/delegates/gpu/cl:gpu_api_delegate",
        "//tflite/delegates/gpu/common:model_builder",
        "//tflite/delegates/gpu/common:status",
        "//tflite/kernels:builtin_ops",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "gpu_model_test",
    srcs = ["gpu_model_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        "//tflite/delegates/gpu/cl/kernels:cl_test",
        "//tflite/delegates/gpu/common:gpu_model_test_util",
        "//tflite/delegates/gpu/common:status",
    ] + gtest_main_no_heapcheck_deps(),  # constant buffers leak on nvidia
)

cc_binary(
    name = "internal_api_samples",
    srcs = ["internal_api_samples.cc"],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": [
            "-lEGL",
            "-lGLESv3",
        ],
        "//conditions:default": [],
    }),
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [
        "//tflite/delegates/gpu:api",
        "//tflite/delegates/gpu/cl:api",
        "//tflite/delegates/gpu/cl:environment",
        "//tflite/delegates/gpu/cl:inference_context",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_builder",
        "//tflite/delegates/gpu/common:status",
        "//tflite/kernels:builtin_ops",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_binary(
    name = "memory_sharing_sample",
    srcs = ["memory_sharing_sample.cc"],
    deps = [
        "//tflite/core/kernels:builtin_ops",
        "//tflite/delegates/gpu/cl:environment",
        "//tflite/delegates/gpu/cl:inference_context",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_builder",
        "//tflite/delegates/gpu/common:status",
        "@com_google_absl//absl/time",
    ],
)

cc_binary(
    name = "performance_profiling",
    srcs = ["performance_profiling.cc"],
    copts = [
        "-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1",
    ],
    deps = [
        "//tflite/core/kernels:builtin_ops",
        "//tflite/delegates/gpu/cl:cl_command_buffer",
        "//tflite/delegates/gpu/cl:environment",
        "//tflite/delegates/gpu/cl:inference_context",
        "//tflite/delegates/gpu/cl:opencl_wrapper",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_builder",
        "//tflite/delegates/gpu/common:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/time",
        "@opencl_headers",
    ],
)
