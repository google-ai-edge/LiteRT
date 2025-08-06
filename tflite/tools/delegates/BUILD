load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")
load("//tflite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite", "cc_test_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

common_copts = tflite_copts() + tflite_copts_warnings()

cc_library_with_tflite(
    name = "delegate_provider_hdr",
    hdrs = [
        "delegate_provider.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = common_copts,
    tflite_deps = [
        "//tflite/c:common",
    ],
    deps = [
        "//tflite/tools:command_line_flags",
        "//tflite/tools:logging",
        "//tflite/tools:tool_params",
    ],
)

cc_library_with_tflite(
    name = "delegate_provider_lib",
    srcs = [
        "delegate_provider.cc",
    ],
    copts = common_copts,
    tflite_deps = [
        ":delegate_provider_hdr",
    ],
)

# A convenient library for all inference execution providers.
cc_library_with_tflite(
    name = "tflite_execution_providers",
    copts = tflite_copts(),
    tflite_deps = [
        ":xnnpack_delegate_provider",
        ":delegate_provider_lib",
    ],
    deps = [
        ":coreml_delegate_provider",
        ":default_execution_provider",
        ":external_delegate_provider",
        ":gpu_delegate_provider",
        ":hexagon_delegate_provider",
        ":nnapi_delegate_provider",
        "//tflite/tools/delegates/experimental/stable_delegate:delegate_provider",
    ],
    alwayslink = 1,
)

cc_library_with_tflite(
    name = "default_execution_provider",
    srcs = ["default_execution_provider.cc"],
    copts = tflite_copts(),
    linkstatic = True,
    tflite_deps = [
        ":delegate_provider_hdr",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_library(
    name = "gpu_delegate_provider",
    srcs = ["gpu_delegate_provider.cc"],
    copts = common_copts + select({
        "@org_tensorflow//tensorflow:ios": [
            "-xobjective-c++",
        ],
        "@org_tensorflow//tensorflow:macos_arm64": [
            "-xobjective-c++",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":delegate_provider_hdr",
        "//tflite/tools/evaluation:utils",
    ] + select({
        "//tflite/delegates/gpu:supports_gpu_delegate": [
            "//tflite/delegates/gpu:delegate",
        ],
        "//conditions:default": [],
    }) + select({
        "@org_tensorflow//tensorflow:ios": [
            "//tflite/delegates/gpu:metal_delegate",
        ],
        "@org_tensorflow//tensorflow:macos_arm64": [
            "//tflite/delegates/gpu:metal_delegate",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library(
    name = "nnapi_delegate_provider",
    srcs = ["nnapi_delegate_provider.cc"],
    copts = common_copts,
    deps = [
        ":delegate_provider_hdr",
        "//tflite/delegates/nnapi:nnapi_delegate",
        "//tflite/nnapi:nnapi_implementation",
        "//tflite/nnapi:nnapi_util",
        "//tflite/nnapi/sl:nnapi_support_library",
        "//tflite/nnapi/sl:nnapi_support_library_headers",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hexagon_delegate_provider",
    srcs = ["hexagon_delegate_provider.cc"],
    copts = common_copts,
    deps = [
        ":delegate_provider_hdr",
        "//tflite/tools/evaluation:utils",
    ] + select({
        "@org_tensorflow//tensorflow:arm_any": [
            "//tflite/delegates/hexagon:hexagon_delegate",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library(
    name = "coreml_delegate_provider",
    srcs = ["coreml_delegate_provider.cc"],
    copts = common_copts + select({
        "@org_tensorflow//tensorflow:ios": [
            "-xobjective-c++",
        ],
        "@org_tensorflow//tensorflow:macos_arm64": [
            "-xobjective-c++",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":delegate_provider_hdr",
        "//tflite/tools/evaluation:utils",
    ] + select({
        "@org_tensorflow//tensorflow:ios": [
            "//tflite/delegates/coreml:coreml_delegate",
        ],
        "@org_tensorflow//tensorflow:macos_arm64": [
            "//tflite/delegates/coreml:coreml_delegate",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library_with_tflite(
    name = "xnnpack_delegate_provider",
    srcs = ["xnnpack_delegate_provider.cc"],
    copts = tflite_copts(),
    linkstatic = True,
    tflite_deps = [
        ":delegate_provider_hdr",
        "//tflite/tools/evaluation:utils",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tflite/delegates/xnnpack:xnnpack_delegate",
        "//tflite/tools:tool_params",
    ],
    alwayslink = 1,
)

cc_test(
    name = "xnnpack_delegate_provider_test",
    srcs = ["xnnpack_delegate_provider_test.cc"],
    copts = tflite_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":delegate_provider_hdr",
        ":xnnpack_delegate_provider",
        "//tflite/delegates/xnnpack:xnnpack_delegate",
        "//tflite/tools:tool_params",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "external_delegate_provider",
    srcs = ["external_delegate_provider.cc"],
    copts = tflite_copts(),
    linkopts = select({
        "@org_tensorflow//tensorflow:windows": [],
        "//conditions:default": ["-ldl"],
    }),
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        ":delegate_provider_hdr",
        "//tflite/delegates/external:external_delegate",
    ],
    alwayslink = 1,
)

cc_test_with_tflite(
    name = "delegate_provider_lib_test",
    size = "small",
    srcs = ["delegate_provider_test.cc"],
    copts = tflite_copts(),
    # See details in https://github.com/bazelbuild/bazel/issues/11552 to avoid
    # lazy symbol binding failure on macOS.
    linkstatic = select({
        "@org_tensorflow//tensorflow:macos": True,
        "//conditions:default": False,
    }),
    tflite_deps = [
        ":default_execution_provider",
        ":delegate_provider_hdr",
        ":delegate_provider_lib",
        ":xnnpack_delegate_provider",
        "//tflite/c:test_util",
    ],
    deps = [
        ":nnapi_delegate_provider",
        "//tflite/delegates/utils/dummy_delegate:dummy_delegate_provider",
        "//tflite/tools:tool_params",
        "@com_google_googletest//:gtest_main",
    ],
)
