load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "calculator",
    srcs = ["calculator.cc"],
    hdrs = ["calculator.h"],
    deps = [
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:types",
        "//tflite/delegates/gpu/gl/compiler:shader_code",
    ],
)

cc_library(
    name = "default_calculator",
    srcs = ["default_calculator.cc"],
    hdrs = ["default_calculator.h"],
    deps = [
        ":calculator",
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:types",
        "//tflite/delegates/gpu/gl/compiler:shader_code",
    ],
)

cc_library(
    name = "calculator_from_metadata",
    srcs = ["calculator_from_metadata.cc"],
    hdrs = ["calculator_from_metadata.h"],
    deps = select({
        "//tflite/delegates/gpu:tflite_gpu_binary_release": [],
        "//conditions:default": [
            ":calculator",
            ":default_calculator",
            "//tflite/delegates/gpu/common:gpu_info",
            "//tflite/delegates/gpu/common:types",
            "//tflite/delegates/gpu/gl:common_cc_fbs",
            "//tflite/delegates/gpu/gl:metadata_cc_fbs",
            "//tflite/delegates/gpu/gl:workgroups_cc_fbs",
            "@com_google_absl//absl/container:flat_hash_map",
            "@com_google_absl//absl/memory",
            "@flatbuffers",
        ],
    }) + [
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/gl/compiler:shader_code",
    ],
)

cc_library(
    name = "best_effort_calculator",
    srcs = ["best_effort_calculator.cc"],
    hdrs = ["best_effort_calculator.h"],
    deps = [
        ":calculator",
        ":default_calculator",
        "//tflite/delegates/gpu/common:gpu_info",
    ] + select({
        "//tflite/delegates/gpu:tflite_gpu_binary_release": [],
        "//conditions:default": [
            ":calculator_from_metadata",
        ],
    }),
)

cc_library(
    name = "ideal_workgroup_picker",
    srcs = ["ideal_workgroup_picker.cc"],
    hdrs = ["ideal_workgroup_picker.h"],
    deps = [
        ":calculator",
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:types",
    ],
)
