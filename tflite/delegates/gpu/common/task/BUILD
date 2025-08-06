load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "arguments",
    srcs = ["arguments.cc"],
    hdrs = ["arguments.h"],
    deps = [
        ":buffer_desc",
        ":gpu_object_desc",
        ":serialization_base_cc_fbs",
        ":util",
        "//tflite/delegates/gpu/common:access_type",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:types",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "buffer_desc",
    srcs = ["buffer_desc.cc"],
    hdrs = ["buffer_desc.h"],
    deps = [
        ":gpu_object_desc",
        ":util",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "compiler_options",
    hdrs = ["compiler_options.h"],
)

cc_library(
    name = "gpu_object_desc",
    hdrs = ["gpu_object_desc.h"],
    deps = [
        ":serialization_base_cc_fbs",
        "//tflite/delegates/gpu/common:access_type",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_operation",
    srcs = ["gpu_operation.cc"],
    hdrs = ["gpu_operation.h"],
    deps = [
        ":arguments",
        ":buffer_desc",
        ":compiler_options",
        ":gpu_tensor",
        ":serialization_base_cc_fbs",
        ":tensor_desc",
        ":tuning_type",
        ":util",
        ":work_group_picking",
        "//tflite/delegates/gpu/common:access_type",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:kernel_info",
        "//tflite/delegates/gpu/common:precision",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:types",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_tensor",
    hdrs = ["gpu_tensor.h"],
    deps = [
        "//tflite/delegates/gpu/common/task:tensor_desc",
    ],
)

cc_library(
    name = "profiling_info",
    srcs = ["profiling_info.cc"],
    hdrs = ["profiling_info.h"],
    deps = [
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "qcom_thin_filter_desc",
    srcs = ["qcom_thin_filter_desc.cc"],
    hdrs = ["qcom_thin_filter_desc.h"],
    deps = [
        ":gpu_object_desc",
        "//tflite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "serialization_base",
    srcs = ["serialization_base.cc"],
    hdrs = ["serialization_base.h"],
    deps = [
        ":arguments",
        ":buffer_desc",
        ":gpu_object_desc",
        ":gpu_operation",
        ":serialization_base_cc_fbs",
        ":tensor_desc",
        "//tflite/delegates/gpu/common:precision",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:types",
    ],
)

flatbuffer_cc_library(
    name = "serialization_base_cc_fbs",
    srcs = ["tflite_serialization_base.fbs"],
    flatc_args = [
        "--scoped-enums",
    ],
)

cc_library(
    name = "tensor_desc",
    srcs = ["tensor_desc.cc"],
    hdrs = ["tensor_desc.h"],
    deps = [
        ":gpu_object_desc",
        ":util",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:tensor",
        "//tflite/delegates/gpu/common:types",
        "//tflite/delegates/gpu/common:util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "testing_util",
    testonly = 1,
    srcs = ["testing_util.cc"],
    hdrs = ["testing_util.h"],
    deps = [
        ":gpu_operation",
        ":tensor_desc",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:gpu_model",
        "//tflite/delegates/gpu/common:precision",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tuning_type",
    hdrs = ["tuning_type.h"],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        ":gpu_object_desc",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:precision",
        "//tflite/delegates/gpu/common:types",
        "//tflite/delegates/gpu/common:util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "weights_conversion",
    srcs = ["weights_conversion.cc"],
    hdrs = ["weights_conversion.h"],
    deps = [
        ":weights_layout",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:tensor",
        "//tflite/delegates/gpu/common:types",
        "//tflite/delegates/gpu/common:util",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "weights_layout",
    srcs = ["weights_layout.cc"],
    hdrs = ["weights_layout.h"],
    deps = [
        "//tflite/delegates/gpu/common:data_type",
    ],
)

cc_library(
    name = "work_group_picking",
    srcs = ["work_group_picking.cc"],
    hdrs = ["work_group_picking.h"],
    deps = [
        ":tuning_type",
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:kernel_info",
        "//tflite/delegates/gpu/common:types",
        "//tflite/delegates/gpu/common:util",
        "//tflite/delegates/gpu/common:workgroup_selection",
    ],
)
