load("@org_tensorflow//tensorflow:tensorflow.bzl", "if_google")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

_selectors_package = (
    "//tflite/delegates/gpu/common/selectors/" +
    if_google("google", "default")
)

cc_library(
    name = "convolution_selector",
    hdrs = ["convolution_selector.h"],
    deps = [
        "//tflite/delegates/gpu/common:model_hints",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common/task:gpu_operation",
        "//tflite/delegates/gpu/common/task:weights_layout",
        _selectors_package + ":convolution_selector",
    ],
)

cc_library(
    name = "convolution_transposed_selector",
    hdrs = ["convolution_transposed_selector.h"],
    deps = [
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common/task:gpu_operation",
        "//tflite/delegates/gpu/common/task:weights_layout",
        "@com_google_absl//absl/memory",
        _selectors_package + ":convolution_transposed_selector",
    ],
)

cc_library(
    name = "default_selector",
    hdrs = ["default_selector.h"],
    deps = [
        ":subgraph",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_hints",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common/task:gpu_operation",
        "//tflite/delegates/gpu/common/task:tensor_desc",
        _selectors_package + ":default_selector",
    ],
)

cc_library(
    name = "dw_convolution_selector",
    hdrs = ["dw_convolution_selector.h"],
    deps = [
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common/task:gpu_operation",
        "@com_google_absl//absl/memory",
        _selectors_package + ":dw_convolution_selector",
    ],
)

cc_library(
    name = "fully_connected_selector",
    hdrs = ["fully_connected_selector.h"],
    deps = [
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common/task:gpu_operation",
        "@com_google_absl//absl/memory",
        _selectors_package + ":fully_connected_selector",
    ],
)

cc_library(
    name = "operation_selector",
    srcs = ["operation_selector.cc"],
    hdrs = ["operation_selector.h"],
    deps = [
        ":convolution_selector",
        ":convolution_transposed_selector",
        ":default_selector",
        ":dw_convolution_selector",
        ":fully_connected_selector",
        ":simple_selectors",
        ":subgraph",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:flops_util",
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_hints",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:tensor",
        "//tflite/delegates/gpu/common:types",
        "//tflite/delegates/gpu/common:util",
        "//tflite/delegates/gpu/common:winograd_util",
        "//tflite/delegates/gpu/common/task:gpu_operation",
        "//tflite/delegates/gpu/common/task:tensor_desc",
        "//tflite/delegates/gpu/common/task:weights_conversion",
        "//tflite/delegates/gpu/common/task:weights_layout",
        "//tflite/delegates/gpu/common/tasks:elementwise",
        "//tflite/delegates/gpu/common/tasks:mean_stddev_normalization",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "simple_selectors",
    srcs = ["simple_selectors.cc"],
    hdrs = ["simple_selectors.h"],
    deps = [
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:tensor",
        "//tflite/delegates/gpu/common/task:gpu_operation",
        "//tflite/delegates/gpu/common/tasks:add",
        "//tflite/delegates/gpu/common/tasks:cast",
        "//tflite/delegates/gpu/common/tasks:concat_xy",
        "//tflite/delegates/gpu/common/tasks:concat_z",
        "//tflite/delegates/gpu/common/tasks:cumsum",
        "//tflite/delegates/gpu/common/tasks:depthwise_conv",
        "//tflite/delegates/gpu/common/tasks:gather",
        "//tflite/delegates/gpu/common/tasks:lstm",
        "//tflite/delegates/gpu/common/tasks:max_unpooling",
        "//tflite/delegates/gpu/common/tasks:one_hot",
        "//tflite/delegates/gpu/common/tasks:padding",
        "//tflite/delegates/gpu/common/tasks:pooling",
        "//tflite/delegates/gpu/common/tasks:prelu",
        "//tflite/delegates/gpu/common/tasks:quantize_and_dequantize",
        "//tflite/delegates/gpu/common/tasks:reduce",
        "//tflite/delegates/gpu/common/tasks:relu",
        "//tflite/delegates/gpu/common/tasks:resampler",
        "//tflite/delegates/gpu/common/tasks:reshape",
        "//tflite/delegates/gpu/common/tasks:reshapex4",
        "//tflite/delegates/gpu/common/tasks:resize",
        "//tflite/delegates/gpu/common/tasks:select_v2",
        "//tflite/delegates/gpu/common/tasks:softmax",
        "//tflite/delegates/gpu/common/tasks:softmax1x1",
        "//tflite/delegates/gpu/common/tasks:space_to_depth",
        "//tflite/delegates/gpu/common/tasks:split",
        "//tflite/delegates/gpu/common/tasks:strided_slice",
        "//tflite/delegates/gpu/common/tasks:tile",
        "//tflite/delegates/gpu/common/tasks:transpose",
        "//tflite/delegates/gpu/common/tasks:winograd",
    ],
)

cc_library(
    name = "special_selector",
    srcs = ["special_selector.cc"],
    hdrs = ["special_selector.h"],
    deps = [
        ":subgraph",
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_hints",
        "//tflite/delegates/gpu/common:precision",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common/task:gpu_operation",
        "//tflite/delegates/gpu/common/task:tensor_desc",
        "//tflite/delegates/gpu/common/tasks:mean_stddev_normalization",
        "//tflite/delegates/gpu/common/tasks/special:conv_pointwise",
        "//tflite/delegates/gpu/common/tasks/special:dw7x7_conv2to6_concat_conv8to8",
        "//tflite/delegates/gpu/common/tasks/special:fc_fc_add",
        "//tflite/delegates/gpu/common/tasks/special:thin_pointwise_fuser",
    ],
)

cc_library(
    name = "subgraph",
    srcs = ["subgraph.cc"],
    hdrs = ["subgraph.h"],
    deps = [
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common/task:gpu_operation",
        "//tflite/delegates/gpu/common/task:tensor_desc",
    ],
)
