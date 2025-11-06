load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "drop_savedmodel_semantics",
    srcs = [
        "transforms/drop_savedmodel_semantics.cc",
    ],
    hdrs = [
        "transforms/drop_savedmodel_semantics.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":stablehlo_util",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
    ],
    alwayslink = 1,
)

cc_library(
    name = "stablehlo_util",
    srcs = [
        "transforms/stablehlo_util.cc",
    ],
    hdrs = [
        "transforms/stablehlo_util.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "@llvm-project//llvm:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hlo_matchers",
    srcs = [
        "transforms/hlo_matchers.cc",
    ],
    hdrs = [
        "transforms/hlo_matchers.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "smuggle_disallowed_ops",
    srcs = [
        "transforms/smuggle_disallowed_ops.cc",
    ],
    hdrs = [
        "transforms/smuggle_disallowed_ops.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":stablehlo_util",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tfl_stablehlo",
    srcs = [
        "transforms/tfl_stablehlo_pass.cc",
    ],
    hdrs = [
        "transforms/tfl_stablehlo_pass.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "//tflite/converter:tensorflow_lite",
        "@flatbuffers",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "transforms",
    srcs = [
        "transforms/transforms.cc",
    ],
    hdrs = [
        "transforms/transforms.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":drop_savedmodel_semantics",
        ":legalize_stablehlo_custom_call_to_composite",
        ":legalize_tf_xla_call_module_to_stablehlo_pass",
        ":optimize",
        ":smuggle_disallowed_ops",
        ":stablehlo_fuse_convolution_pass",
        ":stablehlo_unfuse_batch_norm_pass",
        ":unfold_splat_constant_pass",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/stablehlo:bridge_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/stablehlo:fold_broadcast_pass",
        "@org_tensorflow//tensorflow/compiler/mlir/stablehlo:fuse_convolution_pass",
        "@org_tensorflow//tensorflow/compiler/mlir/stablehlo:rename_entrypoint_to_main",
        "@org_tensorflow//tensorflow/compiler/mlir/stablehlo:tf_stablehlo",
        "@org_tensorflow//tensorflow/compiler/mlir/stablehlo:unfuse_batch_norm_pass",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tf_saved_model_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/tf2xla/transforms:tf_xla_passes",
    ],
)

cc_library(
    name = "check_accepted_ops_pass",
    srcs = [
        "transforms/check_accepted_ops_pass.cc",
    ],
    hdrs = [
        "transforms/check_accepted_ops_pass.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":stablehlo_util",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "op_stat_pass",
    srcs = [
        "transforms/op_stat_pass.cc",
    ],
    hdrs = [
        "transforms/op_stat_pass.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":stablehlo_util",
        "//tflite/converter:tensorflow_lite",
        "//tflite/converter/quantization/common/quantization_lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/stablehlo_passes.h.inc": [
        "-gen-pass-decls",
        "-name=OdmlStablehlo",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/stablehlo_passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "compose_uniform_quantized_type_pass",
    srcs = ["transforms/compose_uniform_quantized_type_pass.cc"],
    hdrs = ["transforms/stablehlo_passes.h"],
    copts = ["-Ithird_party"],
    visibility = ["//visibility:public"],
    deps = [
        ":passes_inc_gen",
        "@com_google_absl//absl/algorithm:container",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/common:attrs_and_constraints",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/common:uniform_quantized_types",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "stablehlo_unfuse_batch_norm_pass",
    srcs = [
        "transforms/stablehlo_unfuse_batch_norm_pass.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":passes_inc_gen",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "stablehlo_fuse_convolution_pass",
    srcs = [
        "transforms/stablehlo_fuse_convolution_pass.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":passes_inc_gen",
        "//tflite/converter:validators",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/common:attrs_and_constraints",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "unfold_splat_constant_pass",
    srcs = [
        "transforms/unfold_splat_constant_pass.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo",
    ],
    alwayslink = 1,
)

cc_library(
    name = "legalize_tf_xla_call_module_to_stablehlo_pass",
    srcs = [
        "transforms/legalize_tf_xla_call_module_to_stablehlo_pass.cc",
    ],
    hdrs = [
        "transforms/legalize_tf_xla_call_module_to_stablehlo_pass.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_serialization",
        "@stablehlo//:vhlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "legalize_stablehlo_composite_to_tfl_custom",
    srcs = [
        "transforms/legalize_stablehlo_composite_to_tfl_custom.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
        "transforms/stablehlo_passes.h.inc",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "//tflite/converter:tensorflow_lite",
        "@flatbuffers",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "legalize_stablehlo_to_vhlo_pass",
    srcs = [
        "transforms/legalize_stablehlo_to_vhlo.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
        "transforms/stablehlo_passes.h.inc",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":uniform_quantized_stablehlo_to_tfl_pass",
        "//tflite/converter:tensorflow_lite",
        "//tflite/converter/core:macros",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:base",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_ops_inc_gen",
        "@stablehlo//:stablehlo_pass_inc_gen",
        "@stablehlo//:stablehlo_passes",
        "@stablehlo//:stablehlo_portable_api",
        "@stablehlo//:stablehlo_type_inference",
        "@stablehlo//:version",
        "@stablehlo//:vhlo_ops",
        "@stablehlo//:vhlo_types",
    ],
    alwayslink = 1,
)

cc_library(
    name = "legalize_stablehlo_custom_call_to_composite",
    srcs = [
        "transforms/legalize_stablehlo_custom_call_to_composite.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
        "transforms/stablehlo_passes.h.inc",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "optimize",
    srcs = [
        "transforms/optimize.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
    alwayslink = 1,
)

cc_library(
    name = "uniform_quantized_stablehlo_to_tfl_pass",
    srcs = ["transforms/uniform_quantized_stablehlo_to_tfl_pass.cc"],
    hdrs = [
        "transforms/stablehlo_passes.h",
        "transforms/stablehlo_passes.h.inc",
    ],
    copts = ["-Ithird_party"],
    deps = [
        ":passes_inc_gen",
        "//tflite/converter:tensorflow_lite",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/common:attrs_and_constraints",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/common:uniform_quantized_types",
        "@stablehlo//:base",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

gentbl_cc_library(
    name = "hlo_legalize_tf_inc_gen",
    compatible_with = get_compatible_with_portable(),
    includes = [
        "/external/org_tensorflow",
        "/third_party/tensorflow",
    ],
    tbl_outs = {"transforms/generated_legalize_hlo.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/legalize_hlo_patterns.td",
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncTdFiles",
        "@local_xla//xla/mlir_hlo:hlo_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "hlo_legalize_tflite_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/generated_tflite_legalize_hlo.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/tflite_legalize_hlo_patterns.td",
    deps = [
        "//tflite/converter:tensorflow_lite_ops_td_files",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncTdFiles",
        "@local_xla//xla/mlir_hlo:hlo_ops_td_files",
    ],
)

cc_library(
    name = "tfl_legalize_hlo",
    srcs = [
        "transforms/tflite_legalize_hlo.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":hlo_legalize_tflite_inc_gen",
        ":passes_inc_gen",
        ":unfold_splat_constant_pass",
        "//tflite/converter:tensorflow_lite",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:case",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:conv",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:custom_call",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:dot_general",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:fft",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:gather",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:get_dimension_size",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:if",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:iota",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:pad",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:reduce",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:reduce_window",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:slice",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:sort",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:util",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:while",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
    ],
    alwayslink = True,
)

gentbl_cc_library(
    name = "prepare_hlo_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/generated_prepare_hlo.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/prepare_hlo.td",
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncTdFiles",
        "@local_xla//xla/mlir_hlo:hlo_ops_td_files",
    ],
)

cc_library(
    name = "prepare_hlo",
    srcs = [
        "transforms/prepare_hlo.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":passes_inc_gen",
        ":prepare_hlo_inc_gen",
        "//tflite/converter:tensorflow_lite_ops",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:conv",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:conv_util",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:custom_call",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:fft",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:pad_util",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:reduce_window",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:slice",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
    alwayslink = True,
)

cc_library(
    name = "tfl_legalize_chlo",
    srcs = [
        "transforms/tfl_legalize_chlo.cc",
    ],
    hdrs = ["transforms/stablehlo_passes.h"],
    compatible_with = get_compatible_with_portable(),
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":passes_inc_gen",
        "//tflite/converter:tensorflow_lite",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:gelu",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tf_legalize_hlo",
    srcs = [
        "transforms/generated_legalize_hlo.inc",
        "transforms/legalize_hlo.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
        "transforms/stablehlo_passes.h.inc",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":hlo_legalize_tf_inc_gen",
        ":hlo_matchers",
        ":passes_inc_gen",
        "//tflite/converter:tensorflow_lite",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:reduce",
        "//tflite/converter/stablehlo/transforms/legalize_hlo_conversions:util",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:lower_tf_lib",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tf_device_pass_inc_gen",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@stablehlo//:broadcast_utils",
        "@stablehlo//:chlo_ops",
    ],
)

td_library(
    name = "composite_td_files",
    srcs = [
        "transforms/composite_avg_pool_patterns.td",
        "transforms/composite_utils.td",
    ],
    compatible_with = get_compatible_with_portable(),
    includes = [
        "/external/org_tensorflow",
        "/third_party/tensorflow",
    ],
    deps = [
        "//tflite/converter:tensorflow_lite_ops_td_files",
        "@llvm-project//mlir:FuncTdFiles",
        "@local_xla//xla/mlir_hlo:hlo_ops_td_files",
    ],
)

cc_library(
    name = "composite_utils",
    srcs = ["transforms/composite_utils.cc"],
    hdrs = ["transforms/composite_utils.h"],
    deps = [
        "//tflite/converter:tensorflow_lite",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "lift_callsite_loc_caller",
    srcs = ["transforms/torch/lift_callsite_loc_caller_pass.cc"],
    copts = ["-Ithird_party"],
    deps = [
        ":passes_inc_gen",
        ":prepare_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = True,
)

cc_library(
    name = "legalize_qdq_custom_call",
    srcs = ["transforms/torch/legalize_qdq_custom_call_pass.cc"],
    copts = ["-Ithird_party"],
    deps = [
        ":passes_inc_gen",
        ":prepare_hlo",
        "@com_google_absl//absl/log",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = True,
)

cc_library(
    name = "build_stablehlo_composite",
    srcs = ["transforms/torch/build_stablehlo_composite_pass.cc"],
    copts = ["-Ithird_party"],
    deps = [
        ":passes_inc_gen",
        ":prepare_hlo",
        "@com_google_absl//absl/strings",
        "@jsoncpp_git//:jsoncpp",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = True,
)

cc_library(
    name = "composite_lowering",
    srcs = [
        "transforms/composite_avg_pool.cc",
        "transforms/composite_lowering_pass.cc",
    ],
    hdrs = [
        "transforms/composite_avg_pool.h",
        "transforms/stablehlo_passes.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":composite_lowering_inc_gen",
        ":composite_utils",
        ":passes_inc_gen",
        "//tflite/converter:tensorflow_lite",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
    alwayslink = True,
)

cc_library(
    name = "optimize_layout",
    srcs = [
        "transforms/optimize_layout.cc",
    ],
    hdrs = ["transforms/stablehlo_passes.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

gentbl_cc_library(
    name = "composite_lowering_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/generated_composite_lowering.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/composite_lowering_patterns.td",
    deps = [
        ":composite_td_files",
        ":composite_utils",
        "//tflite/converter:tensorflow_lite_ops_td_files",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncTdFiles",
        "@local_xla//xla/mlir_hlo:hlo_ops_td_files",
    ],
)

tf_cc_binary(
    name = "odml_to_stablehlo",
    srcs = [
        "odml_to_stablehlo.cc",
    ],
    copts = ["-O3"],
    deprecation = "odml_to_stablehlo is being deprecated, please use TFlite converter with flag: " +
                  "converter.target_spec.supported_ops = " +
                  " [tf.lite.OpsSet.EXPERIMENTAL_STABLEHLO_OPS]",
    deps = [
        ":check_accepted_ops_pass",
        ":op_stat_pass",
        ":stablehlo_util",
        ":transforms",
        "//tflite/converter:canonicalize_boundary_value",
        "//tflite/converter:flatbuffer_export",
        "//tflite/converter:tf_to_tfl_flatbuffer",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:RegisterAllDialects",
        "@llvm-project//mlir:RegisterAllPasses",
        "@llvm-project//mlir:Support",
        "@local_xla//xla/mlir/framework/transforms:passes",
        "@local_xla//xla/mlir_hlo:all_passes",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/compiler/mlir:init_mlir",
        "@org_tensorflow//tensorflow/compiler/mlir:passes",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow:quantize_preprocess",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow:tf_quant_ops",
        "@org_tensorflow//tensorflow/compiler/mlir/stablehlo:legalize_tf",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tf_graph_optimization_pass",
        "@org_tensorflow//tensorflow/compiler/mlir/tf2xla:compile_mlir_util",
        "@org_tensorflow//tensorflow/compiler/mlir/tf2xla/transforms:xla_legalize_tf",
        "@org_tensorflow//tensorflow/core:core_cpu_base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/ir/types:Dialect",
        "@stablehlo//:register",
    ],
)

tf_cc_binary(
    name = "odml-to-stablehlo-opt",
    testonly = True,
    tags = ["hostonly"],
    deps = [
        ":compose_uniform_quantized_type_pass",
        ":legalize_stablehlo_composite_to_tfl_custom",
        ":legalize_stablehlo_custom_call_to_composite",
        ":legalize_stablehlo_to_vhlo_pass",
        ":legalize_tf_xla_call_module_to_stablehlo_pass",
        ":optimize",
        ":passes_inc_gen",
        ":prepare_hlo",
        ":stablehlo_fuse_convolution_pass",
        ":stablehlo_unfuse_batch_norm_pass",
        ":tf_legalize_hlo",
        ":tfl_legalize_chlo",
        ":tfl_legalize_hlo",
        ":tfl_stablehlo",
        ":uniform_quantized_stablehlo_to_tfl_pass",
        "//tflite/converter:litert_mlir_opt_main",
        "@org_tensorflow//tensorflow/compiler/mlir:passes",
        "@org_tensorflow//tensorflow/compiler/mlir/stablehlo:fold_broadcast_pass",
        "@org_tensorflow//tensorflow/compiler/mlir/stablehlo:fuse_convolution_pass",
        "@org_tensorflow//tensorflow/compiler/mlir/stablehlo:tf_stablehlo",
        "@org_tensorflow//tensorflow/compiler/mlir/stablehlo:unfuse_batch_norm_pass",
    ],
)

exports_files([
    "run_lit.sh",
])
