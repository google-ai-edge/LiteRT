load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "if_google", "if_oss", "tf_cc_binary", "tf_cc_test", "tf_native_cc_binary")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "filegroup", "get_compatible_with_portable")
load(
    "@org_tensorflow//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)
load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# Placeholder: load py_proto_library
load("//tflite/converter:build_def.bzl", "tflite_copts_warnings")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        ":friends",
        "//configs/devtools/hawkeye/tflite:__subpackages__",
        "//litert:__subpackages__",
        "//tflite/experimental/tf_runtime:__subpackages__",
        "//tflite/testing:__subpackages__",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//research/language_modeling/sentence_explorer/ondevice/...",
        "//tflite/...",
        "//waymo/accelerator/alpine/tools/...",
        "//waymo/ml/compiler/mlir/...",
    ],
)

exports_files(glob([
    "testdata/*.bin",
]))

filegroup(
    name = "tflite_internal_cc_3p_api_deps_src",
    srcs = [
        "allocation.cc",
        "allocation.h",
        "mmap_allocation.cc",
    ],
    visibility = ["//tflite:__pkg__"],
)

td_library(
    name = "tensorflow_lite_ops_td_files",
    srcs = [
        "ir/tfl_op_enums.td",
        "ir/tfl_op_interfaces.td",
        "ir/tfl_ops.td",
    ],
    compatible_with = get_compatible_with_portable(),
    includes = [
        "/external/org_tensorflow",
        "/third_party/tensorflow",
    ],
    deps = [
        "//tflite/converter/quantization/common/quantization_lib:quantization_td_files",
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

td_library(
    name = "tensorflow_lite_patterns_td_files",
    srcs = [
        "transforms/legalize_patterns.td",
        "transforms/legalize_variables.td",
        "transforms/optimize_batch_matmul.td",
        "transforms/optimize_broadcast_like_patterns.td",
        "transforms/optimize_patterns.td",
        "transforms/post_quantize_patterns.td",
        "transforms/prepare_patterns.td",
        "transforms/quantization/strict_quantize_patterns.td",
        "transforms/quantize_by_converter_patterns.td",
        "transforms/quantize_patterns.td",
        "transforms/tensorlist_patterns.td",
        "utils/utils.td",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:private"],
    deps = [
        ":tensorflow_lite_ops_td_files",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_optimize_td_files",
    ],
)

gentbl_cc_library(
    name = "tensorflow_lite_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/passes.h.inc": [
        "-gen-pass-decls",
        "-name=TensorFlowLiteTd",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "tensorflow_lite_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "ir/tfl_ops.h.inc": ["-gen-op-decls"],
        "ir/tfl_ops.cc.inc": ["-gen-op-defs"],
        "g3doc/tfl_ops.md": [
            "-gen-dialect-doc",
            "-dialect=tfl",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tfl_ops.td",
    deps = [
        ":tensorflow_lite_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "tensorflow_lite_op_interfaces_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "ir/tfl_ops_interface.h.inc": ["-gen-op-interface-decls"],
        "ir/tfl_ops_interface.cc.inc": ["-gen-op-interface-defs"],
        "ir/tfl_ops_dialect.h.inc": ["-gen-dialect-decls"],
        "ir/tfl_ops_dialect.cc.inc": ["-gen-dialect-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tfl_op_interfaces.td",
    deps = [
        ":tensorflow_lite_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "tensorflow_lite_op_enums_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "ir/tfl_ops_enums.h.inc": ["-gen-enum-decls"],
        "ir/tfl_ops_enums.cc.inc": ["-gen-enum-defs"],
        "ir/tfl_ops_attrdefs.h.inc": ["-gen-attrdef-decls"],
        "ir/tfl_ops_attrdefs.cc.inc": ["-gen-attrdef-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tfl_op_enums.td",
    deps = [
        ":tensorflow_lite_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "tensorflow_lite_prepare_tf_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/generated_prepare_tf.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/prepare_patterns.td",
    deps = [":tensorflow_lite_patterns_td_files"],
)

gentbl_cc_library(
    name = "tensorflow_lite_lower_static_tensor_list_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/generated_lower_static_tensor_list.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/tensorlist_patterns.td",
    deps = [":tensorflow_lite_patterns_td_files"],
)

gentbl_cc_library(
    name = "tensorflow_lite_legalize_tf_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/generated_legalize_tf.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/legalize_patterns.td",
    deps = [":tensorflow_lite_patterns_td_files"],
)

gentbl_cc_library(
    name = "tensorflow_lite_legalize_variables_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/generated_legalize_variables.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/legalize_variables.td",
    deps = [":tensorflow_lite_patterns_td_files"],
)

gentbl_cc_library(
    name = "tensorflow_lite_optimize_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/generated_optimize.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/optimize_patterns.td",
    deps = [":tensorflow_lite_patterns_td_files"],
)

gentbl_cc_library(
    name = "tensorflow_lite_optimize_batch_matmul_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/generated_optimize_batch_matmul.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/optimize_batch_matmul.td",
    deps = [":tensorflow_lite_patterns_td_files"],
)

gentbl_cc_library(
    name = "optimize_broadcast_like_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/generated_optimize_broadcast_like.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/optimize_broadcast_like_patterns.td",
    deps = [":tensorflow_lite_patterns_td_files"],
)

gentbl_cc_library(
    name = "tensorflow_lite_quantize_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/generated_quantize.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/quantize_patterns.td",
    deps = [":tensorflow_lite_patterns_td_files"],
)

gentbl_cc_library(
    name = "tensorflow_lite_strict_quantize_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/quantization/generated_strict_quantize.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/quantization/strict_quantize_patterns.td",
    deps = [":tensorflow_lite_patterns_td_files"],
)

gentbl_cc_library(
    name = "tensorflow_lite_quantize_by_converter_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/generated_quantize_by_converter.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/quantize_by_converter_patterns.td",
    deps = [":tensorflow_lite_patterns_td_files"],
)

gentbl_cc_library(
    name = "tensorflow_lite_post_quantize_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/generated_post_quantize.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/post_quantize_patterns.td",
    deps = [":tensorflow_lite_patterns_td_files"],
)

gentbl_cc_library(
    name = "tensorflow_lite_legalize_tensorlist_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/generated_legalize_tensorlist.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/legalize_tensorlist.td",
    deps = [":tensorflow_lite_patterns_td_files"],
)

cc_library(
    name = "validators",
    srcs = [
        "utils/validators.cc",
    ],
    hdrs = [
        "utils/validators.h",
    ],
    deps = [
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "stateful_error_reporter",
    hdrs = ["stateful_error_reporter.h"],
    compatible_with = get_compatible_with_portable(),
    deps = ["//tflite/converter/core/api:error_reporter"],
)

gentbl_cc_library(
    name = "tensorflow_lite_canonicalize_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"ir/tfl_canonicalize.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tfl_canonicalize.td",
    deps = [":tensorflow_lite_patterns_td_files"],
)

cc_library(
    name = "register_lite_dialects",
    testonly = True,
    srcs = ["register_lite_dialects.cc"],
    hdrs = ["register_lite_dialects.h"],
    deps = [
        "//tflite/converter:tensorflow_lite",
        "//tflite/converter/quantization/ir:QuantOps",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "register_lite_dialects_test",
    srcs = ["register_lite_dialects_test.cc"],
    deps = [
        ":register_lite_dialects",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
    ],
)

# Don't add dependencies to this binary. It's only a wrapper around litert_mlir_opt_main.
tf_cc_binary(
    name = "litert-opt",
    testonly = True,
    deps = [
        ":litert_mlir_opt_main",
    ],
)

cc_library(
    name = "litert_mlir_opt_main",
    testonly = True,
    srcs = ["litert_mlir_opt_main.cc"],
    deps = [
        ":register_lite_dialects",
        ":tensorflow_lite",
        ":tf_tfl_passes",  # buildcleaner:keep
        "//tflite/converter:lift_tflite_flex_ops",
        "//tflite/converter/quantization/ir:QuantOps",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:RegisterAllDialects",
        "@llvm-project//mlir:RegisterAllPasses",
        "@org_tensorflow//tensorflow/compiler/mlir:init_mlir",
        "@org_tensorflow//tensorflow/compiler/mlir:passes",
        "@org_tensorflow//tensorflow/compiler/mlir:register_common_dialects",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
    ],
)

cc_library(
    name = "utils",
    hdrs = ["utils/utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "utils_test",
    srcs = ["utils/utils_test.cc"],
    deps = [
        ":utils",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "attribute_utils",
    srcs = ["utils/attribute_utils.cc"],
    hdrs = ["utils/attribute_utils.h"],
    deps = [
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "pass_options",
    hdrs = ["transforms/pass_options.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "pass",
    hdrs = ["transforms/pass.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":pass_options",
        ":pass_options_setter",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "pipeline",
    hdrs = ["transforms/pipeline.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":pass",
        ":pass_options",
        ":pass_options_setter",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "pass_options_setter",
    hdrs = ["transforms/pass_options_setter.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "converter_pass_options_setter",
    srcs = ["transforms/converter_pass_options_setter.cc"],
    hdrs = ["transforms/converter_pass_options_setter.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":converter_flags_proto_cc",
        ":optimize_broadcast_like_pass_options",
        ":optimize_pass_options",
        ":pass_options",
        ":pass_options_setter",
        ":variable_freezing_pipeline_options",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "tensorflow_lite_ops",
    srcs = [
        "ir/tfl_ops.cc",
        "ir/tfl_ops.cc.inc",
        "ir/tfl_ops.h.inc",
        "ir/tfl_ops_attrdefs.h.inc",
        "ir/tfl_ops_dialect.cc.inc",
        "ir/tfl_ops_dialect.h.inc",
        "ir/tfl_ops_enums.cc.inc",
        "ir/tfl_ops_enums.h.inc",
        "ir/tfl_ops_interface.cc.inc",
        "ir/tfl_ops_interface.h.inc",
        "runtime_verifiers.inc",
    ],
    hdrs = [
        "ir/tfl_ops.h",
    ],
    deps = [
        ":converter_inc",
        ":cost_estimators",
        ":shape_and_size_utils",
        ":tensorflow_lite_canonicalize_inc_gen",
        ":tensorflow_lite_op_enums_inc_gen",
        ":tensorflow_lite_op_interfaces_inc_gen",
        ":tensorflow_lite_ops_inc_gen",
        ":utils",
        "//tflite/converter/quantization/common/quantization_lib",
        "//tflite/converter/quantization/ir:QuantOps",
        "//tflite/converter/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_op_interfaces",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_traits",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core/platform:status",
    ],
)

cc_library(
    name = "pass_registry_utils",
    hdrs = ["transforms/pass_registry_utils.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":pass_options",
        ":pipeline",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "variable_freezing_pipeline_options",
    hdrs = ["transforms/variable_freezing_pipeline_options.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "variable_freezing_pipeline",
    srcs = ["transforms/variable_freezing_pipeline.cc"],
    hdrs = ["transforms/variable_freezing_pipeline.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":pass_registry_utils",
        ":pipeline",
        ":tensorflow_lite_tf_unfreeze_global_tensors",
        ":variable_freezing_pipeline_options",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tf_saved_model_passes",
    ],
)

cc_library(
    name = "tensorflow_lite",
    srcs = [
        "ir/tfl_canonicalize.inc",
    ],
    hdrs = [
        "ir/tfl_ops.h",
        "transforms/canonicalize_boundary_value_pass.h",
        "transforms/cleanup_optimization_barrier_pass.h",
        "transforms/optimize_batch_matmul_pass.h",
        "transforms/optimize_broadcast_like_pass.h",
        "transforms/optimize_pass.h",
        "transforms/pass_registry_utils.h",
        "transforms/passes.h",
        "transforms/push_transpose_through_ewise_pass.h",
        "transforms/tf_legalizations/analyze_variables_pass.h",
        "transforms/tf_legalizations/legalize_tensorlist_pass.h",
        "transforms/tf_legalizations/while_loop_outline_pass.h",
        "transforms/tflite_passes/split_merged_operands_pass.h",
        "transforms/tflite_passes/unfold_large_splat_constants_pass.h",
        "transforms/unfreeze_global_constants.h",
        "utils/attribute_utils.h",
        "utils/utils.h",
    ],
    deps = [
        ":attribute_utils",
        ":canonicalize_boundary_value",
        ":cleanup_optimization_barrier",
        ":converter_inc",
        ":cost_estimators",
        ":optimize_broadcast_like_pass",
        ":optimize_broadcast_like_pass_options",
        ":optimize_pass_options",
        ":pass",
        ":pass_options",
        ":pass_registry_utils",
        ":pipeline",
        ":shape_and_size_utils",
        ":tensorflow_lite_canonicalize_inc_gen",
        ":tensorflow_lite_legalize_tf_analyze_variables",
        ":tensorflow_lite_legalize_tf_legalize_tensorlist",
        ":tensorflow_lite_legalize_tf_while_loop_outline",
        ":tensorflow_lite_op_enums_inc_gen",
        ":tensorflow_lite_op_interfaces_inc_gen",
        ":tensorflow_lite_ops",
        ":tensorflow_lite_ops_inc_gen",
        ":tensorflow_lite_optimize",
        ":tensorflow_lite_passes_inc_gen",
        ":tensorflow_lite_push_transpose_through_ewise_pass",
        ":tensorflow_lite_split_merged_operands",
        ":tensorflow_lite_tf_unfreeze_global_tensors",
        ":tensorflow_lite_unfold_large_splat_constants",
        "//tflite/converter/quantization/common/quantization_lib",
        "//tflite/converter/quantization/common/quantization_lib:quantization_config",
        "//tflite/converter/quantization/ir:QuantOps",
        "//tflite/converter/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_op_interfaces",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_traits",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@org_tensorflow//tensorflow/core:framework",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "variables_utils",
    srcs = [
        "utils/variables_utils.cc",
    ],
    hdrs = [
        "utils/variables_utils.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
    ],
)

cc_library(
    name = "shape_and_size_utils",
    srcs = [
        "utils/shape_and_size_utils.cc",
    ],
    hdrs = [
        "utils/shape_and_size_utils.h",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "mlir_module_utils",
    hdrs = ["utils/mlir_module_utils.h"],
    deps = [
        ":const_tensor_utils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
    ],
)

tf_cc_test(
    name = "shape_and_size_utils_test",
    size = "small",
    srcs = ["utils/shape_and_size_utils_test.cc"],
    deps = [
        ":shape_and_size_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "cost_estimators",
    hdrs = [
        "utils/arithmetic_count_util.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "constant_utils",
    srcs = [
        "utils/constant_utils.cc",
    ],
    hdrs = [
        "utils/constant_utils.h",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mangling_util",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/platform:status",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "low_bit_utils",
    srcs = [
        "utils/low_bit_utils.cc",
    ],
    hdrs = [
        "utils/low_bit_utils.h",
    ],
)

cc_library(
    name = "lstm_utils",
    srcs = [
        "utils/lstm_utils.cc",
    ],
    hdrs = [
        "utils/lstm_utils.h",
    ],
    deps = [
        ":tensorflow_lite",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
    ],
)

cc_library(
    name = "metadata_utils",
    hdrs = [
        "utils/metadata_utils.h",
    ],
)

cc_library(
    name = "fake_quant_utils",
    srcs = [
        "utils/fake_quant_utils.cc",
    ],
    hdrs = [
        "utils/fake_quant_utils.h",
    ],
    deps = [
        ":tensorflow_lite",
        "//tflite/converter/quantization/common/quantization_lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
    ],
)

cc_library(
    name = "nms_utils",
    srcs = [
        "utils/nms_utils.cc",
    ],
    hdrs = [
        "utils/nms_utils.h",
    ],
    deps = [
        ":tensorflow_lite",
        "@flatbuffers",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
    ],
)

cc_library(
    name = "tftext_utils",
    srcs = [
        "utils/tftext_utils.cc",
    ],
    hdrs = [
        "utils/tftext_utils.h",
    ],
    deps = [
        ":tensorflow_lite",
        "@flatbuffers",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core/ir/types:Dialect",
    ],
)

tf_cc_test(
    name = "tftext_utils_test",
    size = "small",
    srcs = ["utils/tftext_utils_test.cc"],
    deps = [
        ":tftext_utils",
        "@com_google_absl//absl/status",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:test_main",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "perception_ops_utils",
    srcs = [
        "utils/perception_ops_utils.cc",
    ],
    hdrs = [
        "utils/perception_ops_utils.h",
    ],
    deps = [
        ":tensorflow_lite",
        "//tflite/converter/core/c:tflite_common",
        "@flatbuffers//:runtime_cc",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
    ],
)

cc_library(
    name = "stateful_ops_utils",
    srcs = [
        "utils/stateful_ops_utils.cc",
    ],
    hdrs = [
        "utils/stateful_ops_utils.h",
    ],
    deps = [
        ":tensorflow_lite_ops",
        ":utils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "lstm_utils_test",
    size = "small",
    srcs = ["utils/lstm_utils_test.cc"],
    deps = [
        ":lstm_utils",
        ":tensorflow_lite",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "perception_ops_utils_test",
    size = "small",
    srcs = ["utils/perception_ops_utils_test.cc"],
    deps = [
        ":perception_ops_utils",
        ":tensorflow_lite",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "canonicalize_boundary_value",
    srcs = [
        "transforms/canonicalize_boundary_value_pass.cc",
    ],
    hdrs = [
        "transforms/canonicalize_boundary_value_pass.h",
    ],
    deps = [
        ":pass",
        ":utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "cleanup_optimization_barrier",
    srcs = [
        "transforms/cleanup_optimization_barrier_pass.cc",
    ],
    hdrs = [
        "transforms/cleanup_optimization_barrier_pass.h",
    ],
    deps = [
        ":pass",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "tensorflow_lite_legalize_tf_analyze_variables",
    srcs = [
        "transforms/tf_legalizations/analyze_variables_pass.cc",
    ],
    hdrs = [
        "transforms/tf_legalizations/analyze_variables_pass.h",
    ],
    deps = [
        ":pass",
        ":tensorflow_lite_ops",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
    ],
)

cc_library(
    name = "tensorflow_lite_legalize_tf_legalize_tensorlist",
    srcs = [
        "transforms/tf_legalizations/legalize_tensorlist_pass.cc",
    ],
    hdrs = [
        "transforms/tf_legalizations/legalize_tensorlist_pass.h",
    ],
    deps = [
        ":convert_type",
        ":pass",
        ":tensorflow_lite_legalize_tensorlist_inc_gen",
        ":tensorflow_lite_ops",
        ":utils",
        "//tflite/converter/schema:schema_fbs",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
    ],
)

cc_library(
    name = "tensorflow_lite_legalize_tf_while_loop_outline",
    srcs = [
        "transforms/tf_legalizations/while_loop_outline_pass.cc",
    ],
    hdrs = [
        "transforms/tf_legalizations/while_loop_outline_pass.h",
    ],
    deps = [
        ":pass",
        ":tensorflow_lite_ops",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
    ],
)

cc_library(
    name = "tensorflow_lite_unfold_large_splat_constants",
    srcs = [
        "transforms/tflite_passes/unfold_large_splat_constants_pass.cc",
    ],
    hdrs = [
        "transforms/tflite_passes/unfold_large_splat_constants_pass.h",
    ],
    deps = [
        ":pass",
        ":tensorflow_lite_ops",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
    ],
)

cc_library(
    name = "tensorflow_lite_split_merged_operands",
    srcs = [
        "transforms/tflite_passes/split_merged_operands_pass.cc",
    ],
    hdrs = [
        "transforms/tflite_passes/split_merged_operands_pass.h",
    ],
    deps = [
        ":pass",
        ":pass_options",
        ":stateful_ops_utils",
        ":tensorflow_lite_ops",
        ":utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "tensorflow_lite_legalize_tf",
    srcs = [
        "transforms/dilated_conv.cc",
        "transforms/generated_legalize_tf.inc",
        "transforms/generated_legalize_variables.inc",
        "transforms/generated_lower_static_tensor_list.inc",
        "transforms/generated_prepare_tf.inc",
        "transforms/get_arithmetic_count.cc",
        "transforms/if_outline.cc",
        "transforms/insert_call_once_op.cc",
        "transforms/legalize_hashtables.cc",
        "transforms/legalize_jax_random.cc",
        "transforms/legalize_tf.cc",
        "transforms/legalize_tf_while.cc",
        "transforms/legalize_variables.cc",
        "transforms/lower_static_tensor_list.cc",
        "transforms/optimize_functional_ops.cc",
        "transforms/partitioned_topological_sort.cc",
        "transforms/pin_ops_with_side_effects.cc",
        "transforms/prepare_composite_functions_tf.cc",
        "transforms/prepare_tf.cc",
        "transforms/raise_custom_ops.cc",
        "transforms/reduce_type_precision.cc",
        "transforms/reduce_while_operands.cc",
        "transforms/runtime_verify.cc",
        "transforms/trim_functions_tf.cc",
    ],
    hdrs = [
        "ir/tfl_ops_interface.h.inc",
        "transforms/dilated_conv.h",
        "transforms/passes.h",
    ],
    deps = [
        ":attribute_utils",
        ":constant_utils",
        ":convert_type",
        ":cost_estimators",
        ":fake_quant_utils",
        ":lstm_utils",
        ":nms_utils",
        ":optimize_broadcast_like_pass_options",
        ":perception_ops_utils",
        ":shape_and_size_utils",
        ":stateful_ops_utils",
        ":tensorflow_lite",
        ":tensorflow_lite_legalize_tf_analyze_variables",
        ":tensorflow_lite_legalize_tf_inc_gen",
        ":tensorflow_lite_legalize_tf_legalize_tensorlist",
        ":tensorflow_lite_legalize_variables_inc_gen",
        ":tensorflow_lite_lower_static_tensor_list_inc_gen",
        ":tensorflow_lite_passes_inc_gen",
        ":tensorflow_lite_prepare_tf_inc_gen",
        ":tensorflow_lite_tf_unfreeze_global_tensors",
        ":tftext_utils",
        ":validators",
        ":variables_utils",
        "//tflite/converter/quantization/common/quantization_lib",
        "//tflite/converter/quantization/common/quantization_lib:quantization_config",
        "//tflite/converter/quantization/ir:QuantOps",
        "//tflite/converter/schema:schema_fbs",
        "//tflite/converter/stablehlo:optimize_layout",
        "//tflite/converter/stablehlo:prepare_hlo",
        "//tflite/converter/stablehlo:tf_legalize_hlo",
        "//tflite/converter/stablehlo:tfl_legalize_chlo",
        "//tflite/converter/stablehlo:tfl_legalize_hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/common/ir:QuantOps",
        "@org_tensorflow//tensorflow/compiler/mlir/stablehlo:legalize_tf",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:error_util",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mangling_util",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:verification_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:lower_tf_lib",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:unroll_batch_matmul_pass",
        "@org_tensorflow//tensorflow/compiler/mlir/tf2xla/transforms:xla_legalize_tf",
        "@org_tensorflow//tensorflow/compiler/mlir/tf2xla/transforms:xla_legalize_tf_with_tf2xla",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/kernels:tensor_list",
        "@stablehlo//:stablehlo_ops",
        "@xla//xla/mlir_hlo",
        "@xla//xla/mlir_hlo:type_conversion",
        "@xla//xla/mlir_hlo:unfuse_batch_norm",
    ],
)

cc_library(
    name = "tensorflow_lite_optimize",
    srcs = [
        "transforms/generated_optimize.inc",
        "transforms/optimize_pass.cc",
    ],
    hdrs = [
        "transforms/optimize_pass.h",
    ],
    deps = [
        ":attribute_utils",
        ":constant_utils",
        ":convert_type",
        ":optimize_pass_options",
        ":pass",
        ":tensorflow_lite_ops",
        ":tensorflow_lite_optimize_inc_gen",
        ":utils",
        ":validators",
        "//tflite/converter/quantization/common/quantization_lib",
        "//tflite/converter/quantization/common/quantization_lib:quantization_config",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:verification_utils",
    ],
)

cc_library(
    name = "optimize_batch_matmul_utils",
    srcs = ["transforms/tflite_passes/optimize_batch_matmul_utils.cc"],
    hdrs = ["transforms/tflite_passes/optimize_batch_matmul_utils.h"],
    deps = [
        ":utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "optimize_batch_matmul_utils_test",
    srcs = ["transforms/tflite_passes/optimize_batch_matmul_utils_test.cc"],
    deps = [
        ":optimize_batch_matmul_utils",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tensorflow_lite_optimize_batch_matmul",
    srcs = [
        "transforms/generated_optimize_batch_matmul.inc",
        "transforms/optimize_batch_matmul_pass.cc",
    ],
    hdrs = [
        "transforms/optimize_batch_matmul_pass.h",
    ],
    deps = [
        ":convert_type",
        ":optimize_batch_matmul_utils",
        ":pass",
        ":pass_options",
        ":tensorflow_lite_ops",
        ":tensorflow_lite_optimize_batch_matmul_inc_gen",
        ":tensorflow_lite_passes_inc_gen",
        ":utils",
        ":validators",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:verification_utils",
    ],
)

cc_library(
    name = "optimize_broadcast_like_pass",
    srcs = [
        "transforms/optimize_broadcast_like_pass.cc",
    ],
    hdrs = [
        "transforms/optimize_broadcast_like_pass.h",
    ],
    deps = [
        ":optimize_broadcast_like_inc_gen",
        ":optimize_broadcast_like_pass_options",
        ":pass",
        ":tensorflow_lite_ops",
        ":utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "tensorflow_lite_push_transpose_through_ewise_pass",
    srcs = [
        "transforms/push_transpose_through_ewise_pass.cc",
    ],
    hdrs = [
        "transforms/push_transpose_through_ewise_pass.h",
    ],
    deps = [
        ":pass",
        ":tensorflow_lite_ops",
        ":utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "tensorflow_lite_tf_unfreeze_global_tensors",
    srcs = [
        "transforms/unfreeze_global_constants.cc",
    ],
    hdrs = [
        "transforms/unfreeze_global_constants.h",
    ],
    deps = [
        ":pass",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
    ],
)

cc_library(
    name = "tensorflow_lite_quantize",
    srcs = [
        "transforms/decompose_hybrid_quantization.cc",
        "transforms/default_quant_params.cc",
        "transforms/generated_post_quantize.inc",
        "transforms/generated_quantize.inc",
        "transforms/lower_quant_annotations_helper.cc",
        "transforms/lower_quant_annotations_pass.cc",
        "transforms/modify_io_nodes.cc",
        "transforms/optimize_op_order.cc",
        "transforms/post_quantize.cc",
        "transforms/prepare_quantize.cc",
        "transforms/prepare_quantize_dynamic_range.cc",
        "transforms/prepare_quantize_helper.cc",
        "transforms/quantization/bias_quantizer_pass.cc",
        "transforms/quantization/fuse_qdq_pass.cc",
        "transforms/quantization/propagate_qsv_pass.cc",
        "transforms/quantization/quant_utils.cc",
        "transforms/quantize.cc",
        "transforms/quantize_variables.cc",
        "utils/generated_op_quant_spec_getters.inc",
    ],
    hdrs = [
        "transforms/lower_quant_annotations_helper.h",
        "transforms/passes.h",
        "transforms/prepare_quantize_helper.h",
        "transforms/quantization/quant_utils.h",
    ],
    deps = [
        "convert_type",
        ":op_quant_spec_getters_inc",
        ":optimize_broadcast_like_pass_options",
        ":shape_and_size_utils",
        ":stateful_ops_utils",
        ":tensorflow_lite",
        ":tensorflow_lite_passes_inc_gen",
        ":tensorflow_lite_post_quantize_inc_gen",
        ":tensorflow_lite_quantize_by_converter_inc_gen",
        ":tensorflow_lite_quantize_inc_gen",
        ":tensorflow_lite_strict_quantize_inc_gen",
        ":validators",
        "//tflite/converter/quantization/common/quantization_lib",
        "//tflite/converter/quantization/common/quantization_lib:quantization_config",
        "//tflite/converter/quantization/common/quantization_lib:tfl_quantization_driver",
        "//tflite/converter/quantization/ir:QuantOps",
        "//tflite/converter/quantization/lite:tfl_to_std",
        "//tflite/converter/schema:schema_fbs",
        "//tflite/converter/tools/optimize:operator_property",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/common:uniform_quantized_types",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/common/ir:QuantOps",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/platform:logging",
        "@stablehlo//:stablehlo_ops",
        "@xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "tensorflow_lite_d2s",
    srcs = [
        "transforms/dense_to_sparse_pass.cc",
    ],
    hdrs = [
        "transforms/dense_to_sparse_pass.h",
    ],
    deps = [
        ":pass",
        ":pass_options",
        ":tensorflow_lite_ops",
        "//tflite/converter/kernels/internal/utils:sparsity_format_converter",
        "@com_google_absl//absl/memory",
        "@eigen_archive//:eigen3",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "lift_tflite_flex_ops",
    srcs = [
        "transforms/lift_tflite_flex_ops.cc",
    ],
    hdrs = [
        "transforms/lift_tflite_flex_ops.h",
    ],
    deps = [
        ":tensorflow_lite",
        ":tensorflow_lite_passes_inc_gen",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:convert_attr",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

filegroup(
    name = "generated_op_quant_spec_getters",
    srcs = [
        "utils/generated_op_quant_spec_getters.inc",
    ],
)

gentbl_cc_library(
    name = "op_quant_spec_getters_inc",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"utils/generated_op_quant_spec_getters.inc": []},
    tblgen = "//tflite/converter/quantization:op_quant_spec_getters_gen",
    td_file = "ir/tfl_ops.td",
    deps = [
        ":tensorflow_lite_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "tflite_op_coverage_spec_inc",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"utils/tflite_op_coverage_spec.inc": []},
    tblgen = "//tflite/converter/quantization:tflite_op_coverage_spec_getters_gen",
    td_file = "ir/tfl_ops.td",
    deps = [
        ":tensorflow_lite_ops_td_files",
    ],
)

tf_native_cc_binary(
    name = "converter-gen",
    srcs = [
        "converter_gen.cc",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
        "@llvm-project//mlir:TableGen",
    ],
)

gentbl_cc_library(
    name = "converter_inc",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "operator_converters.inc": ["--gen-operator-converters"],
        "runtime_verifiers.inc": ["--gen-runtime-verifiers"],
    },
    tblgen = ":converter-gen",
    td_file = "ir/tfl_ops.td",
    test = 1,
    deps = [
        ":tensorflow_lite_ops_td_files",
        "//tflite/converter/quantization/common/quantization_lib:quantization_td_files",
    ],
)

cc_library(
    name = "flatbuffer_tflite_operator_lib",
    srcs = [
        "flatbuffer_operator.cc",
        "operator_converters.inc",
    ],
    hdrs = [
        "flatbuffer_operator.h",
    ],
    deps = [
        ":convert_type",
        ":converter_inc",
        ":tensorflow_lite",
        "//tflite/converter/core/c:tflite_common",
        "//tflite/converter/schema:schema_fbs",
        "//tflite/converter/schema:schema_fbs_with_mutable",
        "//tflite/converter/schema:schema_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@org_tensorflow//tensorflow/core/platform:errors",
        "@org_tensorflow//tensorflow/core/platform:status",
        "@org_tensorflow//tensorflow/core/platform:statusor",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:vhlo_ops",
        "@stablehlo//:vhlo_types",
        "@tsl//tsl/platform:status",
    ],
)

tf_native_cc_binary(
    name = "flatbuffer_to_string",
    srcs = ["flatbuffer_to_string.cc"],
    deps = [
        "//tflite/converter/core:absl_error_model_builder",
        "//tflite/converter/schema:schema_fbs_with_reflection",
        "@flatbuffers",
    ],
)

tf_native_cc_binary(
    name = "json_to_flatbuffer",
    srcs = ["json_to_flatbuffer.cc"],
    deps = [
        "//tflite/converter/schema:schema_fbs",
        "@flatbuffers",
    ],
)

cc_library(
    name = "flatbuffer_export",
    srcs = [
        "flatbuffer_export.cc",
    ],
    hdrs = [
        "flatbuffer_export.h",
        "flatbuffer_export_flags.h",
    ],
    deps = [
        ":const_tensor_utils",
        ":control_edges",
        ":convert_type",
        ":converter_flags_proto_cc",
        ":flatbuffer_tflite_operator_lib",
        ":lite_version",
        ":low_bit_utils",
        ":metadata_utils",
        ":mlir_module_utils",
        ":region_isolation",
        ":stateful_ops_utils",
        ":string_utils",
        ":tensorflow_lite",
        "//tflite/converter/core:macros",
        "//tflite/converter/core/c:tflite_common",
        "//tflite/converter/delegates/flex:allowlisted_flex_ops_lib",
        "//tflite/converter/experimental/remat:metadata_util",
        "//tflite/converter/metrics:converter_error_data_proto_cc",
        "//tflite/converter/metrics:error_collector_inst",
        "//tflite/converter/quantization/ir:QuantOps",
        "//tflite/converter/schema:debug_metadata_fbs_with_mutable",
        "//tflite/converter/schema:schema_conversion_utils",
        "//tflite/converter/schema:schema_fbs",
        "//tflite/converter/schema:schema_fbs_with_mutable",
        "//tflite/converter/tools/versioning",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:translate_utils",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:portable_gif_internal",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:vhlo_ops",
        "@tsl//tsl/platform:tstring",
    ],
)

cc_library(
    name = "flatbuffer_import",
    srcs = [
        "flatbuffer_import.cc",
    ],
    hdrs = [
        "flatbuffer_import.h",
    ],
    deps = [
        ":const_tensor_utils",
        ":control_edges",
        ":convert_type",
        ":flatbuffer_tflite_operator_lib",
        ":metadata_utils",
        ":offset_buffer",
        ":shape_and_size_utils",
        ":tensorflow_lite",
        "//tflite/converter/core:absl_error_model_builder",
        "//tflite/converter/experimental/remat:metadata_util",
        "//tflite/converter/quantization/common/quantization_lib",
        "//tflite/converter/quantization/ir:QuantOps",
        "//tflite/converter/schema:debug_metadata_fbs_with_mutable",
        "//tflite/converter/schema:schema_fbs",
        "//tflite/converter/schema:schema_fbs_with_mutable",
        "//tflite/converter/schema:schema_utils",
        "//tflite/converter/stablehlo:legalize_stablehlo_composite_to_tfl_custom",
        "//tflite/converter/stablehlo:legalize_stablehlo_to_vhlo_pass",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mangling_util",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/platform:errors",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:vhlo_ops",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "convert_type",
    srcs = [
        "utils/convert_type.cc",
    ],
    hdrs = [
        "utils/convert_type.h",
    ],
    deps = [
        ":tensorflow_lite_ops",
        "//tflite/converter/schema:schema_fbs",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/platform:errors",
    ],
)

cc_library(
    name = "region_isolation",
    srcs = [
        "utils/region_isolation.cc",
    ],
    hdrs = [
        "utils/region_isolation.h",
    ],
    deps = [
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

tf_cc_test(
    name = "region_isolation_test",
    srcs = [
        "utils/region_isolation_test.cc",
    ],
    deps = [
        ":region_isolation",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "optimize_pass_options",
    hdrs = ["transforms/optimize_pass_options.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "optimize_broadcast_like_pass_options",
    hdrs = ["transforms/optimize_broadcast_like_pass_options.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "flatbuffer_translate_lib",
    hdrs = [
        "flatbuffer_export.h",
        "flatbuffer_export_flags.h",
        "flatbuffer_import.h",
        "utils/convert_type.h",
    ],
    deps = [
        ":converter_flags_proto_cc",
        ":flatbuffer_export",
        ":flatbuffer_import",
        "//tflite/converter/schema:schema_fbs",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@org_tensorflow//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "flatbuffer_translate_registration",
    srcs = [
        "flatbuffer_translate.cc",
    ],
    deps = [
        ":flatbuffer_translate_lib",
        ":tensorflow_lite",
        "//tflite/converter/quantization/ir:QuantOps",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirTranslateMain",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/translate/tools:parsers",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:vhlo_ops",
    ],
    alwayslink = 1,
)

tf_cc_binary(
    name = "flatbuffer_translate",
    deps = [
        ":flatbuffer_translate_registration",
    ],
)

cc_library(
    name = "tf_tfl_translate_cl_options",
    srcs = [
        "tf_tfl_translate_cl.cc",
    ],
    hdrs = [
        "tf_tfl_translate_cl.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "common",
    hdrs = [
        "common/tfl_pass_config.h",
    ],
    deps = [
        "//tflite/converter:converter_flags_proto_cc",
        "//tflite/converter/quantization/common/quantization_lib:quantization_config",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
    ],
)

filegroup(
    name = "tf_tfl_translate_main",
    srcs = [
        "tf_tfl_translate.cc",
    ],
)

tf_cc_binary(
    name = "tf_tfl_translate",
    srcs = [
        ":tf_tfl_translate_main",
    ],
    # visibility = ["//visibility:public"], # copybara:uncomment
    deps = [
        ":common",
        ":converter_flags_proto_cc",
        ":flatbuffer_translate_lib",
        ":flatbuffer_translate_registration",
        ":tensorflow_lite",
        ":tf_tfl_translate_cl_options",
        ":tf_to_tfl_flatbuffer",
        "//tflite/converter/quantization/common/quantization_lib:quantization_config",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir:init_mlir",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "@org_tensorflow//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "@org_tensorflow//tensorflow/core:core_cpu_base",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/platform:errors",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
        "@xla//xla/hlo/translate/hlo_to_mhlo:translate",
        "@xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "tf_tfl_passes",
    srcs = ["tf_tfl_passes.cc"],
    hdrs = [
        "tf_tfl_passes.h",
    ],
    deps = [
        ":common",
        ":converter_flags_proto_cc",
        ":converter_pass_options_setter",
        ":fake_quant_utils",
        ":optimize_broadcast_like_pass",
        ":pass",
        ":pass_registry_utils",
        ":tensorflow_lite_d2s",  # buildcleaner: keep
        ":tensorflow_lite_legalize_tf",  # buildcleaner: keep
        ":tensorflow_lite_optimize",  # buildcleaner: keep
        ":tensorflow_lite_optimize_batch_matmul",  # buildcleaner: keep
        ":tensorflow_lite_push_transpose_through_ewise_pass",  # buildcleaner: keep
        ":tensorflow_lite_quantize",  # buildcleaner: keep
        ":variable_freezing_pipeline",
        "//tflite/converter/core:macros",
        "//tflite/converter/quantization:quantization_passes",
        "//tflite/converter/quantization/common/quantization_lib:quantization_config",
        "//tflite/converter/quantization/tensorflow:tf_quantization_passes",
        "//tflite/converter/stablehlo:build_stablehlo_composite",
        "//tflite/converter/stablehlo:compose_uniform_quantized_type_pass",
        "//tflite/converter/stablehlo:composite_lowering",
        "//tflite/converter/stablehlo:legalize_qdq_custom_call",
        "//tflite/converter/stablehlo:legalize_stablehlo_composite_to_tfl_custom",
        "//tflite/converter/stablehlo:legalize_tf_xla_call_module_to_stablehlo_pass",
        "//tflite/converter/stablehlo:lift_callsite_loc_caller",
        "//tflite/converter/stablehlo:prepare_hlo",  # buildcleaner: keep
        "//tflite/converter/stablehlo:tf_legalize_hlo",  # buildcleaner: keep
        "//tflite/converter/stablehlo:tfl_legalize_chlo",  # buildcleaner: keep
        "//tflite/converter/stablehlo:tfl_legalize_hlo",  # buildcleaner: keep
        "//tflite/converter/stablehlo:transforms",
        "//tflite/converter/stablehlo:uniform_quantized_stablehlo_to_tfl_pass",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/stablehlo:rename_entrypoint_to_main",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tf_saved_model_passes",
        "@org_tensorflow//tensorflow/core:core_cpu_base",
        "@stablehlo//:stablehlo_passes",
        "@xla//xla/mlir_hlo:mhlo_passes",
        "@xla//xla/mlir_hlo:stablehlo_extension_passes",
    ],
)

cc_library(
    name = "tf_to_tfl_flatbuffer",
    srcs = ["tf_to_tfl_flatbuffer.cc"],
    hdrs = [
        "tf_to_tfl_flatbuffer.h",
    ],
    deps = [
        ":common",
        ":converter_flags_proto_cc",
        ":flatbuffer_translate_lib",
        ":mlir_module_utils",
        ":tensorflow_lite",
        ":tf_tfl_passes",
        "//tflite/converter/debug",
        "//tflite/converter/experimental/remat:metadata_util",
        "//tflite/converter/metrics:converter_error_data_proto_cc",
        "//tflite/converter/metrics:error_collector_inst",
        "//tflite/converter/quantization/common/quantization_lib:quantization_config",
        "//tflite/converter/quantization/lite/toco_legacy:quantize_weights",
        "//tflite/converter/schema:schema_fbs",
        "//tflite/converter/stablehlo:legalize_stablehlo_composite_to_tfl_custom",
        "//tflite/converter/stablehlo:legalize_stablehlo_to_vhlo_pass",
        "//tflite/converter/stablehlo:op_stat_pass",
        "//tflite/converter/stablehlo:stablehlo_util",
        "//tflite/converter/stablehlo:transforms",
        "//tflite/converter/tools/optimize:reduced_precision_metadata",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@flatbuffers//:runtime_cc",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/stablehlo:quantize_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow:quantize_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow:quantize_preprocess",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:error_util",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:file_tf_mlir_translate",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mlir_import_options",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tf_dialect_lib",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/transforms:tf_dialect_passes",
        "@org_tensorflow//tensorflow/core:core_cpu_base",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/ir/types:Dialect",
        "@org_tensorflow//tensorflow/core/platform:status",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:vhlo_ops",
        "@tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "offset_buffer",
    hdrs = ["offset_buffer.h"],
)

cc_library(
    name = "const_tensor_utils",
    srcs = ["utils/const_tensor_utils.cc"],
    hdrs = ["utils/const_tensor_utils.h"],
    deps = [
        ":convert_type",
        ":low_bit_utils",
        ":string_utils",
        "//tflite/converter/schema:schema_fbs",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "string_utils",
    srcs = ["utils/string_utils.cc"],
    hdrs = ["utils/string_utils.h"],
    visibility = ["//tflite:__pkg__"],
)

exports_files(srcs = ["allocation.h"])

cc_library(
    name = "allocation",
    srcs = [
        "allocation.cc",
    ] + select({
        ":tflite_mmap_disabled": [
            "mmap_allocation_disabled.cc",
        ],
        "//conditions:default": [
            "mmap_allocation.cc",
        ],
    }),
    hdrs = [
        "allocation.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    visibility = ["//visibility:public"],
    deps = ["//tflite/converter/core/api:error_reporter"],
)

exports_files(srcs = ["utils/control_edges.h"])

cc_library(
    name = "control_edges",
    hdrs = ["utils/control_edges.h"],
    visibility = ["//tflite/converter/experimental/remat:__pkg__"],
)

tf_cc_test(
    name = "offset_buffer_test",
    srcs = ["offset_buffer_test.cc"],
    deps = [
        ":offset_buffer",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:test_main",
    ],
)

build_test(
    name = "tensorflow_lite_build_test",
    targets = [
        ":tensorflow_lite",
    ],
)

# LINT.IfChange

bool_flag(
    name = "enable_fuchsia_mmap",
    build_setting_default = True,
)

config_setting(
    name = "tflite_with_xnnpack_explicit_false",
    define_values = {"tflite_with_xnnpack": "false"},
)

# A config for enabling tensorflow profiler in TFLite. Currently, it only supports dynamic
# allocation. Add '--define=tflite_tensorflow_profiler=true' in your build command line to use it.
config_setting(
    name = "tensorflow_profiler_config",
    define_values = {"tflite_tensorflow_profiler": "true"},
)

config_setting(
    name = "fuchsia_mmap_disabled",
    constraint_values = if_google(
        ["@platforms//os:fuchsia"],
        [],
    ),
    flag_values = {
        ":enable_fuchsia_mmap": "False",
    },
    values = if_oss(
        # TODO(b/149248802) When we have a Fuchsia Bazel SDK update to use the values it sets.
        {"cpu": "fuchsia"},
        {},
    ),
)

config_setting(
    name = "tflite_mmap_disabled_true",
    values = {
        "copt": "-DTFLITE_MMAP_DISABLED",
    },
)

selects.config_setting_group(
    name = "tflite_mmap_disabled",
    match_any = [
        ":fuchsia_mmap_disabled",
        ":tflite_mmap_disabled_true",
        "@org_tensorflow//tensorflow:windows",
    ],
)
# LINT.ThenChange(//tflite/BUILD)

# LINT.IfChange(version)
cc_library(
    name = "lite_version",
    hdrs = ["version.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
)
# LINT.ThenChange(//tflite:version)

# OSS only: This target is header-only. Link `types_proto_cc_impl` only to
# `libtensorflow_framework.so` via `lib_internal_impl`. Do NOT link `types_flags_proto_cc_impl`
# directly unless the target does not link `libtensorflow_framework.so`.
tf_proto_library(
    name = "types_proto",
    srcs = ["types.proto"],
    make_default_target_header_only = True,
    visibility = ["//visibility:public"],
)

# OSS only: This target is header-only. Link `converter_flags_proto_cc_impl` only to
# `libtensorflow_framework.so` via `lib_internal_impl`. Do NOT link `converter_flags_proto_cc_impl`
# directly unless the target does not link `libtensorflow_framework.so`.
tf_proto_library(
    name = "converter_flags_proto",
    srcs = ["converter_flags.proto"],
    make_default_target_header_only = True,
    protodeps = [
        ":types_proto",
        "//tflite/converter/debug:debug_options_proto",
    ],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "model_flags_proto",
    srcs = ["model_flags.proto"],
    make_default_target_header_only = True,
    protodeps = [":types_proto"],
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "types_py_proto",
#     visibility = ["//visibility:public"],
#     deps = [":types_proto"],
# )
#
# py_proto_library(
#     name = "model_flags_py_proto",
#     visibility = ["//visibility:public"],
#     deps = [":model_flags_proto"],
# )
#
# py_proto_library(
#     name = "converter_flags_py_proto",
#     visibility = ["//visibility:public"],
#     deps = [":converter_flags_proto"],
# )
# copybara:uncomment_end

filegroup(
    name = "litfiles",
    srcs = glob(["runlit*py"]),
    visibility = ["//tflite/converter:__subpackages__"],
)
