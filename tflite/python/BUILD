load("@flatbuffers//:build_defs.bzl", "flatbuffer_py_library")
load("@org_tensorflow//tensorflow:pytype.default.bzl", "pytype_strict_contrib_test", "pytype_strict_library")
load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_binary", "py_strict_library", "py_strict_test")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tflite:special_rules.bzl", "internal_visibility_allowlist")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "@org_tensorflow//tensorflow:__subpackages__",
        "@org_tensorflow//tensorflow:internal",
        "@org_tensorflow//third_party/odml/infra/genai/conversion:__subpackages__",
        "@org_tensorflow//third_party/odml/model_customization/quantization:__subpackages__",
        "@org_tensorflow//third_party/py/ai_edge_torch:__subpackages__",
        "@org_tensorflow//third_party/py/tensorflow_federated:__subpackages__",
        "@org_tensorflow//third_party/tflite_micro:__subpackages__",
    ],
    licenses = ["notice"],
)

exports_files(["tflite_convert.py"])

flatbuffer_py_library(
    name = "schema_py",
    srcs = ["@org_tensorflow//tensorflow/compiler/mlir/lite/schema:schema.fbs"],
    visibility = ["//visibility:public"],
)

flatbuffer_py_library(
    name = "conversion_metadata_schema_py",
    srcs = ["@org_tensorflow//tensorflow/compiler/mlir/lite/schema:conversion_metadata.fbs"],
)

py_strict_library(
    name = "interpreter",
    srcs = [
        "interpreter.py",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "//tflite/python/interpreter_wrapper:_pywrap_tensorflow_interpreter_wrapper",
        "//tflite/python/metrics",
        "@org_tensorflow//tensorflow/python/util:tf_export",
        "@org_tensorflow//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "interpreter_test",
    srcs = ["interpreter_test.py"],
    data = [
        "//tflite:testdata/sparse_tensor.bin",
        "//tflite/python/testdata:interpreter_test_data",
        "//tflite/python/testdata:test_delegate.so",
    ],
    # Static linking is required because this loads a cc_binary as a shared
    # library, which would otherwise create ODR violations.
    # copybara:uncomment linking_mode = "static",
    tags = [
        "no_oss",  # TODO(b/190842754): Enable test in OSS.
    ],
    deps = [
        ":interpreter",
        #internal proto upb dep
        "@org_tensorflow//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tflite/python:lite",
        "//tflite/python/metrics",
        "//tflite/python/testdata:_pywrap_test_registerer",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
        "@org_tensorflow//tensorflow/python/platform:resource_loader",
    ],
)

py_strict_binary(
    name = "tflite_convert",
    srcs = ["tflite_convert.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":tflite_convert_main_lib",
        "//tflite/python:lite",
        "@absl_py//absl:app",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_strict_library(
    name = "tflite_convert_main_lib",
    srcs = ["tflite_convert.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":convert",
        "//tflite/toco:toco_flags_proto_py",
        "//tflite/toco/logging:gen_html",
        "@absl_py//absl:app",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python:tf2",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/platform:gfile",
        "@org_tensorflow//tensorflow/python/util:keras_deps",
    ],
)

py_strict_library(
    name = "test_util",
    srcs = ["test_util.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":lite",
        ":schema_py",
        ":schema_util",
        "//tflite/tools:visualize_lib",
    ],
)

py_strict_test(
    name = "test_util_test",
    srcs = ["test_util_test.py"],
    data = [
        "//tflite:testdata/add.bin",
        "//tflite:testdata/softplus_flex.bin",
    ],
    deps = [
        ":test_util",
        #internal proto upb dep
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
        "@org_tensorflow//tensorflow/python/platform:gfile",
        "@org_tensorflow//tensorflow/python/platform:resource_loader",
    ],
)

py_strict_test(
    name = "tflite_convert_test",
    srcs = ["tflite_convert_test.py"],
    data = [
        ":tflite_convert.par",
        "@tflite_mobilenet_ssd_quant_protobuf//:tflite_graph.pb",
    ],
    # Increased thread count for reducing timeout failures.
    shard_count = 10,
    tags = [
        "no_oss",
        "no_windows",
    ],
    deps = [
        ":convert",
        ":test_util",
        ":tflite_convert_main_lib",
        #internal proto upb dep
        "@org_tensorflow//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
        "@org_tensorflow//tensorflow/python:tf2",
        "@org_tensorflow//tensorflow/python/client:session",
        "@org_tensorflow//tensorflow/python/eager:def_function",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:importer",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:random_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
        "@org_tensorflow//tensorflow/python/platform:gfile",
        "@org_tensorflow//tensorflow/python/platform:resource_loader",
        "@org_tensorflow//tensorflow/python/saved_model",
        "@org_tensorflow//tensorflow/python/saved_model:save",
        "@org_tensorflow//tensorflow/python/trackable:autotrackable",
        "@org_tensorflow//tensorflow/python/training:training_util",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "lite",
    srcs = ["lite.py"],
    tags = [
        "ignore_for_dep=third_party.py.keras",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":conversion_metadata_schema_py",
        ":convert",
        ":convert_phase",
        ":convert_saved_model",
        ":interpreter",
        ":lite_constants",
        ":op_hint",
        ":util",
        "//tflite/experimental/microfrontend:audio_microfrontend_py",
        "//tflite/profiling/proto:model_runtime_info_py",
        "//tflite/profiling/proto:profiling_info_py",
        "//tflite/python/metrics",
        "//tflite/python/optimize:calibrator",
        "//tflite/tools:flatbuffer_utils",
        "//tflite/tools/optimize/debugging/python:debugger",
        "@absl_py//absl/logging",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow/python:representative_dataset",
        "@org_tensorflow//tensorflow/core:protos_all_py",
        "@org_tensorflow//tensorflow/python/client:session",
        "@org_tensorflow//tensorflow/python/eager:context",
        "@org_tensorflow//tensorflow/python/eager:def_function",
        "@org_tensorflow//tensorflow/python/eager:function",
        "@org_tensorflow//tensorflow/python/framework:byte_swap_tensor",
        "@org_tensorflow//tensorflow/python/framework:convert_to_constants",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:errors",
        "@org_tensorflow//tensorflow/python/framework:importer",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/framework:tensor_spec",
        "@org_tensorflow//tensorflow/python/framework:versions",
        "@org_tensorflow//tensorflow/python/platform:gfile",
        "@org_tensorflow//tensorflow/python/saved_model:load",
        "@org_tensorflow//tensorflow/python/saved_model:loader",
        "@org_tensorflow//tensorflow/python/saved_model:save",
        "@org_tensorflow//tensorflow/python/saved_model:save_options",
        "@org_tensorflow//tensorflow/python/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/python/saved_model:tag_constants",
        "@org_tensorflow//tensorflow/python/util:deprecation",
        "@org_tensorflow//tensorflow/python/util:keras_deps",
        "@org_tensorflow//tensorflow/python/util:nest",
        "@org_tensorflow//tensorflow/python/util:tf_export",
    ],
)

py_strict_test(
    name = "lite_test",
    srcs = ["lite_test.py"],
    data = [
        "//tflite/python/testdata:control_flow_v1.pbtxt",
        "@tflite_mobilenet_ssd_quant_protobuf//:tflite_graph.pb",
    ],
    shard_count = 4,
    tags = [
        "no_windows",
    ],
    deps = [
        ":conversion_metadata_schema_py",
        ":convert",
        ":interpreter",
        ":lite",
        ":lite_constants",
        ":schema_py",
        ":util",
        #internal proto upb dep
        "@org_tensorflow//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/client:session",
        "@org_tensorflow//tensorflow/python/eager:context",
        "@org_tensorflow//tensorflow/python/eager:def_function",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:convert_to_constants",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/framework:versions",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:logging_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops:nn_ops",
        "@org_tensorflow//tensorflow/python/ops:random_ops",
        "@org_tensorflow//tensorflow/python/ops:variable_scope",
        "@org_tensorflow//tensorflow/python/ops:variables",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
        "@org_tensorflow//tensorflow/python/platform:gfile",
        "@org_tensorflow//tensorflow/python/platform:resource_loader",
        "@org_tensorflow//tensorflow/python/saved_model",
        "@org_tensorflow//tensorflow/python/training:training_util",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "lite_v2_test",
    srcs = ["lite_v2_test.py"],
    data = [
        "//tflite/python/testdata:test_delegate.so",
        "//tflite/python/testdata/control_flow_v1_saved_model:saved_model.pb",
    ],
    shard_count = 18,
    tags = [
        "no_windows",
    ],
    deps = [
        ":conversion_metadata_schema_py",
        ":convert",
        ":interpreter",
        ":lite",
        ":lite_v2_test_util",
        ":schema_py",
        ":test_util",
        ":util",
        #internal proto upb dep
        "@org_tensorflow//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/stablehlo:quantization_options_proto_py",
        "//tflite/python/testdata:_pywrap_test_registerer",
        "//tflite/python/testdata:double_op",
        "//tflite/tools:flatbuffer_utils",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/framework:versions",
        "@org_tensorflow//tensorflow/python/lib/io:file_io",
        "@org_tensorflow//tensorflow/python/ops:map_ops",
        "@org_tensorflow//tensorflow/python/ops:rnn",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
        "@org_tensorflow//tensorflow/python/platform:resource_loader",
        "@org_tensorflow//tensorflow/python/saved_model",
        "@org_tensorflow//tensorflow/python/saved_model:loader",
        "@org_tensorflow//tensorflow/python/saved_model:save",
        "@org_tensorflow//tensorflow/python/saved_model:save_options",
        "@org_tensorflow//tensorflow/python/trackable:autotrackable",
        "@absl_py//absl/testing:parameterized",
        "@pypi_jax//:pkg",
    ],
)

py_strict_library(
    name = "lite_v2_test_util",
    testonly = 1,
    srcs = ["lite_v2_test_util.py"],
    tags = [
        "no_windows",
    ],
    deps = [
        ":interpreter",
        ":lite",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py_no_contrib",
        "@org_tensorflow//tensorflow/python/eager:def_function",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:tensor_spec",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops:variables",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
        "@org_tensorflow//tensorflow/python/trackable:autotrackable",
        "@org_tensorflow//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "lite_flex_test",
    srcs = ["lite_flex_test.py"],
    deps = [
        ":convert",
        ":interpreter",
        ":lite",
        ":test_util",
        #internal proto upb dep
        "@org_tensorflow//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
        "//tflite/python/testdata:double_op",
        "@org_tensorflow//tensorflow/python/client:session",
        "@org_tensorflow//tensorflow/python/eager:def_function",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:importer",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:list_ops",
        "@org_tensorflow//tensorflow/python/ops:nn_ops",
        "@org_tensorflow//tensorflow/python/ops:variables",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
        "@org_tensorflow//tensorflow/python/saved_model",
        "@org_tensorflow//tensorflow/python/trackable:autotrackable",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "util",
    srcs = ["util.py"],
    visibility = internal_visibility_allowlist(),
    deps = [
        ":conversion_metadata_schema_py",
        ":op_hint",
        ":schema_py",
        ":schema_util",
        ":tflite_keras_util",
        "//tflite/tools:flatbuffer_utils",
        "@absl_py//absl/logging",
        "@flatbuffers//:runtime_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
        "@org_tensorflow//tensorflow/python/eager:function",
        "@org_tensorflow//tensorflow/python/framework:convert_to_constants",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:error_interpolation",
        "@org_tensorflow//tensorflow/python/grappler:tf_optimizer",
        "@org_tensorflow//tensorflow/python/training:saver",
        "@org_tensorflow//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "util_test",
    srcs = ["util_test.py"],
    tags = [
        "no_windows",
    ],
    deps = [
        ":util",
        #internal proto upb dep
        "@org_tensorflow//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tflite/python:lite",
        "//tflite/tools:flatbuffer_utils",
        "@org_tensorflow//tensorflow/python/client:session",
        "@org_tensorflow//tensorflow/python/framework:convert_to_constants",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops:while_loop",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "tflite_keras_util",
    srcs = [
        "tflite_keras_util.py",
    ],
    deps = [
        "@org_tensorflow//tensorflow/python/eager:def_function",
        "@org_tensorflow//tensorflow/python/framework:tensor_spec",
        "@org_tensorflow//tensorflow/python/util:compat",
        "@org_tensorflow//tensorflow/python/util:keras_deps",
        "@org_tensorflow//tensorflow/python/util:nest",
    ],
)

py_strict_library(
    name = "lite_constants",
    srcs = ["lite_constants.py"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/mlir/lite:converter_flags_proto_py",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/util:all_util",
        "@org_tensorflow//tensorflow/python/util:tf_export",
    ],
)

pytype_strict_library(
    name = "convert",
    srcs = ["convert.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":convert_phase",
        ":lite_constants",
        ":util",
        "//tflite/python/metrics:metrics_wrapper",
        "//tflite/tools:flatbuffer_utils",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:converter_flags_proto_py",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:model_flags_proto_py",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:types_proto_py",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/metrics:converter_error_data_proto_py",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/python:wrap_converter",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/stablehlo:quantization_options_proto_py",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:tensor_shape",
        "@org_tensorflow//tensorflow/python/util:deprecation",
        "@org_tensorflow//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "op_hint",
    srcs = ["op_hint.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all_py",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:graph_util",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/framework:tensor_util",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/util:all_util",
        "@org_tensorflow//tensorflow/python/util:compat",
        "@org_tensorflow//tensorflow/python/util:deprecation",
        "@org_tensorflow//tensorflow/python/util:tf_export",
    ],
)

pytype_strict_contrib_test(
    name = "convert_test",
    srcs = ["convert_test.py"],
    deps = [
        ":convert",
        ":interpreter",
        ":op_hint",
        "//tflite/python/metrics:metrics_wrapper",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:converter_flags_proto_py",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/metrics:converter_error_data_proto_py",
        "@org_tensorflow//tensorflow/python/client:session",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:graph_util",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops_stack",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
        "@org_tensorflow//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "convert_saved_model",
    srcs = ["convert_saved_model.py"],
    visibility = [
        "//tflite:__subpackages__",
    ],
    deps = [
        ":convert_phase",
        ":util",
        "@org_tensorflow//tensorflow/python/client:session",
        "@org_tensorflow//tensorflow/python/framework:graph_util",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/platform:tf_logging",
        "@org_tensorflow//tensorflow/python/saved_model",
        "@org_tensorflow//tensorflow/python/saved_model:constants",
        "@org_tensorflow//tensorflow/python/saved_model:loader",
    ],
)

py_strict_test(
    name = "convert_saved_model_test",
    srcs = ["convert_saved_model_test.py"],
    tags = [
        "no_windows",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":convert_saved_model",
        #internal proto upb dep
        "@org_tensorflow//tensorflow/python/client:session",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/framework:tensor_shape",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/layers",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops/losses",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
        "@org_tensorflow//tensorflow/python/platform:test",
        "@org_tensorflow//tensorflow/python/saved_model",
        "@org_tensorflow//tensorflow/python/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/python/saved_model:tag_constants",
    ],
)

py_strict_binary(
    name = "convert_file_to_c_source",
    srcs = ["convert_file_to_c_source.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":util",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

sh_test(
    name = "convert_file_to_c_source_test",
    srcs = ["convert_file_to_c_source_test.sh"],
    data = [":convert_file_to_c_source"],
)

py_strict_library(
    name = "schema_util",
    srcs = ["schema_util.py"],
    visibility = ["//tflite/schema:utils_friends"],
    deps = [
        "@org_tensorflow//tensorflow/python/util:all_util",
    ],
)

# Use py_library since the metrics module is imported in a try-except block,
# which doesn't work with the pytype_strict_library.
py_strict_library(
    name = "convert_phase",
    srcs = ["convert_phase.py"],
    visibility = ["//tflite:__subpackages__"],
    deps = [
        "//tflite/python/metrics",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/metrics:converter_error_data_proto_py",
    ],
)

py_strict_library(
    name = "analyzer",
    srcs = [
        "analyzer.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tflite/python/analyzer_wrapper:_pywrap_analyzer_wrapper",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/python:wrap_converter",
        "@org_tensorflow//tensorflow/python/util:tf_export",
    ],
)

py_strict_test(
    name = "analyzer_test",
    srcs = ["analyzer_test.py"],
    data = [
        "//tflite:testdata/add.bin",
        "//tflite:testdata/conv_huge_im2col.bin",
        "//tflite:testdata/multi_add_flex.bin",
    ],
    deps = [
        ":analyzer",
        #internal proto upb dep
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tflite/python:lite",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
        "@org_tensorflow//tensorflow/python/platform:resource_loader",
        "@org_tensorflow//tensorflow/python/trackable:autotrackable",
    ],
)
