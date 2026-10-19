# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# `litert_model`: a Python LiteRT IR backed by the C++ `LiteRtModelT`.

load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_python//python:py_test.bzl", "py_test")
load("//litert/python:pytype.bzl", "pytype_strict_library")

WINDOWS_PYBIND_UNDEFS = select({
    "//litert:windows": [
        "/UPROTOBUF_USE_DLLS",
        "/UABSL_CONSUME_DLL",
    ],
    "//conditions:default": [],
})

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_python_users"],
)

TESTDATA_MODELS = [
    "//litert/test:testdata/add_simple.tflite",
    "//litert/test:testdata/blockwise_quantized.tflite",
    "//litert/test:testdata/constant_output_tensor.tflite",
    "//litert/test:testdata/dynamic_add_model.tflite",
    "//litert/test:testdata/dynamic_string_custom_ops.tflite",
    "//litert/test:testdata/island_partial.tflite",
    "//litert/test:testdata/multi_subgraph.tflite",
    "//litert/test:testdata/reverse_signature_model.tflite",
    "//litert/test:testdata/simple_concatenation_op.tflite",
    "//litert/test:testdata/simple_conv_2d_op.tflite",
    "//litert/test:testdata/simple_model.tflite",
    "//litert/test:testdata/simple_quantized_ops.tflite",
    "//litert/test:testdata/simple_shlo_composite.tflite",
    "//litert/test:testdata/static_w8_a16_quantized_k_einsum.tflite",
]

cc_library(
    name = "litert_model_lib",
    srcs = ["litert_model_lib.cc"],
    hdrs = ["litert_model_lib.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_layout",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_consts",
        "//litert/core:build_stamp",
        "//litert/core:dispatch_op_schema",
        "//litert/core/model",
        "//litert/core/model:graph_validation",
        "//litert/core/util:flatbuffer_tools",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "litert_model_lib_test",
    srcs = ["litert_model_lib_test.cc"],
    data = TESTDATA_MODELS,
    deps = [
        ":litert_model_lib",
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc/internal:litert_consts",
        "//litert/core:build_stamp",
        "//litert/core/model",
        "//litert/core/model:model_load",
        "//litert/core/model:model_serialize",
        "//litert/core/util:flatbuffer_tools",
        "//litert/test:common",
        "//litert/test:matchers",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)

pybind_extension(
    name = "_pywrap_litert_model",
    srcs = ["_pywrap_litert_model.cc"],
    hdrs = ["pywrap_handles.h"],
    common_lib_packages = ["litert/python"],
    copts = WINDOWS_PYBIND_UNDEFS,
    pytype_srcs = ["_pywrap_litert_model.pyi"],
    wrap_py_init = True,
    deps = [
        ":litert_model_lib",
        "//litert/c:litert_common",
        "//litert/c:litert_layout",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc/internal:litert_consts",
        "//litert/core:build_stamp",
        "//litert/core/model",
        "//litert/core/model:model_load",
        "//litert/core/model:model_serialize",
        "//litert/core/model:shape_inference",
        "//litert/core/model:shape_inference_types",
        "//litert/core/util:flatbuffer_tools",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ],
)

pytype_strict_library(
    name = "litert_model",
    srcs = ["litert_model.py"],
    deps = [
        ":_pywrap_litert_model",
        # copybara:uncomment "//third_party/py/flatbuffers",
        "//third_party/py/numpy",
        "//tflite/python:schema_py",
    ],
)

py_test(
    name = "litert_model_test",
    srcs = ["litert_model_test.py"],
    data = TESTDATA_MODELS + [
        "//litert/c:litert_model_types.h",
        "//litert/c:litert_op_code.h",
    ],
    deps = [
        ":_pywrap_litert_model",
        ":litert_model",
        "//litert/python/aot/core:common",
        "//tflite/python:schema_py",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "litert_model_dispatch_test",
    srcs = ["litert_model_dispatch_test.py"],
    data = TESTDATA_MODELS,
    deps = [
        ":litert_model",
        "//litert/python/aot/core:common",
        "//tflite/python:schema_py",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "litert_model_numerics_test",
    srcs = ["litert_model_numerics_test.py"],
    tags = ["no_oss"],
    deps = [
        ":litert_model",
        "//litert/python/litert_wrapper/compiled_model_wrapper:compiled_model",
        "//tflite/python:schema_py",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:absltest",
    ],
)
