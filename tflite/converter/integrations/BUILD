# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load("@org_tensorflow//tensorflow:tensorflow.bzl", "py_test")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//litert/python/tools/model_utils:__subpackages__",
        "//tflite/converter/integrations:__subpackages__",
    ],
    licenses = ["notice"],
)

pybind_extension(
    name = "model_utils_core_pybind",
    srcs = [
        "model_utils_core_pybind.cc",
    ],
    deps = [
        "//tflite/converter:flatbuffer_export",
        "//tflite/converter:flatbuffer_import",
        "//tflite/converter:tensorflow_lite",
        "//tflite/converter/quantization/ir:QuantOps",
        "//tflite/converter/stablehlo:prepare_hlo",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:CAPIIRHeaders",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MLIRBindingsPythonNanobindHeadersAndDeps",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@local_xla//third_party/python_runtime:headers",
        "@nanobind",
        "@org_tensorflow//tensorflow/compiler/mlir:init_mlir",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "@org_tensorflow//tensorflow/python/lib/core:ndarray_tensor",
        "@org_tensorflow//tensorflow/python/lib/core:py_func_lib",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:vhlo_ops",
    ],
)

py_test(
    name = "py_bindings_test",
    srcs = ["py_bindings_test.py"],
    deps = [
        "//tflite/converter/integrations/python/mlir",
    ],
)
