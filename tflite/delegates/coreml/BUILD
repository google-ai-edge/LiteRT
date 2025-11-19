# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load("//tflite:build_def.bzl", "CXX17_BAZEL_ONLY_COPTS")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

exports_files(["coreml_delegate.h"])

objc_library(
    name = "coreml_executor",
    srcs = ["coreml_executor.mm"],
    hdrs = ["coreml_executor.h"],
    copts = CXX17_BAZEL_ONLY_COPTS,
    sdk_frameworks = [
        "CoreML",
        "Foundation",
    ],
    deps = ["@coremltools//:mlmodel_cc_proto"],
)

cc_library(
    name = "mlmodel_proto_cc",
    deps = [
        "@coremltools//:mlmodel_cc_proto",
    ],
)

objc_library(
    name = "coreml_delegate",
    srcs = ["coreml_delegate.mm"],
    hdrs = ["coreml_delegate.h"],
    copts = CXX17_BAZEL_ONLY_COPTS,
    defines = select({
        "//tflite/delegates:tflite_debug_delegate": ["TFLITE_DEBUG_DELEGATE"],
        "//conditions:default": [],
    }),
    module_name = "TensorFlowLiteCCoreML",
    deps = [
        ":coreml_delegate_kernel",
        ":mlmodel_proto_cc",
        "//tflite:kernel_api",
        "//tflite:minimal_logging",
        "//tflite/core/c:common",
        "//tflite/delegates:utils",
        "//tflite/delegates/coreml/builders:op_builder",
        "//tflite/delegates/coreml/builders:op_validator",
        "//tflite/delegates/coreml/builders:util",
        "//tflite/kernels:kernel_util",
    ],
    alwayslink = 1,
)

objc_library(
    name = "coreml_delegate_kernel",
    srcs = [
        "coreml_delegate_kernel.mm",
    ],
    hdrs = [
        "coreml_delegate_kernel.h",
    ],
    copts = CXX17_BAZEL_ONLY_COPTS,
    deps = [
        ":coreml_executor",
        ":mlmodel_proto_cc",
        "//tflite:kernel_api",
        "//tflite/core/c:common",
        "//tflite/delegates/coreml/builders:op_builder",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:optimized_base",
        "//tflite/kernels/internal:types",
        "@FP16",
    ],
)
