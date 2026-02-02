# Copyright 2025 Google LLC.
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

load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pybind_extension")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:py_library.bzl", "py_library")
load("//litert/build_common:litert_build_defs.bzl", "if_google")
load("//litert/build_common:symlink_files.bzl", "symlink_inputs")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

PYBIND_COPTS = [
    "-fexceptions",
    "-frtti",
]

symlink_inputs(
    name = "_mlir_libs",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        ".": [
            "@llvm-project//mlir/python:MlirLibsPyFiles",
        ],
    }},
    deps = [
        ":_mlir",
        ":_stablehlo",
        ":converter_api_ext",
        ":model_utils_ext",
    ],
)

cc_library(
    name = "lib_litert_compiler_mlir",
    srcs = select({
        "@platforms//os:macos": [],
        "@platforms//os:windows": [],
        "//conditions:default": [":libLiteRTCompilerMLIR.so"],
    }),
)

cc_binary(
    name = "libLiteRTCompilerMLIR.so",
    linkshared = 1,
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [
        "//litert/compiler/mlir:converter_api_core",
        "//litert/compiler/mlir:model_utils_core",
        "@llvm-project//mlir:CAPIArithObjects",
        "@llvm-project//mlir:CAPIIRObjects",
        "@llvm-project//mlir:CAPITransformsObjects",
        "@llvm-project//mlir:MLIRBindingsPythonCAPIObjects",
        "@nanobind",
        "@stablehlo//:stablehlo_unified_capi_objects",
    ],
)

pybind_extension(
    name = "_mlir",
    srcs = [
        "@llvm-project//mlir:MLIRBindingsPythonSourceFiles",
        "@llvm-project//mlir:lib/Bindings/Python/MainModule.cpp",
    ],
    hdrs = [
        "@llvm-project//mlir:MLIRBindingsPythonCoreHeaders",
    ],
    common_lib_packages = ["litert/python"],
    copts = PYBIND_COPTS,
    wrap_py_init = False,
    deps = [
        ":lib_litert_compiler_mlir",
        "@llvm-project//mlir:MLIRBindingsPythonCoreNoCAPI",
        "@llvm-project//mlir:MLIRBindingsPythonHeaders",
    ],
)

pybind_extension(
    name = "model_utils_ext",
    srcs = [
        "model_utils_ext.cc",
    ],
    hdrs = [
        "//litert/compiler/mlir:pybind_ext_core_headers",
        "@llvm-project//mlir:MLIRBindingsPythonCoreHeaders",
    ],
    common_lib_packages = ["litert/python"],
    copts = PYBIND_COPTS,
    wrap_py_init = False,
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:MLIRBindingsPythonHeaders",
        "@nanobind",
        "@xla//third_party/python_runtime:headers",
    ] + if_google(
        [
            # google only: link everything statically here.
            "//litert/compiler/mlir:model_utils_core",
            "@llvm-project//llvm:Support",
            "@llvm-project//mlir:IR",
            "@llvm-project//mlir:CAPIIRHeaders",
            "@llvm-project//mlir:CAPITransformsHeaders",
            "@llvm-project//mlir:Pass",
        ],
        [
            # oss only: link to the shared library.
            ":lib_litert_compiler_mlir",
            "@llvm-project//mlir:MLIRBindingsPythonCoreNoCAPI",
        ],
    ),
)

pybind_extension(
    name = "converter_api_ext",
    srcs = [
        "converter_api_ext.cc",
    ],
    hdrs = [
        "//litert/compiler/mlir:pybind_ext_core_headers",
        "@llvm-project//mlir:MLIRBindingsPythonCoreHeaders",
    ],
    common_lib_packages = ["litert/python"],
    copts = PYBIND_COPTS,
    wrap_py_init = False,
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:MLIRBindingsPythonHeaders",
        "@nanobind",
        "@xla//third_party/python_runtime:headers",
    ] + if_google(
        [
            # google only: link everything statically here.
            "//litert/compiler/mlir:converter_api_core",
            "@llvm-project//llvm:Support",
            "@llvm-project//mlir:IR",
            "@llvm-project//mlir:CAPIIRHeaders",
            "@llvm-project//mlir:CAPITransformsHeaders",
            "@llvm-project//mlir:Pass",
        ],
        [
            # oss only: link to the shared library.
            ":lib_litert_compiler_mlir",
            "@llvm-project//mlir:MLIRBindingsPythonCoreNoCAPI",
        ],
    ),
)

# External only target.
pybind_extension(
    name = "_stablehlo",
    srcs = [
        "@stablehlo//:stablehlo_py_api_files",
    ],
    common_lib_packages = ["litert/python"],
    copts = PYBIND_COPTS,
    wrap_py_init = False,
    deps = [
        "@llvm-project//mlir:MLIRBindingsPythonHeaders",
        "@nanobind",
        "@rules_python//python/cc:current_py_cc_headers",
        "@xla//third_party/python_runtime:headers",
    ] + if_google(
        [
            # google only: link everything statically here.
            "@stablehlo//:stablehlo_capi",
            "@llvm-project//llvm:Support",
            "@llvm-project//mlir:IR",
            "@llvm-project//mlir:CAPIIRHeaders",
            "@llvm-project//mlir:CAPITransformsHeaders",
        ],
        [
            # oss only: link to the shared library.
            ":lib_litert_compiler_mlir",
            "@llvm-project//mlir:MLIRBindingsPythonCoreNoCAPI",
            "@stablehlo//:stablehlo_capi_headers",
        ],
    ),
)
