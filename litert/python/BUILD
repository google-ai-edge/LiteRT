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

load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pybind_extension", "pywrap_binaries", "pywrap_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

WINDOWS_PYBIND_UNDEFS = select({
    "@platforms//os:windows": [
        "/UPROTOBUF_USE_DLLS",
        "/UABSL_CONSUME_DLL",
    ],
    "//conditions:default": [],
})

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml/litert:license"],
    default_visibility = ["//litert:litert_python_users"],
)

cc_library(
    name = "converter_api",
    srcs = ["converter.cc"],
    hdrs = ["converter.h"],
    features = ["-parse_headers"],
    deps = [
        "//tflite/converter:converter_flags_proto_cc",
        "//tflite/converter:model_flags_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@xla//third_party/python_runtime:headers",
    ],
)

pybind_extension(
    name = "_pywrap_litert_converter",
    srcs = [
        "converter_wrapper.cc",
    ],
    hdrs = ["converter.h"],
    common_lib_packages = ["litert/python/pywrap_litert_common"],
    copts = WINDOWS_PYBIND_UNDEFS,
    wrap_py_init = True,
    deps = [
        ":converter_api",
        "//tflite/converter:converter_flags_proto_cc",
        "//tflite/converter:model_flags_proto_cc",
        "@com_google_absl//absl/status:statusor",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_abseil//pybind11_abseil:statusor_caster",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

py_strict_library(
    name = "litert_pywrap_deps",
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_litert_converter",
        "//ci/tools/python/wheel:tflite_converter_protos",
        "//litert/python/litert_wrapper/compiled_model_wrapper:_pywrap_litert_compiled_model_wrapper",
        "//litert/python/litert_wrapper/tensor_buffer_wrapper:_pywrap_litert_tensor_buffer_wrapper",
    ],
)

PYWRAP_LITERT_COMMON_LIB_DEF_FILES_OR_FILTERS = "//tflite/python:pywrap_tflite_common.json"

PYWRAP_LITERT_COMMON_LIB_LINKOPTS = select({
    "@bazel_tools//src/conditions:windows": [],
    "@bazel_tools//src/conditions:darwin": [],
    "//conditions:default": ["-Wl,-rpath,$$ORIGIN/"],
})

PYWRAP_LITERT_COMMON_LIB_VERSION_SCRIPTS = select({
    "@bazel_tools//src/conditions:windows": None,
    "@bazel_tools//src/conditions:darwin": "//tflite/python:pywrap_tflite_common_darwin.lds",
    "//conditions:default": "//tflite/python:pywrap_tflite_common.lds",
})

pywrap_library(
    name = "pywrap_litert",
    common_lib_def_files_or_filters = {
        "litert/python/pywrap_litert_common": PYWRAP_LITERT_COMMON_LIB_DEF_FILES_OR_FILTERS,
    },
    common_lib_linkopts = {
        "litert/python/pywrap_litert_common": PYWRAP_LITERT_COMMON_LIB_LINKOPTS,
    },
    common_lib_version_scripts = {
        "litert/python/pywrap_litert_common": PYWRAP_LITERT_COMMON_LIB_VERSION_SCRIPTS,
    },
    pywrap_count = 11,
    visibility = ["//visibility:public"],
    deps = [
        ":litert_pywrap_deps",
        "//tflite/python:tflite_pywrap_deps",
    ],
)

pywrap_library(
    name = "pywrap_litert_with_converter",
    common_lib_def_files_or_filters = {
        "litert/python/pywrap_litert_with_converter_common": PYWRAP_LITERT_COMMON_LIB_DEF_FILES_OR_FILTERS,
    },
    common_lib_linkopts = {
        "litert/python/pywrap_litert_with_converter_common": PYWRAP_LITERT_COMMON_LIB_LINKOPTS,
    },
    common_lib_version_scripts = {
        "litert/python/pywrap_litert_with_converter_common": PYWRAP_LITERT_COMMON_LIB_VERSION_SCRIPTS,
    },
    pywrap_count = 15,
    visibility = ["//visibility:public"],
    deps = [
        ":litert_pywrap_deps",
        "//litert/python/mlir",
        "//tflite/python:tflite_pywrap_deps",
    ],
)

pywrap_binaries(
    name = "pywrap_litert_binaries",
    dep = ":pywrap_litert",
    visibility = ["//visibility:public"],
)

pywrap_binaries(
    name = "pywrap_litert_with_converter_binaries",
    dep = ":pywrap_litert_with_converter",
    visibility = ["//visibility:public"],
)

py_strict_test(
    name = "converter_wrapper_test",
    srcs = ["converter_wrapper_test.py"],
    deps = [
        ":pywrap_litert",
        "//ci/tools/python/wheel:tflite_converter_protos",
        "@absl_py//absl/testing:absltest",
    ],
)
