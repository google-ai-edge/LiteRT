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

load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_library")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pywrap_binaries", "pywrap_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_python_users"],
)

py_strict_library(
    name = "litert_pywrap_deps",
    visibility = ["//visibility:public"],
    deps = [
        "//litert/python/litert_wrapper/compiled_model_wrapper:_pywrap_litert_compiled_model_wrapper",
        "//litert/python/litert_wrapper/tensor_buffer_wrapper:_pywrap_litert_tensor_buffer_wrapper",
    ],
)

pywrap_library(
    name = "pywrap_litert",
    common_lib_def_files_or_filters = {
        "litert/python/pywrap_litert_common": "//tflite/python:pywrap_tflite_common.json",
    },
    common_lib_linkopts = {
        "litert/python/pywrap_litert_common": select({
            "@bazel_tools//src/conditions:windows": [],
            "@bazel_tools//src/conditions:darwin": ["-Wl,-rpath,@loader_path/"],
            "//conditions:default": ["-Wl,-rpath,$$ORIGIN/"],
        }),
    },
    common_lib_version_scripts = {
        "litert/python/pywrap_litert_common": select({
            "@bazel_tools//src/conditions:windows": None,
            "@bazel_tools//src/conditions:darwin": "//tflite/python:pywrap_tflite_common_darwin.lds",
            "//conditions:default": "//tflite/python:pywrap_tflite_common.lds",
        }),
    },
    pywrap_count = 10,
    visibility = ["//visibility:public"],
    deps = [
        ":litert_pywrap_deps",
        "//tflite/python:tflite_pywrap_deps",
    ],
)

pywrap_binaries(
    name = "pywrap_litert_binaries",
    dep = ":pywrap_litert",
    visibility = ["//visibility:public"],
)
