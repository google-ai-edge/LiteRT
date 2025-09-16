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

load(
    "@org_tensorflow//tensorflow:strict.default.bzl",
    "py_strict_library",
    # copybara:uncomment "py_strict_test",
)
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "compiled_model_wrapper_lib",
    srcs = ["compiled_model_wrapper.cc"],
    hdrs = [
        "compiled_model_wrapper.h",
    ],
    deps = [
        "//litert/cc:litert_api_with_dynamic_runtime",
        "//litert/python/litert_wrapper/common:litert_wrapper_utils",
        "//tflite:framework",
        "//tflite/core:framework",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "@local_xla//third_party/python_runtime:headers",
    ],
)

pybind_extension(
    name = "_pywrap_litert_compiled_model_wrapper",
    srcs = [
        "compiled_model_wrapper_pybind11.cc",
    ],
    hdrs = ["compiled_model_wrapper.h"],
    common_lib_packages = ["litert/python"],
    link_in_framework = True,
    pytype_srcs = [
        "_pywrap_litert_compiled_model_wrapper.pyi",
    ],
    wrap_py_init = True,
    deps = [
        ":compiled_model_wrapper_lib",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "//tflite:framework",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

py_strict_library(
    name = "compiled_model",
    srcs = [
        "compiled_model.py",
        "hardware_accelerator.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_litert_compiled_model_wrapper",
        "//litert/python/litert_wrapper/tensor_buffer_wrapper:tensor_buffer",
    ],
)

# copybara:uncomment_begin(google-only)
# py_strict_test(
#     name = "compiled_model_unit_test",
#     srcs = ["compiled_model_unit_test.py"],
#     # Static linking is required because this loads a cc_binary as a shared
#     # library, which would otherwise create ODR violations.
#     linking_mode = "static",
#     tags = [
#         "no_oss",  # TODO(b/190842754): Enable test in OSS.
#     ],
#     deps = [
#         ":_pywrap_litert_compiled_model_wrapper",
#         ":compiled_model",
#         "@com_google_protobuf//:protobuf_python:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
#     ],
# )
# copybara:uncomment_end
