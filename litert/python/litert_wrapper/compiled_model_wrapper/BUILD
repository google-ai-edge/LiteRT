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

# copybara:uncomment load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:py_library.bzl", "py_library")

# PROTOBUF_USE_DLLS and ABSL_CONSUME_DLL tell protobuf/absl headers to assume DLL linkage and decorate symbols with __declspec(dllimport).
# In our build, the pybind module is not consuming prebuilt protobuf/absl DLLs; it’s linking against the static libs
# If those macros are left on, MSVC treats symbols as imported from a DLL that doesn’t exist, which leads to odr violations.
WINDOWS_PYBIND_UNDEFS = select({
    "@platforms//os:windows": [
        "/UPROTOBUF_USE_DLLS",
        "/UABSL_CONSUME_DLL",
    ],
    "//conditions:default": [],
})

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
        "//litert/cc:litert_environment_options",
        "//litert/python/litert_wrapper/common:litert_wrapper_utils",
        "//tflite:framework",
        "//tflite/core:framework",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "@xla//third_party/python_runtime:headers",
    ] + select({
        "@platforms//os:windows": [
            "//litert/c:litert_runtime_c_api_import",
        ],
        "//conditions:default": [],
    }),
)

pybind_extension(
    name = "_pywrap_litert_compiled_model_wrapper",
    srcs = [
        "compiled_model_wrapper_pybind11.cc",
    ],
    hdrs = ["compiled_model_wrapper.h"],
    common_lib_packages = ["litert/python"],
    copts = WINDOWS_PYBIND_UNDEFS,
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
        "@pybind11",
        "@xla//third_party/python_runtime:headers",
    ],
)

py_library(
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
