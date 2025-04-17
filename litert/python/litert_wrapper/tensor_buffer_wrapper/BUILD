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

load("//devtools/python/blaze:strict.bzl", "py_strict_library", "py_strict_test")
load("@org_tensorflow//tensorflow:tensorflow.google.bzl", "pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "tensor_buffer_wrapper_lib",
    srcs = ["tensor_buffer_wrapper.cc"],
    hdrs = [
        "tensor_buffer_wrapper.h",
    ],
    deps = [
        "@com_google_absl//absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/c:litert_tensor_buffer",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_handle",
        "//litert/cc:litert_model",
        "//litert/cc:litert_tensor_buffer",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "//tflite:framework",
        "//tflite/core:framework",
    ],
)

pybind_extension(
    name = "_pywrap_litert_tensor_buffer_wrapper",
    srcs = [
        "tensor_buffer_wrapper_pybind11.cc",
    ],
    hdrs = ["tensor_buffer_wrapper.h"],
    enable_stub_generation = True,
    link_in_framework = True,
    pytype_srcs = [
        "_pywrap_litert_tensor_buffer_wrapper.pyi",
    ],
    deps = [
        ":tensor_buffer_wrapper_lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_model",
        "//litert/cc:litert_tensor_buffer",
        "//third_party/pybind11",
        "//third_party/python_runtime:headers",
        "//tflite:framework",
    ],
)

py_strict_library(
    name = "tensor_buffer",
    srcs = [
        "tensor_buffer.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//litert/python/litert_wrapper/tensor_buffer_wrapper:_pywrap_litert_tensor_buffer_wrapper",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tensor_buffer_test",
    srcs = ["tensor_buffer_test.py"],
    # Static linking is required because this loads a cc_binary as a shared
    # library, which would otherwise create ODR violations.
    # copybara:uncomment linking_mode = "static",
    deps = [
        ":tensor_buffer",
        "//third_party/py/numpy",
    ],
)
