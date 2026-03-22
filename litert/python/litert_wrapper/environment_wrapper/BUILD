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

load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")

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
    name = "environment_wrapper_lib",
    srcs = ["environment_wrapper.cc"],
    hdrs = ["environment_wrapper.h"],
    deps = [
        "//litert/cc:litert_environment_options",
        "//litert/cc/dynamic_runtime:litert_environment",
        "//litert/python/litert_wrapper/common:litert_wrapper_utils",
        "@com_google_absl//absl/types:span",
        "@xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ] + select({
        "@platforms//os:windows": [
            "//litert/c:litert_runtime_c_api_import",
        ],
        "//conditions:default": [],
    }),
)

pybind_extension(
    name = "_pywrap_litert_environment_wrapper",
    srcs = ["environment_wrapper_pybind11.cc"],
    hdrs = ["environment_wrapper.h"],
    common_lib_packages = ["litert/python"],
    copts = WINDOWS_PYBIND_UNDEFS,
    link_in_framework = True,
    linkopts = select({
        "@platforms//os:ios": ["-framework Metal"],
        "@platforms//os:macos": ["-framework Metal"],
        "//conditions:default": [],
    }),
    pytype_srcs = [
        "_pywrap_litert_environment_wrapper.pyi",
    ],
    wrap_py_init = True,
    deps = [
        ":environment_wrapper_lib",
        "@pybind11",
        "@xla//third_party/python_runtime:headers",
    ],
)

py_library(
    name = "environment",
    srcs = ["environment.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_litert_environment_wrapper",
    ],
)
