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
load("//litert/python:pytype.bzl", "pytype_strict_library")
load("//litert/python:strict.bzl", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "flatbuffer_utils",
    srcs = ["flatbuffer_utils.py"],
    deps = [
        ":mmap_utils",
        # Dependency placeholder: gfile,
        # copybara:uncomment "//third_party/py/flatbuffers",
        "@org_tensorflow//third_party/py/numpy:headers",
        "//tflite/python:schema_py",
    ],
)

py_strict_test(
    name = "flatbuffer_utils_test",
    srcs = ["flatbuffer_utils_test.py"],
    tags = ["no_oss"],
    deps = [
        ":flatbuffer_utils",
        "@absl_py//absl/testing:absltest",
        # copybara:uncomment "//third_party/py/flatbuffers",
        "//tflite/python:schema_py",
    ],
)

pytype_strict_library(
    name = "mmap_utils",
    srcs = ["mmap_utils.py"],
    deps = [
        # Dependency placeholder: gfile,
    ],
)

py_strict_test(
    name = "mmap_utils_test",
    srcs = ["mmap_utils_test.py"],
    target_compatible_with = select({
        "//litert:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    deps = [
        ":mmap_utils",
        # Dependency placeholder: gfile,
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_library(
    name = "benchmark_litert_model",
    srcs = ["benchmark_litert_model.py"],
    imports = ["."],
    deps = [
        # Dependency placeholder: gfile,
        # copybara:uncomment "@org_tensorflow//third_party/py/numpy:headers",
        "@org_tensorflow//third_party/py/numpy",  # copybara:comment
    ],
)

py_strict_test(
    name = "benchmark_litert_model_test",
    srcs = ["benchmark_litert_model_test.py"],
    deps = [
        ":benchmark_litert_model",
        "//litert/python/litert_wrapper/compiled_model_wrapper:options",
        # copybara:uncomment "@org_tensorflow//third_party/py/numpy:headers",
        "@org_tensorflow//third_party/py/numpy",  # copybara:comment
    ],
)
