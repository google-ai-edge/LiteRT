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
load("@org_tensorflow//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "flatbuffer_utils",
    srcs = ["flatbuffer_utils.py"],
    deps = [
        # Dependency placeholder: gfile,
        # copybara:uncomment "//third_party/py/flatbuffers",
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
