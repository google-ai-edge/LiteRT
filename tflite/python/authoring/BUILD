# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

load("@org_tensorflow//tensorflow:tensorflow.bzl", "py_test")
load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "authoring",
    srcs = [
        "authoring.py",
    ],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tflite/converter/metrics:converter_error_data_proto_py",
        "//tflite/python:convert",
        "//tflite/python:lite",
        "@org_tensorflow//tensorflow/python/util:tf_export",
    ],
)

py_test(
    name = "authoring_test",
    srcs = ["authoring_test.py"],
    strict_deps = True,
    deps = [
        ":authoring",
        #internal proto upb dep
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tflite/python:lite",
    ],
)
