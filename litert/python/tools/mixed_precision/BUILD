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

load(
    "//litert/python:pytype.bzl",
    "pytype_strict_binary",
    "pytype_strict_contrib_test",
    "pytype_strict_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

pytype_strict_library(
    name = "mixed_precision",
    srcs = ["mixed_precision.py"],
    deps = [
        "//litert/python/tools/model_utils",
        "//third_party/py/mlir:ir",
        "@org_tensorflow//third_party/py/numpy:headers",
        "@xdsl",
    ],
)

pytype_strict_binary(
    name = "mixed_precision_main",
    srcs = ["mixed_precision_main.py"],
    deps = [
        ":mixed_precision",
        # Dependency placeholder: gfile,
        "//litert/python/tools/model_utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

pytype_strict_contrib_test(
    name = "mixed_precision_test",
    srcs = ["mixed_precision_test.py"],
    deps = [
        ":mixed_precision",
        "//litert/python/tools/model_utils",
        "//litert/python/tools/model_utils:testing",
        "//third_party/py/mlir:ir",
        "@absl_py//absl/testing:absltest",
        "@xdsl",
    ],
)

pytype_strict_library(
    name = "detect_tflite_model_overflow_lib",
    srcs = ["detect_tflite_model_overflow_lib.py"],
    deps = [
        "//third_party/py/tensorflow",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@org_tensorflow//third_party/py/numpy:headers",
    ],
)

pytype_strict_binary(
    name = "detect_tflite_model_overflow",
    srcs = ["detect_tflite_model_overflow.py"],
    deps = [
        ":detect_tflite_model_overflow_lib",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)
