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
load("@rules_python//python:py_library.bzl", "py_library")
load("//litert/build_common:litert_build_defs.bzl", "if_google")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment "//visibility:private",
        "//visibility:public",  # copybara:comment
    ],
    licenses = ["notice"],
)

py_library(
    name = "model_utils",
    srcs = glob(
        include = ["**/*.py"],
        exclude = [
            "testing.py",
            "test/**/*",
            "model_explorer_integration/**/*",
        ],
    ),
    deps = [
        # Dependency placeholder: gfile,
        "//litert/python/mlir/_mlir_libs:model_utils_ext",
        "@lark",
        "@ml_dtypes_py//ml_dtypes",
        "@org_tensorflow//third_party/py/numpy:headers",
        "@xdsl",
    ] + if_google(
        [
            "//third_party/py/mlir/_mlir_libs:_mlirRegisterEverything",
            "//third_party/py/mlir",
            "//third_party/py/mlir:arithmetic_dialect",
            "//third_party/py/mlir:builtin_dialect",
            "//third_party/py/mlir:chlo_dialect",
            "//third_party/py/mlir:func_dialect",
            "//third_party/py/mlir:ir",
            "//third_party/py/mlir:pass_manager",
            "//third_party/py/mlir:stablehlo_dialect",
        ],
        [
            "//litert/python/mlir",
        ],
    ),
)

pytype_strict_library(
    name = "testing",
    testonly = True,
    srcs = ["testing.py"],
    deps = [
        ":model_utils",
        "@absl_py//absl/testing:absltest",
    ] + if_google(
        [
            "//third_party/py/mlir:ir",
        ],
        [
            "//litert/python/mlir",
        ],
    ),
)
