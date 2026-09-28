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

load("@rules_python//python:py_library.bzl", "py_library")
load("//litert/build_common:symlink_files.bzl", "symlink_inputs")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "mlir",
    srcs = [
        "__init__.py",
    ],
    deps = [
        ":arithmetic_dialect",
        ":builtin_dialect",
        ":core",
        ":extras",
        ":func_dialect",
        ":ir",
        ":pass_manager",
        ":stablehlo_dialect",
    ],
)

symlink_inputs(
    name = "core",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        "dialects": ["@llvm-project//mlir/python:DialectCorePyFiles"],
    }},
)

symlink_inputs(
    name = "extras",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        "extras": ["@llvm-project//mlir/python:ExtrasPyFiles"],
    }},
    deps = [
        ":ir",
        "//litert/python/mlir/_mlir_libs",
    ],
)

symlink_inputs(
    name = "ir",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        ".": [
            "@llvm-project//mlir/python:IRPyFiles",
        ],
    }},
    deps = [
        "//litert/python/mlir/_mlir_libs",
    ],
)

symlink_inputs(
    name = "func_dialect",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        "dialects": ["@llvm-project//mlir/python:FuncPyFiles"],
    }},
    deps = [
        ":core",
        ":ir",
        "//litert/python/mlir/_mlir_libs",
    ],
)

symlink_inputs(
    name = "arithmetic_dialect",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        "dialects": ["@llvm-project//mlir/python:ArithOpsPyFiles"],
    }},
    deps = [
        ":core",
        ":ir",
        "//litert/python/mlir/_mlir_libs",
    ],
)

symlink_inputs(
    name = "builtin_dialect",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        "dialects": ["@llvm-project//mlir/python:BuiltinOpsPyFiles"],
    }},
    deps = [
        ":core",
        ":extras",
        ":ir",
        "//litert/python/mlir/_mlir_libs",
    ],
)

symlink_inputs(
    name = "stablehlo_dialect",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        "dialects": ["@stablehlo//:stablehlo_ops_py_files"],
    }},
    deps = [
        ":core",
        ":ir",
        "//litert/python/mlir/_mlir_libs",
    ],
)

symlink_inputs(
    name = "pass_manager",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        ".": [
            "@llvm-project//mlir/python:PassManagerPyFiles",
        ],
    }},
    deps = [
        "//litert/python/mlir/_mlir_libs",
    ],
)
