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

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@org_tensorflow//tensorflow:pytype.default.bzl", "pytype_strict_contrib_test", "pytype_strict_library")
load("//litert/python/aot/core:build_defs.bzl", "tflxx_deps_if_enabled")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_python_internal_users"],
)

bool_flag(
    name = "enable_tflxx",
    build_setting_default = False,
)

config_setting(
    name = "tflxx_enabled",
    flag_values = {":enable_tflxx": "True"},
)

pytype_strict_library(
    name = "test_common",
    srcs = ["test_common.py"],
    data = [
        "//litert/test:testdata/add_simple.tflite",
        "//litert/test:testdata/mul_simple.tflite",
    ],
    deps = [
        ":common",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_library(
    name = "common",
    srcs = ["common.py"],
    visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "types",
    srcs = ["types.py"],
    visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "apply_plugin",
    srcs = ["apply_plugin.py"],
    data = [
        "//litert/tools:apply_plugin_main",
    ],
    deps = [
        ":common",
        ":components",
        ":types",
    ],
)

pytype_strict_contrib_test(
    name = "apply_plugin_test",
    srcs = ["apply_plugin_test.py"],
    # San builds result in too large of a build (vendor data deps). Remove once build size is trimmed down.
    tags = [
        "noasan",
        "nosan",
    ],
    deps = [
        ":apply_plugin",
        ":test_common",
        ":types",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_contrib_test(
    name = "common_test",
    srcs = ["common_test.py"],
    data = [
        "//litert/test:testdata/add_simple.tflite",
    ],
    deps = [
        ":common",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_library(
    name = "mlir_transforms",
    srcs = ["mlir_transforms.py"],
    deps = [
        ":components",
        ":tflxx_util",
        ":types",
    ],
)

pytype_strict_library(
    name = "components",
    srcs = ["components.py"],
    visibility = ["//visibility:public"],
    deps = [":types"],
)

pytype_strict_library(
    name = "tflxx_util",
    srcs = ["tflxx_util.py"],
    deps = [
        ":types",
    ] + tflxx_deps_if_enabled(),
)

pytype_strict_contrib_test(
    name = "mlir_transforms_test",
    srcs = ["mlir_transforms_test.py"],
    deps = [
        ":common",
        ":mlir_transforms",
        ":test_common",
        ":types",
        "@absl_py//absl/testing:absltest",
    ],
)
