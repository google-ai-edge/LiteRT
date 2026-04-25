# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
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

load("//litert/python:pytype.bzl", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "intel_openvino_backend",
    srcs = ["intel_openvino_backend.py"],
    deps = [
        ":target",
        "//litert/python/aot/core:aot_types",
        "//litert/python/aot/core:common",
        "//litert/python/aot/core:components",
        "//litert/python/aot/vendors:import_vendor",
    ],
)

pytype_strict_library(
    name = "target",
    srcs = ["target.py"],
    deps = ["//litert/python/aot/core:aot_types"],
)

pytype_strict_contrib_test(
    name = "intel_openvino_backend_test",
    srcs = ["intel_openvino_backend_test.py"],
    deps = [
        ":intel_openvino_backend",
        ":target",
        "//litert/python/aot/core:aot_types",
        "//litert/python/aot/core:components",
        "//litert/python/aot/core:test_common",
        "@absl_py//absl/testing:absltest",
    ],
)
