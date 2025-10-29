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

load("@org_tensorflow//tensorflow:pytype.default.bzl", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_python_internal_users"],
)

pytype_strict_library(
    name = "aot_compile",
    srcs = ["aot_compile.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":prepare_for_npu",
        "//litert/python/aot/core:apply_plugin",
        "//litert/python/aot/core:components",
        "//litert/python/aot/core:mlir_transforms",
        "//litert/python/aot/core:types",
        "//litert/python/aot/vendors:import_vendor",
    ],
)

pytype_strict_contrib_test(
    name = "aot_compile_test",
    srcs = ["aot_compile_test.py"],
    data = [
        "//litert/test:testdata/mobilenet_v2_1.0_224.tflite",
        "//litert/test:testdata/torchaudio_conformer.tflite",
    ],
    # Re-enable sans once test libs can be linked in a sans friendly way.
    tags = [
        "manual",
        "noasan",
        "nomsan",
        "nosan",
        "notap",
        "notsan",
    ],
    deps = [
        ":aot_compile",
        "//litert/python/aot/core:common",
        "//litert/python/aot/core:test_common",
        "//litert/python/aot/core:types",
        "//litert/python/aot/vendors/mediatek:target",
        "//litert/python/aot/vendors/qualcomm:target",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_library(
    name = "prepare_for_npu",
    srcs = ["prepare_for_npu.py"],
    deps = [
        "//litert/python/aot/core:common",
        "//litert/python/aot/core:components",
        "//litert/python/aot/core:types",
        "//litert/python/aot/vendors",  # buildcleaner: keep
        "//litert/python/aot/vendors:import_vendor",
        "@tqdm",
    ],
)

pytype_strict_contrib_test(
    name = "prepare_for_npu_test",
    srcs = ["prepare_for_npu_test.py"],
    deps = [
        ":prepare_for_npu",
        "//litert/python/aot/core:common",
        "//litert/python/aot/core:components",
        "//litert/python/aot/core:test_common",
        "//litert/python/aot/core:types",
        "//litert/python/aot/vendors/example:example_backend",
        "@absl_py//absl/testing:absltest",
    ],
)
