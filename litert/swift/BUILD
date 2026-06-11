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

load("@build_bazel_rules_swift//swift:swift.bzl", "swift_interop_hint", "swift_library", "swift_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//litert/build_common:special_rule.bzl", "litert_gpu_accelerator_prebuilts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "litert_c",
    hdrs = ["Sources/LiteRtC.h"],
    aspect_hints = [":litert_c_swift_hint"],
    deps = [
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_compiled_model",
        "//litert/c:litert_custom_op_kernel",
        "//litert/c:litert_environment",
        "//litert/c:litert_environment_options",
        "//litert/c:litert_layout",
        "//litert/c:litert_model",
        "//litert/c:litert_model_types",
        "//litert/c:litert_opaque_options",
        "//litert/c:litert_options",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_requirements",
    ],
)

swift_interop_hint(
    name = "litert_c_swift_hint",
    module_name = "LiteRtC",
)

swift_library(
    name = "litert_swift",
    srcs = glob(["Sources/*.swift"]),
    copts = ["-enable-testing"],
    module_name = "LiteRt",
    deps = [
        ":litert_c",
        # Dependency placeholder: apple_frameworks Darwin,
        # Dependency placeholder: swift foundation,
    ],
)

swift_test(
    name = "litert_swift_test",
    srcs = glob(["Tests/*.swift"]),
    data = [
        "//litert/test:testdata/simple_model.tflite",
    ] + litert_gpu_accelerator_prebuilts(),
    deps = [
        ":litert_c",
        ":litert_swift",
        # Dependency placeholder: apple_frameworks Metal,
        # Dependency placeholder: swift foundation,
        # Dependency placeholder: swift xctest,
    ],
)
