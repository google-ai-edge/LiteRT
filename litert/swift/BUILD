# Copyright 2026 Google LLC
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tools/build_defs/swift:swift_interop_hint.bzl", "swift_interop_hint")
load("//tools/build_defs/swift:swift_library.bzl", "swift_library")
load("//tools/build_defs/swift:swift_test.bzl", "swift_test")

package(
    default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = [
        "//litert:litert_public",
    ],
)

swift_interop_hint(
    name = "litert_c_api_hint",
    module_name = "LiteRtC",
)

cc_library(
    name = "litert_c_api_swift",
    hdrs = [
        "//litert/build_common:build_config.h",
        "//litert/c:litert_any.h",
        "//litert/c:litert_common.h",
        "//litert/c:litert_compiled_model.h",
        "//litert/c:litert_custom_op_kernel.h",
        "//litert/c:litert_custom_tensor_buffer.h",
        "//litert/c:litert_environment.h",
        "//litert/c:litert_environment_options.h",
        "//litert/c:litert_event.h",
        "//litert/c:litert_event_type.h",
        "//litert/c:litert_gl_types.h",
        "//litert/c:litert_layout.h",
        "//litert/c:litert_model.h",
        "//litert/c:litert_model_types.h",
        "//litert/c:litert_op_code.h",
        "//litert/c:litert_opaque_options.h",
        "//litert/c:litert_opencl_types.h",
        "//litert/c:litert_options.h",
        "//litert/c:litert_tensor_buffer.h",
        "//litert/c:litert_tensor_buffer_requirements.h",
        "//litert/c:litert_tensor_buffer_types.h",
        "//litert/c:litert_webgpu_types.h",
        "//litert/c/internal:litert_scheduling_info.h",
        "//litert/c/options:litert_cpu_options.h",
        "//litert/c/options:litert_gpu_options.h",
    ],
    aspect_hints = [":litert_c_api_hint"],
    deps = [
        "//litert/build_common:build_config",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_compiled_model",
        "//litert/c:litert_custom_tensor_buffer",
        "//litert/c:litert_environment",
        "//litert/c:litert_event",
        "//litert/c:litert_layout",
        "//litert/c:litert_model",
        "//litert/c:litert_options",
        "//litert/c:litert_tensor_buffer",
        "//litert/c/internal:litert_scheduling_info",
        "//litert/c/options:litert_cpu_options",
        "//litert/c/options:litert_gpu_options",
        "@opencl_headers",
    ],
)

swift_library(
    name = "LiteRt",
    srcs = [
        "Common.swift",
        "Environment.swift",
        "Model.swift",
        "TensorBuffer.swift",
        "TensorType.swift",
    ],
    module_name = "LiteRt",
    deps = [
        ":litert_c_api_swift",
        "//third_party/apple_frameworks:Metal",
        "//third_party/swift/foundation",
    ],
)

swift_library(
    name = "LiteRtTestsLib",
    testonly = True,
    srcs = ["LiteRtTests.swift"],
    data = ["//litert/test:testdata/simple_model.tflite"],
    deps = [
        ":LiteRt",
        ":litert_c_api_swift",
        "//third_party/swift/foundation",
        "//third_party/swift/xctest",
    ],
)

swift_test(
    name = "LiteRtTests",
    deps = [":LiteRtTestsLib"],
)
