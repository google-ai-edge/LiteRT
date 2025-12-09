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

# copybara:uncomment_begin(google-only)
# load("@rules_cc//cc:cc_library.bzl", "cc_library")
# load("@rules_cc//cc:cc_test.bzl", "cc_test")
# copybara:uncomment_end

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert/c:__subpackages__",
        # "//litert/cc:__subpackages__",
        # "//litert/test:__subpackages__",
        # "//litert/runtime:__pkg__",
        # "//litert/runtime/accelerators:__subpackages__",
        # "//litert/runtime/dispatch:__subpackages__",
        # "//litert/vendors/c:__subpackages__",
        # "//litert/kotlin:__subpackages__",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
)

cc_library(
    name = "litert_accelerator",
    srcs = ["litert_accelerator.cc"],
    hdrs = ["litert_accelerator.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "//litert/core:environment",
        "//litert/runtime:accelerator",
    ],
)

cc_test(
    name = "litert_accelerator_test",
    srcs = ["litert_accelerator_test.cc"],
    deps = [
        ":litert_accelerator",
        ":litert_accelerator_registration",
        "//litert/c:litert_common",
        "//litert/c:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/runtime:accelerator",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_accelerator_registration",
    srcs = ["litert_accelerator_registration.cc"],
    hdrs = ["litert_accelerator_registration.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_metrics",
        "//litert/c:litert_opaque_options",
        "//litert/cc:litert_expected",
        "//litert/core:environment",
        "//litert/runtime:accelerator",
        "//litert/runtime:accelerator_registry",
    ],
    # Keep the symbols in libLiteRt.dylib.
    alwayslink = 1,
)

cc_test(
    name = "litert_accelerator_registration_test",
    srcs = ["litert_accelerator_registration_test.cc"],
    deps = [
        ":litert_accelerator_registration",
        "//litert/c:litert_common",
        "//litert/c:litert_environment",
        "//litert/c:litert_opaque_options",
        "//litert/runtime:accelerator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_dispatch_delegate",
    hdrs = ["litert_dispatch_delegate.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options",
        "//litert/c:litert_metrics",
        "//litert/runtime/dispatch:dispatch_delegate",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//tflite/c:c_api",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/delegates/utils:simple_opaque_delegate",
    ],
)

cc_library(
    name = "litert_tensor_buffer_registry",
    srcs = ["litert_tensor_buffer_registry.cc"],
    hdrs = ["litert_tensor_buffer_registry.h"],
    visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert/c:__subpackages__",
        # "//litert/cc:__subpackages__",
        # "//litert/cc/internal:__subpackages__",
        # "//litert/runtime:__subpackages__",
        # "//litert/kotlin:__subpackages__",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_custom_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_macros",
        "//litert/core:environment",
        "//litert/runtime:tensor_buffer_registry",
    ],
    # Keep the symbols in libLiteRt.dylib.
    alwayslink = 1,
)

# Collection of headers for dispatch APIs.
# Warning: Do not use this target to include common C API headers. Use individual targets instead.
cc_library(
    name = "litert_dispatch_headers",
    hdrs = [
        # Internal C API headers.
        ":litert_accelerator.h",
        ":litert_accelerator_registration.h",
        ":litert_dispatch_delegate.h",
        ":litert_external_litert_buffer_context.h",
        ":litert_tensor_buffer_registry.h",
        "//litert/c:litert_any.h",
        "//litert/c:litert_common.h",
        "//litert/c:litert_compiled_model.h",
        "//litert/c:litert_custom_tensor_buffer.h",
        "//litert/c:litert_custom_op_kernel.h",
        "//litert/c:litert_environment.h",
        "//litert/c:litert_environment_options.h",
        "//litert/c:litert_event.h",
        "//litert/c:litert_event_type.h",
        "//litert/c:litert_gl_types.h",
        "//litert/c:litert_layout.h",
        "//litert/c:litert_metrics.h",
        "//litert/c:litert_model_types.h",
        "//litert/c:litert_op_code.h",
        "//litert/c:litert_op_options.h",
        "//litert/c:litert_opaque_options.h",
        "//litert/c:litert_options.h",
        "//litert/c:litert_platform_support.h",
        "//litert/c:litert_tensor_buffer.h",
        "//litert/c:litert_tensor_buffer_requirements.h",
        "//litert/c:litert_tensor_buffer_types.h",
        # Needed for litert/c/litert_op_code.h
        "//tflite:builtin_ops.h",
        # Needed for litert/c/litert_model.h
        "//tflite/c:tensorflowlite_c_api_hdrs_filegroup",
        "//tflite/c:tensorflowlite_c_impl_hdrs_filegroup",
    ],  # Export all header files (.h) in this directory
    visibility = [
        "//litert/runtime:__pkg__",  # Temporaray only for metal_memory
        "//litert/runtime/dispatch:__pkg__",
        "//litert/vendors/c:__pkg__",
    ],
    deps = [
        "//litert/build_common:build_config",
        "//tflite/core/api",
        "//tflite/profiling:memory_info",
        "@com_google_absl//absl/types:span",
        "@opencl_headers",
    ],
)

cc_library(
    name = "litert_external_litert_buffer_context",
    srcs = ["litert_external_litert_buffer_context.cc"],
    hdrs = ["litert_external_litert_buffer_context.h"],
    # copybara:uncomment_begin(oss litert_lm)
    # visibility = [
    # "//litert:__subpackages__",
    # "//third_party/odml/litert_lm/runtime/executor:__pkg__",
    # ],
    # copybara:uncomment_end
    deps = [
        ":litert_dispatch_headers",
        "//litert/c:litert_common",
        "//litert/cc:litert_macros",
        "//litert/runtime:external_litert_buffer_context",
        "//tflite/c:common",
    ],
    alwayslink = 1,
)

# This target has deliberately limited visibility:
# it is exposed only to the LiteRT runtime implementation, and in
# particular to LiteRT runtime's built-in accelerator implementations that are
# implemented using TF Lite delegates.
# It should not be made visible elsewhere, to avoid exposing the dependency on
# the TF Lite API.
cc_library(
    name = "litert_delegate_wrapper",
    srcs = ["litert_delegate_wrapper.cc"],
    hdrs = ["litert_delegate_wrapper.h"],
    visibility = [
        "//litert/runtime:__pkg__",
        "//litert/runtime/accelerators:__subpackages__",
    ],
    deps = [
        "//litert/c:litert_common",
        "//tflite/c:common",
    ],
)

cc_library(
    name = "litert_logging",
    srcs = ["litert_logging.cc"],
    hdrs = ["litert_logging.h"],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-llog"],
        "//conditions:default": [],
    }),
    visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "@mediapipe//mediapipe/tasks/web/genai:__subpackages__",
        # "//litert:__subpackages__",
        # "//third_party/odml/litert_lm/c:__pkg__",
        # "//third_party/odml/litert_lm/kotlin:__subpackages__",
        # "//third_party/odml/litert_lm/runtime/engine:__pkg__",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    deps = [
        "//litert/c:litert_common",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_logging_test",
    srcs = ["litert_logging_test.cc"],
    deps = [
        ":litert_logging",
        "//litert/c:litert_common",
        "//litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

# Header files that should be part of the shared library: litert/c:litert_runtime_c_api_shared_lib.
_HEADER_FILES_FOR_SHARED_LIB = [
    "litert_accelerator.h",
    "litert_accelerator_registration.h",
    "litert_delegate_wrapper.h",
    "litert_dispatch_delegate.h",
    "litert_external_litert_buffer_context.h",
    "litert_logging.h",
    "litert_tensor_buffer_registry.h",
]

exports_files(srcs = _HEADER_FILES_FOR_SHARED_LIB)

filegroup(
    name = "internal_headers",
    srcs = _HEADER_FILES_FOR_SHARED_LIB,
    visibility = [
        "//litert/c:__pkg__",
        "//litert/cc:__pkg__",
    ],
)
