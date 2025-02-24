# Copyright 2024 Google LLC.
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

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "litert_environment",
    hdrs = ["litert_environment.h"],
    deps = [
        ":litert_any",
        ":litert_expected",
        ":litert_handle",
        "//third_party/absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_environment",
    ],
)

cc_library(
    name = "litert_event",
    hdrs = ["litert_event.h"],
    deps = [
        ":litert_expected",
        ":litert_handle",
        "//third_party/absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_event",
    ],
)

cc_library(
    name = "litert_any",
    hdrs = ["litert_any.h"],
    deps = [
        ":litert_expected",
        ":litert_macros",
        "//third_party/absl/strings:str_format",
        "//third_party/absl/strings:string_view",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
    ],
)

cc_test(
    name = "litert_any_test",
    srcs = [
        "litert_any_test.cc",
    ],
    linkopts = select({
        "//third_party/tensorflow:android": ["-llog"],
        "//conditions:default": [],
    }),
    deps = [
        ":litert_any",
        "//testing/base/public:gunit_main",
        "//litert/c:litert_common",
    ],
)

cc_library(
    name = "litert_model",
    srcs = ["litert_model.cc"],
    hdrs = [
        "litert_consts.h",
        "litert_model.h",
    ],
    deps = [
        ":litert_buffer_ref",
        ":litert_detail",
        ":litert_element_type",
        ":litert_expected",
        ":litert_handle",
        ":litert_layout",
        "//third_party/absl/container:inlined_vector",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_model",
    ],
)

cc_test(
    name = "litert_model_test",
    srcs = [
        "litert_model_test.cc",
    ],
    data = [
        "//litert/test:mlir_test_data",
    ],
    deps = [
        ":litert_element_type",
        ":litert_layout",
        ":litert_model",
        "//testing/base/public:gunit_main",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/core/model",
        "//litert/test:common",
    ],
)

cc_library(
    name = "litert_handle",
    hdrs = ["litert_handle.h"],
)

cc_library(
    name = "litert_tensor_buffer",
    hdrs = [
        "litert_tensor_buffer.h",
        "litert_tensor_buffer_requirements.h",
    ],
    deps = [
        ":litert_detail",
        ":litert_event",
        ":litert_expected",
        ":litert_handle",
        ":litert_model",
        "//third_party/absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_event",
        "//litert/c:litert_model",
        "//litert/c:litert_tensor_buffer",
        "//third_party/opencl_headers",
        "//third_party/tensorflow/lite/c:c_api_types",
    ],
)

cc_test(
    name = "litert_tensor_buffer_test",
    srcs = [
        "litert_tensor_buffer_test.cc",
    ],
    linkopts = select({
        "//third_party/tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        ":litert_layout",
        ":litert_model",
        ":litert_tensor_buffer",
        "//testing/base/public:gunit_main",
        "//third_party/absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/c:litert_tensor_buffer",
        "//litert/runtime:tensor_buffer",
    ],
)

cc_library(
    name = "litert_tensor_buffer_requirements",
    hdrs = [
        "litert_tensor_buffer_requirements.h",
    ],
    deps = [
        ":litert_detail",
        ":litert_handle",
        ":litert_macros",
        "//third_party/absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_tensor_buffer",
        "//litert/cc:litert_expected",
        "//third_party/tensorflow/lite/c:c_api_types",
    ],
)

cc_test(
    name = "litert_tensor_buffer_requirements_test",
    srcs = [
        "litert_tensor_buffer_requirements_test.cc",
    ],
    deps = [
        ":litert_tensor_buffer",
        "//testing/base/public:gunit_main",
        "//third_party/absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_tensor_buffer",
    ],
)

cc_library(
    name = "litert_buffer_ref",
    hdrs = [
        "litert_buffer_ref.h",
    ],
    deps = [
        "//third_party/absl/strings:str_format",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
    ],
)

cc_library(
    name = "litert_macros",
    hdrs = ["litert_macros.h"],
    deps = [
        ":litert_expected",
        "//third_party/absl/log:absl_check",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
    ],
)

cc_test(
    name = "litert_macros_test",
    srcs = ["litert_macros_test.cc"],
    deps = [
        ":litert_expected",
        ":litert_macros",
        "//testing/base/public:gunit_main",
        "//litert/c:litert_common",
    ],
)

cc_library(
    name = "litert_expected",
    hdrs = ["litert_expected.h"],
    deps = [
        ":litert_detail",
        "//third_party/absl/log:absl_check",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//litert/c:litert_common",
    ],
)

cc_test(
    name = "litert_expected_test",
    srcs = ["litert_expected_test.cc"],
    deps = [
        ":litert_buffer_ref",
        ":litert_expected",
        "//testing/base/public:gunit_main",
        "//third_party/absl/strings",
        "//litert/c:litert_common",
    ],
)

cc_library(
    name = "litert_detail",
    hdrs = ["litert_detail.h"],
    deps = [
        "//third_party/absl/log:absl_check",
        "//litert/c:litert_common",
    ],
)

# Dispatch Delegate of LiteRt.
# Warning: This API is not ABI stable and is subject to change.
cc_library(
    name = "litert_dispatch_delegate",
    hdrs = [
        "litert_dispatch_delegate.h",
    ],
    deps = [
        "//litert/c:litert_environment",
        "//litert/runtime/dispatch:dispatch_delegate",
        "//third_party/tensorflow/lite/c:c_api",
        "//third_party/tensorflow/lite/c:c_api_opaque",
        "//third_party/tensorflow/lite/c:c_api_types",
        "//third_party/tensorflow/lite/c:common",
        "//third_party/tensorflow/lite/delegates/utils:simple_opaque_delegate",
    ],
)

cc_test(
    name = "litert_buffer_ref_test",
    srcs = ["litert_buffer_ref_test.cc"],
    deps = [
        ":litert_buffer_ref",
        "//testing/base/public:gunit_main",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
        "//litert/core/util:flatbuffer_tools",
    ],
)

cc_library(
    name = "litert_element_type",
    hdrs = ["litert_element_type.h"],
    deps = ["//litert/c:litert_model"],
)

cc_test(
    name = "litert_element_type_test",
    srcs = ["litert_element_type_test.cc"],
    deps = [
        ":litert_element_type",
        "//testing/base/public:gunit_main",
    ],
)

cc_library(
    name = "litert_model_predicates",
    srcs = ["litert_model_predicates.cc"],
    hdrs = ["litert_model_predicates.h"],
    deps = [
        ":litert_detail",
        ":litert_model",
        "//third_party/absl/container:inlined_vector",
        "//third_party/absl/types:span",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
    ],
)

cc_library(
    name = "litert_layout",
    hdrs = ["litert_layout.h"],
    deps = [
        "//third_party/absl/types:span",
        "//litert/c:litert_layout",
    ],
)

cc_test(
    name = "litert_model_predicates_test",
    srcs = ["litert_model_predicates_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
    ],
    deps = [
        ":litert_element_type",
        ":litert_model",
        ":litert_model_predicates",
        "//testing/base/public:gunit_main",
        "//third_party/absl/container:inlined_vector",
        "//third_party/absl/log:absl_check",
        "//third_party/absl/types:span",
        "//litert/c:litert_op_code",
        "//litert/test:common",
    ],
)

cc_test(
    name = "litert_layout_test",
    srcs = ["litert_layout_test.cc"],
    deps = [
        ":litert_layout",
        "//testing/base/public:gunit_main",
    ],
)

cc_library(
    name = "litert_compiled_model",
    srcs = ["litert_compiled_model.cc"],
    hdrs = ["litert_compiled_model.h"],
    deps = [
        ":litert_environment",
        ":litert_expected",
        ":litert_handle",
        ":litert_macros",
        ":litert_model",
        ":litert_tensor_buffer",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/c:litert_compiled_model",
        "//litert/c:litert_compiled_model_options",
        "//litert/c:litert_environment",
        "//litert/c:litert_model",
        "//litert/c:litert_tensor_buffer",
        "//third_party/tensorflow/lite:framework",
        "//third_party/tensorflow/lite/c:c_api_opaque",
        "//third_party/tensorflow/lite/c:c_api_types",
        "//third_party/tensorflow/lite/c:common",
        "//third_party/tensorflow/lite/core:cc_api_stable",
        "//third_party/tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_test(
    name = "litert_compiled_model_test",
    srcs = ["litert_compiled_model_test.cc"],
    data = [
        "//litert/test:testdata/simple_model.tflite",
    ],
    deps = [
        ":litert_compiled_model",
        ":litert_environment",
        ":litert_model",
        ":litert_tensor_buffer",
        "//testing/base/public:gunit_main",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/log:absl_log",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_tensor_buffer",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model",
        "//third_party/tensorflow/lite:framework",
        "//third_party/tensorflow/lite/c:c_api_opaque",
        "//third_party/tensorflow/lite/c:common",
        "//third_party/tensorflow/lite/kernels:builtin_ops",
    ],
)

# copybara:uncomment_begin(google-only)
# cc_test(
#     name = "litert_compiled_model_gpu_test",
#     srcs = ["litert_compiled_model_gpu_test.cc"],
#     data = [
#         "//litert/test:testdata/simple_model.tflite",
#     ],
#     tags = [
#         "requires-gpu-nvidia",
#     ],
#     deps = [
#         ":litert_compiled_model",
#         ":litert_environment",
#         ":litert_model",
#         ":litert_tensor_buffer",
#         "//testing/base/public:gunit_main",
#         "//third_party/absl/debugging:leak_check",
#         "//third_party/absl/log:absl_log",
#         "//third_party/absl/strings:string_view",
#         "//third_party/absl/types:span",
#         "//third_party/odml/infra/ml_drift_delegate:ml_drift_cl_accelerator",  # buildcleaner: keep
#         "//litert/c:litert_common",
#         "//litert/test:common",
#         "//litert/test:simple_model",
#         "//third_party/tensorflow/lite:framework",
#         "//third_party/tensorflow/lite/c:c_api_opaque",
#         "//third_party/tensorflow/lite/c:common",
#         "//third_party/tensorflow/lite/kernels:builtin_ops",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "litert_tensor_buffer_utils",
    srcs = ["litert_tensor_buffer_utils.cc"],
    hdrs = ["litert_tensor_buffer_utils.h"],
    deps = [
        "//litert/c:litert_logging",
        "//litert/c:litert_tensor_buffer",
    ],
)

exports_files(srcs = glob(["litert_*.h"]))
