load("//tflite/experimental/litert/build_common:litert_build_defs.bzl", "export_lrt_runtime_only_linkopt", "export_lrt_runtime_only_script")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment "@org_tensorflow//third_party/mediapipe/calculators/tensor:__subpackages__",
        # copybara:uncomment "@org_tensorflow//third_party/odml/infra:__subpackages__",
        "//tflite/experimental/litert:__subpackages__",
    ],
)

cc_library(
    name = "litert_common",
    srcs = ["litert_common.cc"],
    hdrs = ["litert_common.h"],
)

cc_test(
    name = "litert_common_test",
    srcs = ["litert_common_test.cc"],
    deps = [
        ":litert_common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_any",
    hdrs = ["litert_any.h"],
)

cc_library(
    name = "litert_environment",
    srcs = ["litert_environment.cc"],
    hdrs = ["litert_environment.h"],
    deps = [
        ":litert_any",
        ":litert_common",
        ":litert_logging",
        "//tflite/experimental/litert/core:environment",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "litert_logging",
    srcs = [
        "litert_logging.cc",
    ],
    hdrs = [
        "litert_logging.h",
    ],
    deps = [
        ":litert_common",
        "//tflite:minimal_logging",
    ],
)

cc_test(
    name = "litert_logging_test",
    srcs = [
        "litert_logging_test.cc",
    ],
    deps = [
        ":litert_common",
        ":litert_logging",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_layout",
    hdrs = ["litert_layout.h"],
    deps = [
        ":litert_common",
        ":litert_op_code",
        "//tflite/core/c:c_api_types",
        "//tflite/experimental/litert/cc:litert_buffer_ref",
        "//tflite/experimental/litert/cc:litert_macros",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "litert_model",
    srcs = ["litert_model.cc"],
    hdrs = ["litert_model.h"],
    deps = [
        ":litert_common",
        ":litert_layout",
        ":litert_op_code",
        "//tflite/core/c:c_api_types",
        "//tflite/experimental/litert/cc:litert_buffer_ref",
        "//tflite/experimental/litert/core/model",
        "//tflite/experimental/litert/core/model:model_load",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_model_test",
    srcs = ["litert_model_test.cc"],
    deps = [
        ":litert_common",
        ":litert_model",
        ":litert_op_code",
        "//tflite/experimental/litert/cc:litert_buffer_ref",
        "//tflite/experimental/litert/core/model",
        "//tflite/experimental/litert/core/util:flatbuffer_tools",
        "//tflite/experimental/litert/test:test_macros",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_op_code",
    hdrs = ["litert_op_code.h"],
    deps = ["//tflite:builtin_ops"],
)

cc_library(
    name = "litert_options",
    srcs = ["litert_options.cc"],
    hdrs = [
        "litert_options.h",
    ],
    deps = [
        ":litert_common",
        ":litert_op_code",
        "//tflite/c:c_api_types",
        "//tflite/experimental/litert/core/model",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/core:model_builder_base",
    ],
)

cc_test(
    name = "litert_options_test",
    srcs = ["litert_options_test.cc"],
    data = [
        "//tflite/experimental/litert/test:mlir_test_data",
    ],
    tags = ["no_oss"],
    deps = [
        ":litert_options",
        "//tflite/experimental/litert/test:common",
        "//tflite/experimental/litert/test:test_macros",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_event",
    srcs = ["litert_event.cc"],
    hdrs = ["litert_event.h"],
    deps = [
        ":litert_common",
        ":litert_logging",
        "//tflite/experimental/litert/runtime:event",
    ],
)

cc_library(
    name = "litert_tensor_buffer",
    srcs = [
        "litert_tensor_buffer.cc",
        "litert_tensor_buffer_requirements.cc",
    ],
    hdrs = [
        "litert_tensor_buffer.h",
        "litert_tensor_buffer_requirements.h",
    ],
    deps = [
        ":litert_common",
        ":litert_event",
        ":litert_logging",
        ":litert_model",
        "//tflite/experimental/litert/runtime:tensor_buffer",
        "@com_google_absl//absl/types:span",
        "@opencl_headers",
    ],
)

cc_test(
    name = "litert_tensor_buffer_test",
    srcs = [
        "litert_tensor_buffer_test.cc",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    # require GPU to run OpenCL tests.
    tags = [
        "requires-gpu-nvidia",
    ],
    deps = [
        ":litert_common",
        ":litert_model",
        ":litert_tensor_buffer",
        "//tflite/experimental/litert/cc:litert_layout",
        "//tflite/experimental/litert/runtime:tensor_buffer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "litert_tensor_buffer_requirements_test",
    srcs = [
        "litert_tensor_buffer_requirements_test.cc",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        ":litert_common",
        ":litert_tensor_buffer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_dispatch_delegate",
    hdrs = [
        "litert_dispatch_delegate.h",
    ],
    deps = [
        ":litert_environment",
        "//tflite/c:c_api",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/delegates/utils:simple_opaque_delegate",
        "//tflite/experimental/litert/runtime/dispatch:dispatch_delegate",
        "//tflite/experimental/litert/vendors/c:litert_dispatch_c_api",
    ],
)

cc_library(
    name = "litert_compiled_model_options",
    srcs = ["litert_compiled_model_options.cc"],
    hdrs = [
        "litert_compiled_model_options.h",
    ],
    deps = [
        ":litert_accelerator_options",
        ":litert_common",
        ":litert_logging",
    ],
)

cc_test(
    name = "litert_compiled_model_options_test",
    srcs = ["litert_compiled_model_options_test.cc"],
    deps = [
        ":litert_accelerator_options",
        ":litert_common",
        ":litert_compiled_model_options",
        "//tflite/experimental/litert/core:accelerator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_compiled_model",
    srcs = ["litert_compiled_model.cc"],
    hdrs = [
        "litert_compiled_model.h",
    ],
    deps = [
        ":litert_common",
        ":litert_compiled_model_options",
        ":litert_environment",
        ":litert_logging",
        ":litert_model",
        ":litert_tensor_buffer",
        "//tflite/c:c_api_types",
        "//tflite/experimental/litert/runtime:compiled_model",
    ],
)

cc_test(
    name = "litert_compiled_model_test",
    srcs = [
        "litert_compiled_model_test.cc",
    ],
    data = [
        "//tflite/experimental/litert/test:testdata/simple_model.tflite",
    ],
    deps = [
        ":litert_common",
        ":litert_compiled_model",
        ":litert_compiled_model_options",
        ":litert_environment",
        ":litert_model",
        ":litert_tensor_buffer",
        "//tflite/experimental/litert/test:common",
        "//tflite/experimental/litert/test:simple_model",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

# The same test as `litert_compiled_model_test` but using the shared library `libLiteRtRuntimeCApi.so`.
cc_test(
    name = "litert_compiled_model_shared_lib_test",
    srcs = [
        "litert_compiled_model_test.cc",
    ],
    data = [
        "//tflite/experimental/litert/test:testdata/simple_model.tflite",
    ],
    deps = [
        ":litert_runtime_c_api_shared_lib",
        "//tflite/experimental/litert/test:common",
        "//tflite/experimental/litert/test:simple_model",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_accelerator_options",
    srcs = ["litert_accelerator_options.cc"],
    hdrs = ["litert_accelerator_options.h"],
    deps = [
        ":litert_common",
        "//tflite/experimental/litert/core:accelerator",
    ],
)

cc_test(
    name = "litert_accelerator_options_test",
    srcs = ["litert_accelerator_options_test.cc"],
    deps = [
        ":litert_accelerator_options",
        ":litert_common",
        "//tflite/experimental/litert/core:accelerator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_accelerator",
    srcs = ["litert_accelerator.cc"],
    hdrs = ["litert_accelerator.h"],
    deps = [
        ":litert_common",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/core:accelerator",
        "//tflite/experimental/litert/core:environment",
    ],
)

cc_test(
    name = "litert_accelerator_test",
    srcs = ["litert_accelerator_test.cc"],
    deps = [
        ":litert_accelerator",
        ":litert_accelerator_registration",
        ":litert_common",
        ":litert_environment",
        "//tflite/experimental/litert/core:accelerator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_accelerator_registration",
    srcs = ["litert_accelerator_registration.cc"],
    hdrs = ["litert_accelerator_registration.h"],
    deps = [
        ":litert_accelerator",
        ":litert_common",
        ":litert_compiled_model",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/core:accelerator",
        "//tflite/experimental/litert/core:environment",
    ],
)

cc_test(
    name = "litert_accelerator_registration_test",
    srcs = ["litert_accelerator_registration_test.cc"],
    deps = [
        ":litert_accelerator",
        ":litert_accelerator_registration",
        ":litert_common",
        ":litert_compiled_model",
        ":litert_environment",
        "//tflite/experimental/litert/core:accelerator",
        "@com_google_googletest//:gtest_main",
    ],
)

filegroup(
    name = "litert_model_srcs",
    srcs = ["litert_model.cc"],
    visibility = ["//tflite/experimental/litert/core/model:__pkg__"],
)

filegroup(
    name = "litert_model_hdrs",
    srcs = ["litert_model.h"],
    visibility = ["//tflite/experimental/litert/core/model:__pkg__"],
)

# Collection of all C API targets.
LITERT_C_API_COMMON_DEPS = [
    ":litert_accelerator",
    ":litert_accelerator_registration",
    ":litert_any",
    ":litert_common",
    ":litert_compiled_model",
    ":litert_compiled_model_options",
    ":litert_dispatch_delegate",
    ":litert_event",
    ":litert_environment",
    ":litert_layout",
    ":litert_logging",
    ":litert_model",
    ":litert_op_code",
    ":litert_options",
    ":litert_tensor_buffer",
]

# This test verifies that the C API header files can build via C compiler.
cc_test(
    name = "litert_c_api_common_test",
    srcs = ["litert_c_api_common_test.c"],
    copts = ["--std=c11"],
    linkopts = ["-ldl"],
    deps = LITERT_C_API_COMMON_DEPS,
)

# Build `litert/c:litert_runtime_c_api_so` for `libLiteRtRuntimeCApi.so`.
cc_shared_library(
    name = "litert_runtime_c_api_so",
    additional_linker_inputs = export_lrt_runtime_only_script(),
    shared_lib_name = "libLiteRtRuntimeCApi.so",
    user_link_flags = export_lrt_runtime_only_linkopt() + select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = LITERT_C_API_COMMON_DEPS,
)

# This is cc_library target based on `libLiteRtRuntimeCApi.so`.
cc_library(
    name = "litert_runtime_c_api_shared_lib",
    srcs = [":litert_runtime_c_api_so"],
    hdrs = glob(["litert_*.h"]) + [
        # Needed for litert/c/litert_op_code.h
        "//tflite:builtin_ops.h",
        # Neeeded for litert/c/litert_model.h
        "//tflite/c:tensorflowlite_c_api_hdrs_filegroup",
        "//tflite/core/c:headers_filegroup",
        "//tflite/experimental/litert/vendors/c:litert_dispatch.h",
    ],
    linkstatic = 1,
    deps = ["@opencl_headers"],
)

exports_files(srcs = glob(["litert_*.h"]))
