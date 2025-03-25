# copybara:uncomment_begin(google-only)
# load("//devtools/deps/check:deps_check.bzl", "check_dependencies")
# 
# copybara:uncomment_end
load("//litert/build_common:litert_build_defs.bzl", "copy_file", "export_lrt_runtime_only_linkopt", "export_lrt_runtime_only_script")
load("//litert/build_common:special_rule.bzl", "gles_deps", "gles_headers", "gles_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
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
        ":litert_common",
        ":litert_environment_options",
        "@com_google_absl//absl/types:span",
        "//litert/cc:litert_macros",
        "//litert/core:environment",
        "//litert/runtime:gpu_environment",
        "//litert/runtime/accelerators:auto_registration",
    ],
)

cc_library(
    name = "litert_environment_options",
    srcs = ["litert_environment_options.cc"],
    hdrs = ["litert_environment_options.h"],
    deps = [
        ":litert_any",
        ":litert_common",
        "//litert/cc:litert_macros",
        "//litert/core:environment_options",
    ],
)

cc_library(
    name = "litert_environment_options_header",
    hdrs = ["litert_environment_options.h"],
    tags = ["avoid_dep"],
    deps = [
        ":litert_any",
        ":litert_common",
    ],
)

cc_test(
    name = "litert_environment_options_test",
    srcs = ["litert_environment_options_test.cc"],
    deps = [
        ":litert_any",
        ":litert_common",
        ":litert_environment_options",
        "@com_google_googletest//:gtest_main",
        "//litert/core:environment_options",
        "//litert/test:matchers",
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
        "@com_google_absl//absl/strings:string_view",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_macros",
        "//tflite/core/c:private_c_api_types",
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
        "@com_google_absl//absl/strings:string_view",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_macros",
        "//litert/core/model",
        "//litert/core/model:model_load",
        "//litert/core/model:model_serialize",
        "//tflite/core/c:private_c_api_types",
    ],
)

cc_test(
    name = "litert_model_test",
    srcs = ["litert_model_test.cc"],
    deps = [
        ":litert_common",
        ":litert_model",
        ":litert_op_code",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//litert/cc:litert_buffer_ref",
        "//litert/core/model",
        "//litert/core/util:flatbuffer_tools",
        "//litert/test:matchers",
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
        "//litert/core/model",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tflite/c:c_api_types",
    ],
)

cc_test(
    name = "litert_options_test",
    srcs = ["litert_options_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
    ],
    tags = ["no_oss"],
    deps = [
        ":litert_common",
        ":litert_options",
        "@com_google_googletest//:gtest_main",
        "//litert/test:common",
        "//litert/test:matchers",
    ],
)

cc_library(
    name = "litert_event_type",
    hdrs = ["litert_event_type.h"],
)

cc_library(
    name = "litert_event",
    srcs = ["litert_event.cc"],
    hdrs = ["litert_event.h"],
    deps = [
        ":litert_common",
        ":litert_event_type",
        ":litert_logging",
        "//litert/cc:litert_macros",
        "//litert/runtime:event",
    ],
)

cc_library(
    name = "litert_tensor_buffer_types",
    srcs = [],
    hdrs = ["litert_tensor_buffer_types.h"],
)

cc_library(
    name = "litert_gl_types",
    srcs = [],
    hdrs = ["litert_gl_types.h"],
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
    linkopts = select({
        "@org_tensorflow//tensorflow:android": [
            "-landroid",
        ],
        "//conditions:default": [],
    }) + gles_linkopts(),
    deps = [
        ":litert_common",
        ":litert_event",
        ":litert_gl_types",
        ":litert_logging",
        ":litert_model",
        ":litert_tensor_buffer_types",
        "@com_google_absl//absl/types:span",
        "//litert/cc:litert_expected",
        "//litert/runtime:tensor_buffer",
        "@opencl_headers",
    ] + gles_deps(),
)

cc_test(
    name = "litert_tensor_buffer_test",
    srcs = [
        "litert_tensor_buffer_test.cc",
    ],
    # require GPU to run OpenCL tests.
    tags = [
        "requires-gpu-nvidia",
    ],
    deps = [
        ":litert_common",
        ":litert_model",
        ":litert_tensor_buffer",
        ":litert_tensor_buffer_types",
        "@com_google_googletest//:gtest_main",
        "//litert/cc:litert_layout",
        "//litert/runtime:tensor_buffer",
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
        ":litert_environment_options",
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
    name = "litert_compilation_options",
    srcs = ["litert_compilation_options.cc"],
    hdrs = [
        "litert_compilation_options.h",
    ],
    deps = [
        ":litert_accelerator_compilation_options",
        ":litert_common",
        ":litert_logging",
        "//litert/cc:litert_macros",
        "//litert/runtime:compilation_options",
    ],
)

cc_test(
    name = "litert_compilation_options_test",
    srcs = ["litert_compilation_options_test.cc"],
    deps = [
        ":litert_accelerator_compilation_options",
        ":litert_common",
        ":litert_compilation_options",
        "@com_google_googletest//:gtest_main",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
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
        ":litert_compilation_options",
        ":litert_environment",
        ":litert_logging",
        ":litert_model",
        ":litert_tensor_buffer",
        "//litert/runtime:compiled_model",
        "//tflite/c:c_api_types",
    ],
)

cc_test(
    name = "litert_compiled_model_test",
    srcs = [
        "litert_compiled_model_test.cc",
    ],
    data = [
        "//litert/test:testdata/simple_model.tflite",
    ],
    deps = [
        ":litert_common",
        ":litert_compilation_options",
        ":litert_compiled_model",
        ":litert_environment",
        ":litert_model",
        ":litert_tensor_buffer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//litert/test:common",
        "//litert/test:simple_model",
    ],
)

# The same test as `litert_compiled_model_test` but using the shared library `libLiteRtRuntimeCApi.so`.
cc_test(
    name = "litert_compiled_model_shared_lib_test",
    srcs = [
        "litert_compiled_model_test.cc",
    ],
    data = [
        "//litert/test:testdata/simple_model.tflite",
    ],
    deps = [
        ":litert_runtime_c_api_shared_lib",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//litert/test:common",
        "//litert/test:simple_model",
    ],
)

cc_library(
    name = "litert_accelerator",
    srcs = ["litert_accelerator.cc"],
    hdrs = ["litert_accelerator.h"],
    deps = [
        ":litert_common",
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
        ":litert_common",
        ":litert_environment",
        "@com_google_googletest//:gtest_main",
        "//litert/runtime:accelerator",
    ],
)

cc_library(
    name = "litert_accelerator_compilation_options",
    srcs = ["litert_accelerator_compilation_options.cc"],
    hdrs = ["litert_accelerator_compilation_options.h"],
    deps = [
        ":litert_common",
    ],
)

cc_test(
    name = "litert_accelerator_compilation_options_test",
    srcs = ["litert_accelerator_compilation_options_test.cc"],
    deps = [
        ":litert_accelerator_compilation_options",
        ":litert_common",
        "@com_google_googletest//:gtest_main",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core:version",
        "//litert/test:matchers",
    ],
)

cc_library(
    name = "litert_accelerator_registration",
    srcs = ["litert_accelerator_registration.cc"],
    hdrs = ["litert_accelerator_registration.h"],
    deps = [
        ":litert_accelerator",
        ":litert_accelerator_compilation_options",
        ":litert_common",
        "//litert/cc:litert_expected",
        "//litert/core:environment",
        "//litert/runtime:accelerator",
    ],
)

cc_test(
    name = "litert_accelerator_registration_test",
    srcs = ["litert_accelerator_registration_test.cc"],
    deps = [
        ":litert_accelerator",
        ":litert_accelerator_compilation_options",
        ":litert_accelerator_registration",
        ":litert_common",
        ":litert_environment",
        "@com_google_googletest//:gtest_main",
        "//litert/runtime:accelerator",
    ],
)

filegroup(
    name = "litert_model_srcs",
    srcs = ["litert_model.cc"],
    visibility = ["//litert/core/model:__pkg__"],
)

filegroup(
    name = "litert_model_hdrs",
    srcs = ["litert_model.h"],
    visibility = ["//litert/core/model:__pkg__"],
)

# Collection of all C API targets.
LITERT_C_API_COMMON_DEPS = [
    ":litert_accelerator",
    ":litert_accelerator_registration",
    ":litert_any",
    ":litert_common",
    ":litert_compiled_model",
    ":litert_compilation_options",
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
    }) + ["-Wl,-soname=libLiteRtRuntimeCApi.so"],
    deps = LITERT_C_API_COMMON_DEPS,
)

cc_library(
    name = "litert_dispatch_headers",
    hdrs = [
        ":litert_environment.h",
        ":litert_environment_options.h",
        ":litert_accelerator.h",
        ":litert_accelerator_compilation_options.h",
        ":litert_any.h",
        ":litert_common.h",
        ":litert_compiled_model.h",
        ":litert_compilation_options.h",
        ":litert_event.h",
        ":litert_event_type.h",
        ":litert_layout.h",
        ":litert_logging.h",
        ":litert_model.h",
        ":litert_op_code.h",
        ":litert_options.h",
        ":litert_tensor_buffer.h",
        ":litert_tensor_buffer_requirements.h",
        ":litert_tensor_buffer_types.h",
        ":litert_gl_types.h",
        # Needed for litert/c/litert_op_code.h
        "//tflite:builtin_ops.h",
        # Neeeded for litert/c/litert_model.h
        "//tflite/c:tensorflowlite_c_api_hdrs_filegroup",
        "//tflite/c:tensorflowlite_c_impl_hdrs_filegroup",
    ],  # Export all header files (.h) in this directory
    deps = [
        "@opencl_headers",
    ],
)

copy_file(
    name = "copy_litert_runtime_c_api_so",
    src = "//litert/c:litert_runtime_c_api_so",
    target = "libLiteRtRuntimeCApi.so",
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
    ],
    linkstatic = 1,
    deps = [
        # only depend on headers
        "@opencl_headers",
        "//litert/vendors/c:litert_dispatch_c_api",
    ] + gles_headers(),
)

# copybara:uncomment_begin(google-only)
# # Check that litert runtime doesn't depend on MLIR.
# check_dependencies(
#     of = [":litert_runtime_c_api_shared_lib"],
#     dont_match_regexp = "^//third_party/llvm/llvm-project/mlir",
# )
# copybara:uncomment_end

exports_files(srcs = glob(["litert_*.h"]))
