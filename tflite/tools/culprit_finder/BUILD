package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "culprit_finder_utils",
    hdrs = ["culprit_finder_utils.h"],
    deps = [
        "//tflite:framework",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core:cc_api",
        "//tflite/kernels:builtin_ops",
        "//tflite/tools:command_line_flags",
        "//tflite/tools:logging",
        "//tflite/tools:tool_params",
    ],
)

cc_library(
    name = "model_metadata_lib",
    srcs = ["model_metadata_lib.cc"],
    hdrs = ["model_metadata_lib.h"],
    copts = ["-fexceptions"],
    features = [
        "-use_header_modules",
        "-parse_headers",
    ],  # Incompatible with -fexceptions.
    visibility = ["//visibility:public"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "//tflite:framework_stable",
        "//tflite/profiling:model_runtime_info",
        "//tflite/tools:logging",
    ],
)

cc_library(
    name = "tflite_input_manager",
    srcs = ["tflite_input_manager.cc"],
    hdrs = ["tflite_input_manager.h"],
    copts = ["-fexceptions"],
    features = [
        "-use_header_modules",
        "-parse_headers",
    ],  # Incompatible with -fexceptions.
    visibility = ["//visibility:public"],
    deps = [
        "//tflite:framework_stable",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/tools:logging",
        "//tflite/tools:utils",
    ],
)

cc_library(
    name = "interpreter_handler",
    srcs = ["interpreter_handler.cc"],
    hdrs = ["interpreter_handler.h"],
    copts = ["-fexceptions"],
    features = [
        "-use_header_modules",
        "-parse_headers",
    ],  # Incompatible with -fexceptions.
    visibility = ["//visibility:public"],
    deps = [
        ":tflite_input_manager",
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "//tflite:framework",
        "//tflite:framework_stable",
        "//tflite/c:c_api_types",
        "//tflite/core:model_builder",
        "//tflite/kernels:builtin_ops",
        "//tflite/tools:logging",
        "//tflite/tools:model_loader",
        "//tflite/tools/delegates:delegate_provider_hdr",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "culprit_finder_lib",
    srcs = ["culprit_finder_lib.cc"],
    hdrs = ["culprit_finder_lib.h"],
    copts = ["-fexceptions"],
    features = [
        "-use_header_modules",
        "-parse_headers",
    ],  # Incompatible with -fexceptions.
    visibility = ["//visibility:public"],
    deps = [
        ":culprit_finder_utils",
        ":interpreter_handler",
        ":model_metadata_lib",
        ":tflite_input_manager",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment_begin(google-only)
        # "@org_tensorflow//third_party/odml/infra/ml_drift_delegate:ml_drift_delegate_provider",
        # "//litert/cc:litert_expected",
        # "//litert/cc:litert_macros",
        # "//litert/cc:litert_tflite_error_status_builder",
        # "//tflite/profiling/proto:model_runtime_info_cc",
        # copybara:uncomment_end
        "//tflite:framework",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core:cc_api",
        "//tflite/kernels:builtin_ops",
        "//tflite/profiling:memory_usage_monitor",
        "//tflite/profiling:model_runtime_info",
        "//tflite/tools:command_line_flags",
        "//tflite/tools:logging",
        "//tflite/tools:model_loader",
        "//tflite/tools:tool_params",
        "//tflite/tools:utils",
        "//tflite/tools/delegates:default_execution_provider",
        "//tflite/tools/delegates:delegate_provider_hdr",
    ] + select({
        # Metal GPU delegate for iOS has its own setups for kernel tests, so
        # skipping linking w/ the gpu_delegate_provider.
        "@org_tensorflow//tensorflow:ios": [],
        "//conditions:default": [
            "//tflite/tools/delegates:gpu_delegate_provider",
        ],
    }),
)

cc_binary(
    name = "culprit_finder_main",
    srcs = ["culprit_finder_main.cc"],
    deps = [
        ":culprit_finder_lib",
        "//tflite/tools:logging",
    ],
)
