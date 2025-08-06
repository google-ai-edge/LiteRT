load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_features_nolayering_check_if_ios")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tflite:special_rules.bzl", "nonportable_visibility_allowlist")

# Kernel for custom parse_example
package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "parse_example",
    srcs = [
        "example_proto_fast_parsing.cc",
        "parse_example.cc",
    ],
    hdrs = [
        "example_proto_fast_parsing.h",
        "parse_example.h",
    ],
    compatible_with = get_compatible_with_portable(),
    features = tf_features_nolayering_check_if_ios(),
    deps = [
        "//tflite:framework",
        "//tflite:string_util",
        "//tflite/core/c:common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:tensor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/core/platform:hash",
    ] + select({
        "@org_tensorflow//tensorflow:android": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "@org_tensorflow//tensorflow:ios": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:core_cpu",
            "@org_tensorflow//tensorflow/core:feature_util",
            "@org_tensorflow//tensorflow/core:framework",
            "@org_tensorflow//tensorflow/core:framework_internal",
            "@org_tensorflow//tensorflow/core:lib",
            "@org_tensorflow//tensorflow/core:lib_internal",
            "@org_tensorflow//tensorflow/core:protos_all_cc",
        ],
    }),
)

tf_cc_test(
    name = "parse_example_test",
    srcs = ["parse_example_test.cc"],
    features = tf_features_nolayering_check_if_ios(),
    deps = [
        ":parse_example",
        "//tflite:framework",
        "//tflite:string_util",
        "//tflite/core:framework",
        "//tflite/core/api:op_resolver",
        "//tflite/core/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/kernels:test_main",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@flatbuffers",
    ] + select({
        "@org_tensorflow//tensorflow:android": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "@org_tensorflow//tensorflow:ios": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:protos_all_cc",
            "@org_tensorflow//tensorflow/core/example:feature_util",
            "@org_tensorflow//tensorflow/core/platform:protobuf",
            "@org_tensorflow//tensorflow/core/platform:tstring",
        ],
    }),
)

# Same as "parse_example" above, but doesn't use the "select" statements
# to use "portable_tensorflow_lib_lite" for portable builds.
cc_library(
    name = "nonportable_parse_example",
    srcs = [
        "example_proto_fast_parsing.cc",
        "parse_example.cc",
    ],
    hdrs = [
        "example_proto_fast_parsing.h",
        "parse_example.h",
    ],
    visibility = nonportable_visibility_allowlist(),
    deps = [
        "//tflite:framework",
        "//tflite:string_util",
        "//tflite/core/c:common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:tensor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:feature_util",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:framework_internal",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:lib_internal",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "nonportable_parse_example_test",
    srcs = ["parse_example_test.cc"],
    deps = [
        ":nonportable_parse_example",
        "//tflite:framework",
        "//tflite:string_util",
        "//tflite/c:common",
        "//tflite/core:framework",
        "//tflite/core/api:op_resolver",
        "//tflite/core/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/kernels:test_main",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/example:feature_util",
        "@org_tensorflow//tensorflow/core/platform:protobuf",
        "@org_tensorflow//tensorflow/core/platform:tstring",
    ],
)
