package(default_visibility = ["//visibility:public"])

cc_library(
    name = "fuzzing_support",
    srcs = [
        "fuzzer_quota_allocator.cc",
        "one_op_fuzz_model.cc",
    ],
    hdrs = [
        "fuzzer_quota_allocator.h",
        "fuzzing_util.h",
        "one_op_fuzz_model.h",
    ],
    deps = [
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/schema:schema_fbs",
        "@flatbuffers",
    ],
)

cc_test(
    name = "pad_fuzz_test",
    srcs = ["pad_fuzz_test.cc"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "mirror_pad_fuzz_test",
    srcs = ["mirror_pad_fuzz_test.cc"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "stablehlo_pad_fuzz_test",
    srcs = ["stablehlo_pad_fuzz_test.cc"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "convolution_fuzz_test",
    srcs = ["convolution_fuzz_test.cc"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/kernels:builtin_op_kernels",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "pad_xnnpack_fuzz_test",
    srcs = ["pad_fuzz_test.cc"],
    defines = ["TFLITE_PAD_FUZZ_ENABLE_XNNPACK"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/delegates/xnnpack:xnnpack_delegate",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "arg_min_max_fuzz_test",
    srcs = ["arg_min_max_fuzz_test.cc"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "cumsum_fuzz_test",
    srcs = ["cumsum_fuzz_test.cc"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "reduce_fuzz_test",
    srcs = ["reduce_fuzz_test.cc"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "reduce_xnnpack_fuzz_test",
    srcs = ["reduce_fuzz_test.cc"],
    defines = ["TFLITE_REDUCE_FUZZ_ENABLE_XNNPACK"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/delegates/xnnpack:xnnpack_delegate",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "segment_fuzz_test",
    srcs = ["segment_fuzz_test.cc"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "topk_v2_fuzz_test",
    srcs = ["topk_v2_fuzz_test.cc"],
    deps = [
        ":fuzzing_support",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)
