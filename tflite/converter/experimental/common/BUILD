load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

# copybara:uncomment package(default_applicable_licenses = ["@org_tensorflow//tensorflow:license"])

cc_library(
    name = "outline_operations",
    srcs = ["outline_operations.cc"],
    hdrs = ["outline_operations.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "//tflite/converter:tensorflow_lite",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:cluster_util",
    ],
)
