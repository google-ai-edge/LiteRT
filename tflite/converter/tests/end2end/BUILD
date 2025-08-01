load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tflite/converter:glob_lit_test.bzl", "glob_lit_tests")

# copybara:uncomment package(default_applicable_licenses = ["@org_tensorflow//tensorflow:license"])

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

glob_lit_tests(
    name = "all_tests",
    data = [
        ":quant_stats_files",
        ":test_utilities",
    ],
    driver = "@llvm-project//mlir:run_lit.sh",
    size_override = {
        "quant_stats.pbtxt": "medium",
    },
    tags_override = {
        "add.pbtxt": ["no_rocm"],
        "fake_quant_per_channel.pbtxt": ["no_rocm"],
    },
    test_file_exts = [
        "pbtxt",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tflite/converter:flatbuffer_to_string",
        "//tflite/converter:tf_tfl_translate",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)

# Bundle together all the quant stats files that are used by the tests.
filegroup(
    name = "quant_stats_files",
    srcs = glob(
        ["**/*.stats"],
    ),
)
