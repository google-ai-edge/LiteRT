load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_binary")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tflite/converter:glob_lit_test.bzl", "glob_lit_tests")

# copybara:uncomment package(default_applicable_licenses = ["@org_tensorflow//tensorflow:license"])

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

glob_lit_tests(
    name = "all_tests",
    data = [
        ":debug_info_files",
        ":test_utilities",
    ],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = [
        "pbtxt",
        #        "py", TODO(b/150304798)
    ],
)

# Bundle together all the debug info files that are used by the tests.
filegroup(
    name = "debug_info_files",
    srcs = glob(
        ["**/*.debug"],
    ),
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":concrete_function_error",
        ":saved_model_error",
        "//tflite/converter:flatbuffer_to_string",
        "//tflite/converter:tf_tfl_translate",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)

py_strict_binary(
    name = "saved_model_error",
    srcs = ["saved_model_error.py"],
    main = "saved_model_error.py",
    deps = [
        "@absl_py//absl:app",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "concrete_function_error",
    srcs = ["concrete_function_error.py"],
    main = "concrete_function_error.py",
    deps = [
        "@absl_py//absl:app",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
