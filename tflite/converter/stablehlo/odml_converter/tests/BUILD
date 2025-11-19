load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tflite/converter:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//tflite/converter/stablehlo/odml_converter:__subpackages__"],
)

glob_lit_tests(
    name = "filecheck_tests",
    data = [":test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = [
        "mlir",
    ],
)

filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tflite/converter/stablehlo/odml_converter:odml-converter",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//mlir:run_lit.sh",
    ],
)
