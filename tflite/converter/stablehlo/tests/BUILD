load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tflite/converter:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:private",
    ],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "//tflite/converter/stablehlo:run_lit.sh",
    size_override = {
        "legalize-skip-quantization-ops.mlir": "medium",
    },
    test_file_exts = [
        "mlir",
        "cc",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tflite/converter:flatbuffer_translate",
        "//tflite/converter:tf_tfl_translate",
        "//tflite/converter/stablehlo:odml-to-stablehlo-opt",
        "//tflite/converter/stablehlo:odml_to_stablehlo",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//mlir:run_lit.sh",
    ],
)
