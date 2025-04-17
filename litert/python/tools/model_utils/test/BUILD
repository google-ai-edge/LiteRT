load("@org_tensorflow//tensorflow:pytype.default.bzl", "pytype_strict_contrib_test")
load("//litert/python/tools/model_utils:testing.bzl", "model_utils_filecheck_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

model_utils_filecheck_tests(
    name = "filecheck_tests",
    srcs = glob(["*.filecheck.py"]),
    deps = [
        "//litert/python/tools/model_utils",
        "@absl_py//absl:app",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
    ],
)

pytype_strict_contrib_test(
    name = "match_test",
    srcs = ["match_test.py"],
    deps = [
        "@absl_py//absl/testing:absltest",
        "//litert/python/tools/model_utils",
        "//third_party/py/numpy",
        "//third_party/py/xdsl",
    ],
)
