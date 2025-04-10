load("@org_tensorflow//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//litert/python/google/tools/model_utils:testing.bzl", "model_utils_filecheck_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

model_utils_filecheck_tests(
    name = "all_tests",
    srcs = glob(["*.filecheck.py"]),
    deps = [
        ":passes",  # buildcleaner: keep
        "@absl_py//absl:app",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
    ],
)

pytype_strict_library(
    name = "passes",
    srcs = ["passes.py"],
    visibility = ["//litert:__subpackages__"],
    deps = ["//litert/python/google/tools/model_utils"],
)
