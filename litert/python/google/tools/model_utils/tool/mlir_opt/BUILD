load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//litert/python/google/tools/model_utils:__subpackages__",
        "//litert/python/google/tools/model_utils:model_utils_users",
    ],
    licenses = ["notice"],
)

py_library(
    name = "mlir_opt",
    srcs = [
        "__init__.py",
        "_opt_lib.py",
    ],
    deps = [
        "//litert/python/google/tools/model_utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "//third_party/py/mlir",
        "//third_party/py/mlir:ir",
    ],
)

py_binary(
    name = "bin",
    srcs = [
        "bin.py",
    ],
    deps = [
        ":mlir_opt",
        "//litert/python/google/tools/model_utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "//third_party/py/mlir",
        "//third_party/py/mlir:ir",
        "//third_party/py/xdsl",
    ],
)
