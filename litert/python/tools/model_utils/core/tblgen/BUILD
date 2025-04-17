load("//third_party/antlr/v4:build_defs.bzl", "antlr4_py_combined")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/python/tools/model_utils:__subpackages__"],
    licenses = ["notice"],
)

antlr4_py_combined(
    # Build name doesn't matter, just generates: {TableGenParser,TableGenLexer}.py
    name = "grammar",
    src = "//litert/python/tools/model_utils/core/tblgen:TableGen.g4",
    visitor = True,
)

py_library(
    name = "tblgen",
    srcs = [
        "__init__.py",
    ],
    deps = [
        ":grammar",
        "//third_party/py/antlr4:py_runtime",
    ],
)
