"""Default (OSS) build versions of Python pytype rules."""

load("@xla//third_party/rules_python/python:defs.bzl", "py_binary", "py_library", "py_test")

# Placeholder to use until bazel supports pytype_library.
def pytype_library(name, pytype_deps = [], pytype_srcs = [], **kwargs):
    _ = (pytype_deps, pytype_srcs)  # @unused
    py_library(name = name, **kwargs)

# Placeholder to use until bazel supports pytype_strict_binary.
def pytype_strict_binary(name, **kwargs):
    py_binary(name = name, **kwargs)

# Placeholder to use until bazel supports pytype_strict_library.
def pytype_strict_library(name, **kwargs):
    py_library(name = name, **kwargs)

# Placeholder to use until bazel supports pytype_strict_contrib_test.
def pytype_strict_contrib_test(name, **kwargs):
    py_test(name = name, **kwargs)
