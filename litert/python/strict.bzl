"""Default (OSS) build versions of Python strict rules."""

load("@xla//third_party/rules_python/python:defs.bzl", "py_binary", "py_library", "py_test")

# Placeholder to use until bazel supports py_strict_binary.
def py_strict_binary(name, **kwargs):
    py_binary(name = name, **kwargs)

# Placeholder to use until bazel supports py_strict_library.
def py_strict_library(name, **kwargs):
    py_library(name = name, **kwargs)

# Placeholder to use until bazel supports py_strict_test.
def py_strict_test(name, **kwargs):
    py_test(name = name, **kwargs)
