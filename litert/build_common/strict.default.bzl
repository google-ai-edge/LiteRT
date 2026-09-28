"""Default (OSS) build versions of Python strict rules."""

load(
    "@rules_python//python:defs.bzl",
    _py_binary = "py_binary",
    _py_library = "py_library",
    _py_test = "py_test",
)

# Placeholder to use until bazel supports py_strict_binary.
def py_strict_binary(name, **kwargs):
    _py_binary(name = name, **kwargs)

# Placeholder to use until bazel supports py_strict_library.
def py_strict_library(name, **kwargs):
    _py_library(name = name, **kwargs)

# Placeholder to use until bazel supports py_strict_test.
def py_strict_test(name, **kwargs):
    _py_test(name = name, **kwargs)
