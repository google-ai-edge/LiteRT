load("@rules_python//python:py_library.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

# `tflite/python/interpreter.py` imports `tf_export` only when it is part of
# the TensorFlow pip package. The LiteRT wheel does not include TensorFlow, so
# this target is intentionally empty.
py_library(name = "tf_export")
