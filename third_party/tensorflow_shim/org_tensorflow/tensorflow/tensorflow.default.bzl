# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal replacement for TensorFlow's tensorflow/tensorflow.default.bzl."""

load(
    "@rules_ml_toolchain//py/rules_pywrap:pywrap.default.bzl",
    _pywrap_binaries = "pywrap_binaries",
    _pywrap_library = "pywrap_library",
)
load(
    "//tensorflow:tensorflow.bzl",
    _get_compatible_with_portable = "get_compatible_with_portable",
    _if_portable = "if_portable",
    _py_test = "py_test",
    _pybind_extension = "pybind_extension",
    _requires_tensorflow = "tf_custom_op_library",
)

filegroup = native.filegroup
get_compatible_with_portable = _get_compatible_with_portable
if_portable = _if_portable
pybind_extension = _pybind_extension
pywrap_binaries = _pywrap_binaries
pywrap_library = _pywrap_library
tf_python_pybind_extension = _pybind_extension
tf_py_strict_test = _py_test

def tf_custom_op_py_strict_library(name, **kwargs):
    _requires_tensorflow(name, **kwargs)
