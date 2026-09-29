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

"""Minimal stand-in for @org_tensorflow//tensorflow/core/platform:build_config_root.bzl."""

load("@rules_ml_toolchain//py/rules_pywrap:pywrap.default.bzl", "use_pywrap_rules")

def if_pywrap(if_true = [], if_false = []):
    return if_true if use_pywrap_rules() else if_false

def tf_gpu_tests_tags():
    return ["requires-gpu", "gpu"]

def tf_cuda_tests_tags():
    return tf_gpu_tests_tags()
