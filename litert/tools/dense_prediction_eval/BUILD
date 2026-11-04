# Copyright 2026 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_test.bzl", "py_test")

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    deps = ["@org_tensorflow//third_party/py/numpy"],
)

py_test(
    name = "metrics_test",
    size = "small",
    srcs = ["metrics_test.py"],
    deps = [":metrics", "@org_tensorflow//third_party/py/numpy"],
)

py_library(
    name = "evaluation",
    srcs = ["evaluate.py", "run.py"],
    deps = [":metrics", "@org_tensorflow//third_party/py/numpy"],
)

py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    deps = [":metrics", "@org_tensorflow//third_party/py/numpy"],
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    deps = [":evaluation"],
)

py_test(
    name = "evaluate_test",
    size = "small",
    srcs = ["evaluate_test.py"],
    deps = [":evaluation", "@org_tensorflow//third_party/py/numpy"],
)

py_test(
    name = "runner_test",
    size = "small",
    srcs = ["runner_test.py"],
    data = [
        "//litert/test:testdata/simple_add_op.tflite",
        "//litert/tools:run_model",
    ],
    deps = [
        "@org_tensorflow//third_party/py/numpy",
        "@rules_python//python/runfiles",
    ],
)
