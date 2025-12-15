# Copyright 2025 Google LLC.
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
load("@org_tensorflow//tensorflow:pytype.default.bzl", "pytype_strict_contrib_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

pytype_strict_contrib_test(
    name = "match_test",
    srcs = ["match_test.py"],
    tags = ["no_oss"],
    deps = [
        "//litert/python/tools/model_utils",
        "//litert/python/tools/model_utils:testing",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//third_party/py/numpy:headers",
        "@xdsl",
    ],
)

pytype_strict_contrib_test(
    name = "signature_builder_test",
    srcs = ["signature_builder_test.py"],
    tags = ["no_oss"],
    deps = [
        # Dependency placeholder: gfile,
        "@absl_py//absl/testing:absltest",
        "//litert/python/tools/model_utils",
        "//litert/python/tools/model_utils:testing",
        "@xdsl",
    ],
)

pytype_strict_contrib_test(
    name = "numpy_to_elements_attr_test",
    srcs = ["numpy_to_elements_attr_test.py"],
    tags = ["no_oss"],
    deps = [
        "//litert/python/tools/model_utils",
        "//litert/python/tools/model_utils:testing",
        "@absl_py//absl/testing:absltest",
        "@ml_dtypes_py//ml_dtypes",
        "@org_tensorflow//third_party/py/numpy:headers",
    ],
)
