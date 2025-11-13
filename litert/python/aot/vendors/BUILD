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

load("@org_tensorflow//tensorflow:pytype.default.bzl", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_python_internal_users"],
)

pytype_strict_contrib_test(
    name = "import_vendor_test",
    srcs = ["import_vendor_test.py"],
    deps = [
        ":import_vendor",
        "//litert/python/aot/core:types",
        "//litert/python/aot/vendors/example:example_backend",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_library(
    name = "import_vendor",
    srcs = ["import_vendor.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fallback_backend",
        "//litert/python/aot/core:types",
    ],
)

pytype_strict_library(
    name = "vendors",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//litert/python/aot/vendors/example:example_backend",  # buildcleaner: keep
        "//litert/python/aot/vendors/google_tensor:google_tensor_backend",  # buildcleaner: keep
        "//litert/python/aot/vendors/mediatek:mediatek_backend",  # buildcleaner: keep
        "//litert/python/aot/vendors/qualcomm:qualcomm_backend",  # buildcleaner: keep
    ],
)

pytype_strict_library(
    name = "fallback_backend",
    srcs = ["fallback_backend.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//litert/python/aot/core:components",
        "//litert/python/aot/core:types",
    ],
)
