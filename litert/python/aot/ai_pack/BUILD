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
    default_visibility = ["//litert:litert_python_users"],
)

pytype_strict_library(
    name = "export_lib",
    srcs = ["export_lib.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//litert/python/aot/core:common",
        "//litert/python/aot/core:types",
        "//litert/python/aot/vendors:fallback_backend",
        "//litert/python/aot/vendors/google_tensor:target",
        "//litert/python/aot/vendors/mediatek:mediatek_backend",
        "//litert/python/aot/vendors/mediatek:target",
        "//litert/python/aot/vendors/qualcomm:qualcomm_backend",
        "//litert/python/aot/vendors/qualcomm:target",
    ],
)

pytype_strict_contrib_test(
    name = "export_lib_test",
    srcs = ["export_lib_test.py"],
    deps = [
        ":export_lib",
        "//litert/python/aot/core:types",
        "//litert/python/aot/vendors:fallback_backend",
        "//litert/python/aot/vendors/google_tensor:target",
        "@absl_py//absl/testing:absltest",
    ],
)
