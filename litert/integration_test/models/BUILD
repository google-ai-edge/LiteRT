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

load("//litert/build_common:tfl_model_gen.bzl", "tfl_model_gen")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/integration_test:__subpackages__"],
)

filegroup(
    name = "classic_ml",
    srcs = glob(["classic_ml/*.tflite"]),
)

tfl_model_gen(
    name = "single_op",
    srcs = glob(["single_op/*.mlir"]),
    subdir = "single_op",
)

filegroup(
    name = "pre_compiled",
    srcs = glob(["pre_compiled/*.tflite"]),
)
