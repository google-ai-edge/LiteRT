# Copyright 2024 Google LLC.
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

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors/mediatek/compiler:__subpackages__"],
)

cc_library(
    name = "operand_map",
    srcs = ["operand_map.cc"],
    hdrs = ["operand_map.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "//litert/c:litert_common",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "add_op_legalization",
    srcs = ["add_op_legalization.cc"],
    hdrs = ["add_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "operand_map",
        "//litert/c:litert_common",
        "//litert/c:litert_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)
