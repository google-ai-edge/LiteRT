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
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

cc_test(
    name = "jit_compilation_qualcomm_test",
    srcs = ["jit_compilation_qualcomm_test.cc"],
    data = [
        "//litert/test:simple_model",
        "//litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so",
        "//litert/vendors/qualcomm/dispatch:dispatch_api_so",
    ],
    linkopts = select({
        "//third_party/tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        "//testing/base/public:gunit_main",
        "//third_party/absl/log",
        "//third_party/absl/log:absl_log",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
        "//litert/c:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_model",
        "//litert/cc:litert_tensor_buffer",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model_npu",
        "//third_party/tensorflow/lite:framework",
        "//third_party/tensorflow/lite/c:c_api_opaque",
        "//third_party/tensorflow/lite/c:common",
        "//third_party/tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_test(
    name = "jit_compilation_mediatek_test",
    srcs = ["jit_compilation_mediatek_test.cc"],
    data = [
        "//litert/test:simple_model",
        "//litert/vendors/mediatek/compiler:compiler_plugin_so",
        "//litert/vendors/mediatek/dispatch:dispatch_api_so",
    ],
    linkopts = select({
        "//third_party/tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    tags = [
        "no_oss",
        "nobuilder",
        "notap",
    ],
    deps = [
        "//testing/base/public:gunit_main",
        "//third_party/absl/log",
        "//third_party/absl/log:absl_log",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
        "//litert/c:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_model",
        "//litert/cc:litert_tensor_buffer",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model_npu",
        "//third_party/tensorflow/lite:framework",
        "//third_party/tensorflow/lite/c:c_api_opaque",
        "//third_party/tensorflow/lite/c:common",
        "//third_party/tensorflow/lite/kernels:builtin_ops",
    ],
)
