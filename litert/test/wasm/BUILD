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

load("@emsdk//emscripten_toolchain:wasm_rules.bzl", "wasm_cc_binary")  # copybara:comment
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

# copybara:uncomment_begin(google-only)
# load("//third_party/bazel_rules/rules_wasm/wasm:defs.bzl", "wasm_cc_binary")
#
# copybara:uncomment_end
load("//litert/build_common:selective_ops.bzl", "litert_selected_op_resolver")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

# For local validation, place a model here and select its label using this flag.
# Downloaded application models are not checked into LiteRT.
exports_files(glob(["*.tflite"]))

label_flag(
    name = "model",
    build_setting_default = "//litert/test:testdata/simple_model.tflite",
)

litert_selected_op_resolver(
    name = "model_op_resolver",
    models = [":model"],
)

_LINKOPTS = [
    "--no-entry",
    "-Oz",
    "-sMODULARIZE=1",
    "-sEXPORTED_RUNTIME_METHODS=['HEAPU8','HEAP32','HEAPF32']",
    "-sEXPORT_NAME=createByteModelModule",
    "-sENVIRONMENT=web,worker,node",
    "-sALLOW_MEMORY_GROWTH=1",
    "-sFILESYSTEM=0",
    "-sAUTO_NATIVE_LIBRARIES=0",
    "-sASSERTIONS=0",
    # Some native dependencies add -pthread to linkopts. Keep this harness
    # usable without SharedArrayBuffer or cross-origin isolation.
    "-sUSE_PTHREADS=0",
    "-sMALLOC=emmalloc",
    "-sEXPORTED_FUNCTIONS=['_malloc','_free','_load_model','_get_input_length','_predict','_get_output_count','_get_output_length','_get_output']",
]

cc_binary(
    name = "compiled_model",
    srcs = ["byte_model_runner.cc"],
    linkopts = _LINKOPTS,
    linkstatic = True,
    # Only linkable through the wasm_cc_binary targets below (emscripten
    # linkopts) with the build flags from the README. Tricorder Builder and TAP
    # ignore "manual" but honor "nobuilder" and "notap".
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_tensor_buffer",
    ],
)

cc_binary(
    name = "interpreter",
    srcs = ["byte_model_runner.cc"],
    copts = ["-DLITERT_WASM_TFLITE_BASELINE"],
    linkopts = _LINKOPTS,
    linkstatic = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        "//litert/build_common:op_resolver",
        "//litert/runtime:op_resolver_header",
        "//tflite:framework",
        "//tflite:model_builder",
    ],
)

wasm_cc_binary(
    name = "compiled_model_wasm",
    cc_target = ":compiled_model",
    # copybara:uncomment include_data_dependencies = False,
    # copybara:uncomment simd = "simd128",
    simd = True,  # copybara:comment
    tags = [
        "manual",
        "notap",
    ],
    threads = "off",
)

wasm_cc_binary(
    name = "interpreter_wasm",
    cc_target = ":interpreter",
    # copybara:uncomment include_data_dependencies = False,
    # copybara:uncomment simd = "simd128",
    simd = True,  # copybara:comment
    tags = [
        "manual",
        "notap",
    ],
    threads = "off",
)
