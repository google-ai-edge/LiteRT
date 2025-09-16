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

# copybara:uncomment load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# copybara:uncomment_begin(google-only)
# py_strict_test(
#     name = "litert_wrapper_test",
#     srcs = ["litert_wrapper_test.py"],
#     data = [
#         "testdata/simple_model_float.tflite",
#         "testdata/simple_model_int.tflite",
#     ],
#     # Static linking is required because this loads a cc_binary as a shared
#     # library, which would otherwise create ODR violations.
#     # copybara:uncomment linking_mode = "static",
#     tags = [
#         "no_oss",  # TODO(b/190842754): Enable test in OSS.
#     ],
#     deps = [
#         "//litert/python/litert_wrapper/compiled_model_wrapper:compiled_model",
#         "//litert/python/litert_wrapper/tensor_buffer_wrapper:tensor_buffer",
#         "@com_google_protobuf//:protobuf_python:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
#         "@org_tensorflow//third_party/py/numpy:headers",
#         "//third_party/py/tensorflow",
#         "@org_tensorflow//tensorflow/python/platform:resource_loader",
#     ],
# )
#
# OUT_DIR = "$(RULEDIR)"
#
# CONVERTER = "//tflite/converter:tf_tfl_translate"
#
# CMD = """
# for mlir_file in $(SRCS); do
#     $(location {converter}) --input-mlir $$mlir_file --o={out_dir}/testdata/$$(basename $$mlir_file .mlir).tflite
# done
# """.format(
#     converter = CONVERTER,
#     out_dir = OUT_DIR,
# )
#
# genrule(
#     name = "mlir_test_data",
#     srcs = glob(["testdata/*.mlir"]),
#     outs = [s.removesuffix(".mlir") + ".tflite" for s in glob(["testdata/*.mlir"])],
#     cmd = CMD,
#     tools = [CONVERTER],
# )
# copybara:uncomment_end
