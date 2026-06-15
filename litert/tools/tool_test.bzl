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

"""Simple macro to generate a test that runs a binary tool."""

load("@rules_shell//shell:sh_test.bzl", "sh_test")

def runfile_path(label):
    return "../$(rlocationpath {})".format(label)

def tool_test(
        name,
        tool,
        data = [],
        tool_args = [],
        tags = [
            "no-remote-exec",
            "notap",
        ]):
    """Generates a test that runs a binary tool.

    Args:
      name: The name of the test.
      tool: The binary tool to run.
      data: Data dependencies of the tool.
      tool_args: Arguments to pass to the tool.
      tags: Tags to pass to the test.
    """
    args = " ".join(tool_args)
    cmd = """
    #!/bin/bash
    {} {}
    exit $$?
    """.format(runfile_path(tool), args)
    exec_name = "_" + name
    native.genrule(
        name = exec_name,
        outs = [exec_name + ".sh"],
        cmd = "echo '{}' > $@".format(cmd),
        srcs = [tool] + data,
    )
    sh_test(
        name = name,
        srcs = [exec_name + ".sh"],
        tags = tags,
        data = [tool] + data,
    )

NUMERICS_CHECK_DEPS = [
    "@com_google_absl//absl/flags:flag",
    "@com_google_absl//absl/flags:parse",
    "@com_google_absl//absl/log:absl_log",
    "@com_google_absl//absl/strings:str_format",
    "@com_google_absl//absl/strings:string_view",
    "@com_google_absl//absl/types:span",
    "//litert/c:litert_common",
    "//litert/cc:litert_compiled_model",
    "//litert/cc:litert_element_type",
    "//litert/cc:litert_environment",
    "//litert/cc:litert_environment_options",
    "//litert/cc:litert_expected",
    "//litert/cc:litert_macros",
    "//litert/cc:litert_model",
    "//litert/cc:litert_options",
    "//litert/cc:litert_tensor_buffer",
    "//tflite/profiling:time",
]

RUN_MODEL_DEPS = [
    "//litert/tools:tensor_utils",
    "@com_google_absl//absl/flags:config",
    "@com_google_absl//absl/flags:flag",
    "@com_google_absl//absl/flags:parse",
    "@com_google_absl//absl/log:absl_log",
    "@com_google_absl//absl/random",
    "@com_google_absl//absl/strings",
    "@com_google_absl//absl/strings:string_view",
    "@com_google_absl//absl/time",
    "@com_google_absl//absl/types:span",
    "//litert/c:litert_common",
    "//litert/c/options:litert_cpu_options",
    "//litert/cc:litert_common",
    "//litert/cc:litert_compiled_model",
    "//litert/cc:litert_element_type",
    "//litert/cc:litert_environment",
    "//litert/cc:litert_environment_options",
    "//litert/cc:litert_expected",
    "//litert/cc:litert_macros",
    "//litert/cc:litert_model",
    "//litert/cc:litert_options",
    "//litert/cc:litert_tensor_buffer",
    "//litert/cc/internal:scoped_file",
    "//litert/cc/internal:scoped_weight_source",
    "//litert/cc/options:litert_gpu_options",
    "//litert/tools/flags:options_parser_registry",
    "//litert/tools/flags/vendors:google_tensor_flags",
    "//litert/tools/flags/vendors:intel_openvino_flags",
    "//litert/tools/flags/vendors:mediatek_flags",
    "//litert/tools/flags/vendors:qualcomm_flags",
    "//litert/tools/flags/vendors:samsung_flags",
    "//tflite/profiling:time",
]
