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
        tool_args = []):
    """Generates a test that runs a binary tool.

    Args:
      name: The name of the test.
      tool: The binary tool to run.
      data: Data dependencies of the tool.
      tool_args: Arguments to pass to the tool.
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
        tags = [
            "no-remote-exec",
            "notap",
        ],
        data = [tool] + data,
    )
