# Copyright 2017 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dummy test runner rule. Does not actually run tests."""

load(
    "//apple:providers.bzl",
    "apple_provider",
)

def _dummy_test_runner_impl(ctx):
    ctx.actions.expand_template(
        template = ctx.file._test_template,
        output = ctx.outputs.test_runner_template,
        substitutions = {},
    )

    return [
        apple_provider.make_apple_test_runner_info(
            test_runner_template = ctx.outputs.test_runner_template,
        ),
        DefaultInfo(
            runfiles = ctx.runfiles(files = []),
        ),
    ]

dummy_test_runner = rule(
    _dummy_test_runner_impl,
    attrs = {
        "_test_template": attr.label(
            default = Label("//test/testdata/rules:dummy_test_runner.template"),
            allow_single_file = True,
        ),
    },
    outputs = {
        "test_runner_template": "%{name}.sh",
    },
    fragments = ["apple", "objc"],
)
