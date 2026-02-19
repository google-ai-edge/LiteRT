# Copyright 2018 The Bazel Authors. All rights reserved.
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

"""iOS test runner rule."""

load(
    "//apple:providers.bzl",
    "AppleDeviceTestRunnerInfo",
    "apple_provider",
)

def _get_template_substitutions(
        *,
        device_type,
        os_version,
        simulator_creator,
        testrunner,
        pre_action_binary,
        post_action_binary,
        post_action_determines_exit_code):
    """Returns the template substitutions for this runner."""
    subs = {
        "device_type": device_type,
        "os_version": os_version,
        "simulator_creator": simulator_creator,
        "testrunner_binary": testrunner,
        "pre_action_binary": pre_action_binary,
        "post_action_binary": post_action_binary,
        "post_action_determines_exit_code": post_action_determines_exit_code,
    }
    return {"%(" + k + ")s": subs[k] for k in subs}

def _get_execution_environment(*, xcode_config):
    """Returns environment variables the test runner requires"""
    execution_environment = {}
    xcode_version = str(xcode_config.xcode_version())
    if xcode_version:
        execution_environment["XCODE_VERSION_OVERRIDE"] = xcode_version

    return execution_environment

def _ios_test_runner_impl(ctx):
    """Implementation for the ios_test_runner rule."""

    os_version = str(ctx.attr.os_version or ctx.fragments.objc.ios_simulator_version or "")
    device_type = ctx.attr.device_type or ctx.fragments.objc.ios_simulator_device or ""

    runfiles = ctx.attr._simulator_creator[DefaultInfo].default_runfiles
    runfiles = runfiles.merge(ctx.attr._testrunner[DefaultInfo].default_runfiles)

    default_action_binary = "/usr/bin/true"

    pre_action_binary = default_action_binary
    post_action_binary = default_action_binary

    if ctx.executable.pre_action:
        pre_action_binary = ctx.executable.pre_action.short_path
        runfiles = runfiles.merge(ctx.attr.pre_action[DefaultInfo].default_runfiles)

    post_action_determines_exit_code = False
    if ctx.executable.post_action:
        post_action_binary = ctx.executable.post_action.short_path
        post_action_determines_exit_code = ctx.attr.post_action_determines_exit_code
        runfiles = runfiles.merge(ctx.attr.post_action[DefaultInfo].default_runfiles)

    ctx.actions.expand_template(
        template = ctx.file._test_template,
        output = ctx.outputs.test_runner_template,
        substitutions = _get_template_substitutions(
            device_type = device_type,
            os_version = os_version,
            simulator_creator = ctx.executable._simulator_creator.short_path,
            testrunner = ctx.executable._testrunner.short_path,
            pre_action_binary = pre_action_binary,
            post_action_binary = post_action_binary,
            post_action_determines_exit_code = "true" if post_action_determines_exit_code else "false",
        ),
    )
    return [
        apple_provider.make_apple_test_runner_info(
            execution_requirements = ctx.attr.execution_requirements,
            execution_environment = _get_execution_environment(
                xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
            ),
            test_environment = ctx.attr.test_environment,
            test_runner_template = ctx.outputs.test_runner_template,
        ),
        AppleDeviceTestRunnerInfo(
            device_type = device_type,
            os_version = os_version,
        ),
        DefaultInfo(runfiles = runfiles),
    ]

ios_test_runner = rule(
    _ios_test_runner_impl,
    attrs = {
        "device_type": attr.string(
            default = "",
            doc = """
The device type of the iOS simulator to run test. The supported types correspond
to the output of `xcrun simctl list devicetypes`. E.g., iPhone 6, iPad Air.
By default, it is the latest supported iPhone type.'
""",
        ),
        "execution_requirements": attr.string_dict(
            allow_empty = False,
            default = {"requires-darwin": ""},
            doc = """
Dictionary of strings to strings which specifies the execution requirements for
the runner. In most common cases, this should not be used.
""",
        ),
        "os_version": attr.string(
            default = "",
            doc = """
The os version of the iOS simulator to run test. The supported os versions
correspond to the output of `xcrun simctl list runtimes`. ' 'E.g., 11.2, 9.3.
By default, it is the latest supported version of the device type.'
""",
        ),
        "test_environment": attr.string_dict(
            doc = """
Optional dictionary with the environment variables that are to be propagated
into the XCTest invocation.
""",
        ),
        "pre_action": attr.label(
            executable = True,
            cfg = "exec",
            doc = """
A binary to run prior to test execution. Runs after simulator creation. Sets any environment variables available to the test runner.
""",
        ),
        "post_action": attr.label(
            executable = True,
            cfg = "exec",
            doc = """
A binary to run following test execution. Runs after testing but before test result handling and coverage processing. Sets the `$TEST_EXIT_CODE` environment variable, in addition to any other variables available to the test runner.
""",
        ),
        "post_action_determines_exit_code": attr.bool(
            default = False,
            doc = """
When true, the exit code of the test run will be set to the exit code of the post action. This is useful for tests that need to fail the test run based on their own criteria.
""",
        ),
        "_test_template": attr.label(
            default = Label(
                "//apple/testing/default_runner:ios_test_runner.template.sh",
            ),
            allow_single_file = True,
        ),
        "_testrunner": attr.label(
            default = Label(
                "@xctestrunner//:ios_test_runner",
            ),
            executable = True,
            cfg = "exec",
            doc = """
It is the rule that needs to provide the AppleTestRunnerInfo provider. This
dependency is the test runner binary.
""",
        ),
        "_simulator_creator": attr.label(
            default = Label(
                "//apple/testing/default_runner:simulator_creator",
            ),
            executable = True,
            cfg = "exec",
        ),
        "_xcode_config": attr.label(
            default = configuration_field(
                fragment = "apple",
                name = "xcode_config_label",
            ),
        ),
    },
    outputs = {
        "test_runner_template": "%{name}.sh",
    },
    fragments = ["apple", "objc"],
    doc = """
Rule to identify an iOS runner that runs tests for iOS.

The runner will create a new simulator according to the given arguments to run
tests.

Outputs:
  AppleTestRunnerInfo:
    test_runner_template: Template file that contains the specific mechanism
        with which the tests will be performed.
    execution_requirements: Dictionary that represents the specific hardware
        requirements for this test.
  Runfiles:
    files: The files needed during runtime for the test to be performed.
""",
)
