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

"""Sample Apple test runner rule."""

load(
    "@build_bazel_apple_support//lib:xcode_support.bzl",
    "xcode_support",
)
load(
    "//apple:providers.bzl",
    "apple_provider",
)

def _get_xctestrun_template_substitutions(xcode_config):
    """Returns the template substitutions for the xctestrun template."""

    # Xcode version is only set when Xcode is installed. Xcode tool
    # chain can be available both on macOS and Linux.
    if not xcode_config.xcode_version():
        return {
            "%(xctestrun_insert_libraries)s": "",
        }

    # Xcode 10 introduced new dylibs that are required when running unit tests, so we need to
    # provide different values in the xctestrun file that describes the test.
    # TODO(kaipi): Revisit this when Nitro has support for macOS. Nitro should be the one detecting
    #              Xcode version and configuring it appropriately.
    if xcode_support.is_xcode_at_least_version(xcode_config, "10.0"):
        xctestrun_insert_libraries = [
            "__PLATFORMS__/MacOSX.platform/Developer/usr/lib/libXCTestBundleInject.dylib",
            "__DEVELOPERUSRLIB__/libMainThreadChecker.dylib",
        ]
    else:
        xctestrun_insert_libraries = [
            "__PLATFORMS__/MacOSX.platform/Developer/Library/PrivateFrameworks/IDEBundleInjection.framework/IDEBundleInjection",
        ]

    subs = {
        "xctestrun_insert_libraries": ":".join(xctestrun_insert_libraries),
    }

    return {"%(" + k + ")s": subs[k] for k in subs}

def _get_template_substitutions(
        *,
        xctestrun_template,
        pre_action_binary,
        post_action_binary,
        post_action_determines_exit_code):
    """Returns the template substitutions for this runner."""
    subs = {
        "xctestrun_template": xctestrun_template.short_path,
        "pre_action_binary": pre_action_binary,
        "post_action_binary": post_action_binary,
        "post_action_determines_exit_code": post_action_determines_exit_code,
    }

    return {"%(" + k + ")s": subs[k] for k in subs}

def _get_execution_environment(xcode_config):
    """Returns environment variables the test runner requires"""
    execution_environment = {}
    xcode_version = str(xcode_config.xcode_version())
    if xcode_version:
        execution_environment["XCODE_VERSION_OVERRIDE"] = xcode_version

    return execution_environment

def _macos_test_runner_impl(ctx):
    """Implementation for the macos_runner rule."""
    preprocessed_xctestrun_template = ctx.actions.declare_file(
        "{}.generated.xctestrun".format(ctx.label.name),
    )

    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]

    ctx.actions.expand_template(
        template = ctx.file._xctestrun_template,
        output = preprocessed_xctestrun_template,
        substitutions = _get_xctestrun_template_substitutions(xcode_config),
    )

    runfiles = ctx.runfiles(files = [preprocessed_xctestrun_template])

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
            xctestrun_template = preprocessed_xctestrun_template,
            pre_action_binary = pre_action_binary,
            post_action_binary = post_action_binary,
            post_action_determines_exit_code = "true" if post_action_determines_exit_code else "false",
        ),
    )

    return [
        apple_provider.make_apple_test_runner_info(
            test_runner_template = ctx.outputs.test_runner_template,
            execution_requirements = {"requires-darwin": ""},
            execution_environment = _get_execution_environment(xcode_config),
        ),
        DefaultInfo(runfiles = runfiles),
    ]

macos_test_runner = rule(
    _macos_test_runner_impl,
    attrs = {
        "pre_action": attr.label(
            executable = True,
            cfg = "exec",
            doc = """
A binary to run prior to test execution. Sets any environment variables available to the test runner.
""",
        ),
        "post_action": attr.label(
            executable = True,
            cfg = "exec",
            doc = """
A binary to run following test execution. Runs after testing but before test result handling and coverage processing. Sets the `$TEST_EXIT_CODE`, `$TEST_LOG_FILE`, and `$TEST_XCRESULT_BUNDLE_PATH` environment variables, in addition to any other variables available to the test runner.
""",
        ),
        "post_action_determines_exit_code": attr.bool(
            default = False,
            doc = """
When true, the exit code of the test run will be set to the exit code of the post action. This is useful for tests that need to fail the test run based on their own criteria.
""",
        ),
        "_test_template": attr.label(
            default = Label("//apple/testing/default_runner:macos_test_runner.template.sh"),
            allow_single_file = True,
        ),
        "_xcode_config": attr.label(
            default = configuration_field(
                fragment = "apple",
                name = "xcode_config_label",
            ),
        ),
        "_xctestrun_template": attr.label(
            default = Label("//apple/testing/default_runner:macos_test_runner.template.xctestrun"),
            allow_single_file = True,
        ),
    },
    outputs = {
        "test_runner_template": "%{name}.sh",
    },
    fragments = ["apple", "objc"],
    doc = """
Rule to identify an macOS runner that runs tests for macOS.

Provides:
  AppleTestRunnerInfo:
    test_runner_template: Template file that contains the specific mechanism
        with which the tests will be performed.
    execution_requirements: Dictionary that represents the specific hardware
        requirements for this test.
    execution_environment: Dictionary with the environment variables the test runner requires
  Runfiles:
    files: The files needed during runtime for the test to be performed.
""",
)
