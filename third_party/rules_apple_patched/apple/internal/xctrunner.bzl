# Copyright 2025 The Bazel Authors. All rights reserved.
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

"""
Rule for creating a XCTRunner.app with one or more .xctest bundles. Retains same 
platform and architectures as the given `tests` bundles.
"""

load(
    "@build_bazel_apple_support//lib:apple_support.bzl",
    "apple_support",
)
load(
    "//apple:providers.bzl",
    "AppleBundleInfo",
)
load(
    "//apple/internal:providers.bzl",
    "new_applebundleinfo",
)

_TestBundleInfo = provider(
    "Test bundle info for tests that will be run.",
    fields = {
        "platform_type": "The platform to bundle for.",
        "infoplists": "A `depset` of `File`s of `Info.plist` files.",
        "xctests": "A `depset` of paths of XCTest bundles.",
    },
)

PLATFORM_MAP = {
    "ios": "iPhoneOS.platform",
    "macos": "MacOSX.platform",
    "tvos": "AppleTVOS.platform",
    "watchos": "WatchOS.platform",
    "visionos": "VisionOS.platform",
}

def _test_bundle_info_aspect_impl(target, ctx):
    rule_attr = ctx.rule.attr

    if AppleBundleInfo in target:
        info = target[AppleBundleInfo]
        xctests = depset([info.archive])
        infoplists = depset([info.infoplist])
        platform_type = target[AppleBundleInfo].platform_type
    else:
        deps = getattr(rule_attr, "tests", [])
        xctests = depset(
            transitive = [
                dep[_TestBundleInfo].xctests
                for dep in deps
            ],
        )
        infoplists = depset(
            transitive = [
                dep[_TestBundleInfo].infoplists
                for dep in deps
            ],
        )
        platform_types = [
            dep[_TestBundleInfo].platform_type
            for dep in deps
        ]

        # Ensure all test bundles are for the same platform
        for type in platform_types:
            if type != platform_types[0]:
                ctx.attr.test_bundle_info_aspect.error(
                    "All test bundles must be for the same platform: %s" % platform_types,
                )
        platform_type = platform_types[0]  # Pick one, all are same

    return [
        _TestBundleInfo(
            infoplists = infoplists,
            xctests = xctests,
            platform_type = platform_type,
        ),
    ]

test_bundle_info_aspect = aspect(
    attr_aspects = ["tests"],
    implementation = _test_bundle_info_aspect_impl,
)

def _xctrunner_impl(ctx):
    apple_fragment = ctx.fragments.apple
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]

    output = ctx.actions.declare_directory(ctx.attr.name + ".app")
    infos = [target[_TestBundleInfo] for target in ctx.attr.tests]
    infoplists = depset(
        transitive = [info.infoplists for info in infos],
    )
    xctests = depset(
        transitive = [info.xctests for info in infos],
    )
    platform = infos[0].platform_type  # Pick one, all should be same

    # Args for `_xctrunnertool`
    args = ctx.actions.args()
    args.add("--name", ctx.attr.name)
    args.add("--platform", PLATFORM_MAP[platform])
    args.add("--xcode-version", xcode_config.xcode_version())
    if ctx.attr.verbose:
        args.add("--verbose", ctx.attr.verbose)

    args.add_all(
        xctests,
        before_each = "--xctest",
        expand_directories = False,
    )

    args.add("--output", output.path)

    apple_support.run(
        actions = ctx.actions,
        xcode_config = xcode_config,
        apple_fragment = apple_fragment,
        inputs = depset(transitive = [xctests, infoplists]),
        outputs = [output],
        executable = ctx.attr._xctrunnertool[DefaultInfo].files_to_run,
        arguments = [args],
        mnemonic = "MakeXCTRunner",
    )

    bundle_info = new_applebundleinfo(
        archive = output,
        bundle_name = ctx.attr.name,
        bundle_extension = ".app",
        bundle_id = "com.apple.test.{}".format(ctx.attr.name),
        executable_name = ctx.attr.name,
        infoplist = "{}/Info.plist".format(output.path),
        platform_type = platform,
        product_type = "com.apple.product-type.bundle.ui-testing",
    )

    return [
        DefaultInfo(files = depset([output])),
        bundle_info,
    ]

xctrunner = rule(
    implementation = _xctrunner_impl,
    attrs = {
        "tests": attr.label_list(
            mandatory = True,
            aspects = [test_bundle_info_aspect],
            doc = "List of test targets and suites to include.",
        ),
        "verbose": attr.bool(
            mandatory = False,
            default = False,
            doc = "Print logs from xctrunnertool to console.",
        ),
        "_xctrunnertool": attr.label(
            default = Label("//tools/xctrunnertool:run"),
            executable = True,
            cfg = "exec",
            doc = """
An executable binary that can merge separate xctest into a single XCTestRunner
bundle.
""",
        ),
    } | apple_support.action_required_attrs(),
    doc = """
Packages one or more .xctest bundles into a XCTRunner.app. Retains same 
platform and architectures as the given `tests` bundles.

Example:

````starlark
load("//apple:xctrunner.bzl", "xctrunner")

ios_ui_test(
    name = "HelloWorldSwiftUITests",
    minimum_os_version = "15.0",
    runner = "@build_bazel_rules_apple//apple/testing/default_runner:ios_xctestrun_ordered_runner",
    test_host = ":HelloWorldSwift",
    deps = [":UITests"],
)

xctrunner(
    name = "HelloWorldSwiftXCTRunner",
    tests = [":HelloWorldSwiftUITests"],
    testonly = True,
)
````
""",
    fragments = ["apple"],
)
