# Copyright 2019 The Bazel Authors. All rights reserved.
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

"""Test rule to perform generic bundle verification tests.

This rule is meant to be used only for rules_apple tests and are considered implementation details
that may change at any time. Please do not depend on this rule.
"""

load(
    "@bazel_skylib//lib:dicts.bzl",
    "dicts",
)
load(
    "@bazel_skylib//lib:paths.bzl",
    "paths",
)
load(
    "//apple:providers.bzl",
    "AppleBinaryInfo",
    "AppleBundleInfo",
)
load(
    "//apple/build_settings:build_settings.bzl",
    "build_settings_labels",
)
load(
    "//apple/internal:apple_product_type.bzl",  # buildifier: disable=bzl-visibility
    "apple_product_type",
)  # buildifier: disable=bzl-visibility

_supports_visionos = hasattr(apple_common.platform_type, "visionos")

_CUSTOM_BUILD_SETTINGS = build_settings_labels.all_labels + [
]

def _apple_verification_transition_impl(settings, attr):
    """Implementation of the apple_verification_transition transition."""

    has_apple_platforms = True if getattr(attr, "apple_platforms", []) else False
    has_apple_cpus = True if getattr(attr, "cpus", {}) else False

    # Kept mutually exclusive as a preference to test new-style toolchain resolution separately from
    # old-style toolchain resolution.
    if has_apple_platforms and has_apple_cpus:
        fail("""
Internal Error: A verification test should only specify `apple_platforms` or `cpus`, but not both.
""")

    output_dictionary = {
        "//command_line_option:apple_platforms": [],
        "//command_line_option:cpu": getattr(attr, "apple_cpu", "darwin_x86_64"),
        "//command_line_option:macos_cpus": "x86_64",
        "//command_line_option:compilation_mode": attr.compilation_mode,
        "//command_line_option:objc_enable_binary_stripping": getattr(attr, "objc_enable_binary_stripping") if hasattr(attr, "objc_enable_binary_stripping") else False,
        "//command_line_option:objc_generate_linkmap": getattr(attr, "objc_generate_linkmap", "False"),
        "//command_line_option:apple_generate_dsym": getattr(attr, "apple_generate_dsym", "False"),
        "//command_line_option:incompatible_enable_apple_toolchain_resolution": has_apple_platforms,
    }
    if attr.build_type == "simulator":
        output_dictionary.update({
            "//command_line_option:ios_multi_cpus": "x86_64",
            "//command_line_option:tvos_cpus": "x86_64",
            "//command_line_option:watchos_cpus": "x86_64",
        })

        if _supports_visionos:
            output_dictionary["//command_line_option:visionos_cpus"] = "sim_arm64"
    else:
        output_dictionary.update({
            "//command_line_option:ios_multi_cpus": "arm64,arm64e",
            "//command_line_option:tvos_cpus": "arm64",
            "//command_line_option:watchos_cpus": "arm64_32,armv7k",
        })

        if _supports_visionos:
            output_dictionary["//command_line_option:visionos_cpus"] = "arm64"

    if has_apple_platforms:
        output_dictionary.update({
            "//command_line_option:apple_platforms": ",".join(attr.apple_platforms),
        })
    elif has_apple_cpus:
        for cpu_option, cpus in attr.cpus.items():
            if not _supports_visionos and cpu_option == "visionos_cpus":
                continue
            command_line_option = "//command_line_option:%s" % cpu_option
            output_dictionary.update({command_line_option: ",".join(cpus)})

    # Features
    existing_features = settings.get("//command_line_option:features") or []
    if hasattr(attr, "target_features"):
        existing_features.extend(attr.target_features)
    if hasattr(attr, "sanitizer") and attr.sanitizer != "none":
        existing_features.append(attr.sanitizer)
    output_dictionary["//command_line_option:features"] = existing_features

    # Build settings
    test_build_settings = {
        build_settings_labels.signing_certificate_name: "-",
    }
    test_build_settings.update(getattr(attr, "build_settings", {}))
    for build_setting in _CUSTOM_BUILD_SETTINGS:
        if build_setting in test_build_settings:
            build_setting_value = test_build_settings[build_setting]
            build_setting_type = type(settings[build_setting])

            # The `build_settings` rule attribute requires string values. However, build
            # settings can have many types. In order to set the correct type, we inspect
            # the default value from settings, and cast accordingly.
            if build_setting_type == "bool":
                build_setting_value = build_setting_value.lower() in ("true", "yes", "1")

            output_dictionary[build_setting] = build_setting_value
        else:
            output_dictionary[build_setting] = settings[build_setting]

    return output_dictionary

apple_verification_transition = transition(
    implementation = _apple_verification_transition_impl,
    inputs = [
        "//command_line_option:features",
    ] + _CUSTOM_BUILD_SETTINGS,
    outputs = [
        "//command_line_option:cpu",
        "//command_line_option:ios_multi_cpus",
        "//command_line_option:macos_cpus",
        "//command_line_option:tvos_cpus",
        "//command_line_option:watchos_cpus",
        "//command_line_option:compilation_mode",
        "//command_line_option:features",
        "//command_line_option:apple_generate_dsym",
        "//command_line_option:apple_platforms",
        "//command_line_option:incompatible_enable_apple_toolchain_resolution",
        "//command_line_option:objc_enable_binary_stripping",
        "//command_line_option:objc_generate_linkmap",
    ] + _CUSTOM_BUILD_SETTINGS + (["//command_line_option:visionos_cpus"] if _supports_visionos else []),
)

def _apple_verification_test_impl(ctx):
    """Implementation of the apple_verification_test rule."""

    # Should be using split_attr instead, but it has been disabled due to
    # https://github.com/bazelbuild/bazel/issues/8633
    target_under_test = ctx.attr.target_under_test[0]
    if AppleBundleInfo in target_under_test:
        bundle_info = target_under_test[AppleBundleInfo]
        archive = bundle_info.archive

        bundle_with_extension = bundle_info.bundle_name + bundle_info.bundle_extension

        if bundle_info.platform_type in [
            "ios",
            "tvos",
        ] and bundle_info.product_type in [
            apple_product_type.application,
            apple_product_type.app_clip,
            apple_product_type.messages_application,
        ] and not archive.is_directory:
            archive_relative_bundle = paths.join("Payload", bundle_with_extension)
        else:
            archive_relative_bundle = bundle_with_extension

        if bundle_info.platform_type == "macos":
            archive_relative_contents = paths.join(archive_relative_bundle, "Contents")
            archive_relative_binary = paths.join(
                archive_relative_contents,
                "MacOS",
                bundle_info.executable_name,
            )
            archive_relative_resources = paths.join(archive_relative_contents, "Resources")
        else:
            archive_relative_contents = archive_relative_bundle
            archive_relative_binary = paths.join(
                archive_relative_bundle,
                bundle_info.executable_name,
            )
            archive_relative_resources = archive_relative_bundle

        archive_short_path = archive.short_path
        output_to_verify = archive
        standalone_binary_short_path = ""
    elif AppleBinaryInfo in target_under_test:
        output_to_verify = target_under_test[AppleBinaryInfo].binary
        standalone_binary_short_path = target_under_test[AppleBinaryInfo].binary.short_path
        archive_short_path = ""
        archive_relative_binary = ""
        archive_relative_bundle = ""
        archive_relative_contents = ""
        archive_relative_resources = ""
    else:
        fail(("Target %s does not provide AppleBundleInfo or AppleBinaryInfo") %
             target_under_test.label)

    source_dependencies = ""
    for dep in ctx.attr._test_deps.files.to_list():
        source_dependencies += "source {}\n".format(dep.short_path)

    output_script = ctx.actions.declare_file("{}_test_script".format(ctx.label.name))
    ctx.actions.expand_template(
        template = ctx.file._runner_script,
        output = output_script,
        substitutions = {
            "%{archive}s": archive_short_path,
            "%{dependencies}s": source_dependencies,
            "%{standalone_binary}s": standalone_binary_short_path,
            "%{archive_relative_binary}s": archive_relative_binary,
            "%{archive_relative_bundle}s": archive_relative_bundle,
            "%{archive_relative_contents}s": archive_relative_contents,
            "%{archive_relative_resources}s": archive_relative_resources,
            "%{verifier_script}s": ctx.file.verifier_script.short_path,
        },
        is_executable = True,
    )

    # Apply knowledge of the Xcode version to the test environnment
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]
    xcode_version_split = str(xcode_config.xcode_version()).split(".")
    xcode_versions_separated = len(xcode_version_split)

    # Extra test environment to set during the test.
    test_env = {
        "BUILD_TYPE": ctx.attr.build_type,
        "XCODE_VERSION_MAJOR": xcode_version_split[0] if xcode_versions_separated >= 1 else 0,
        "XCODE_VERSION_MINOR": xcode_version_split[1] if xcode_versions_separated >= 2 else 0,
    }

    # Create APPLE_TEST_ENV_# environmental variables for each `env` attribute that are transformed
    # into bash arrays. This allows us to not need any extra sentinal/delimiter characters in the
    # values.
    test_env["APPLE_TEST_ENV_KEYS"] = " ".join(ctx.attr.env.keys())
    for key in ctx.attr.env:
        for num, value in enumerate(ctx.attr.env[key]):
            test_env["APPLE_TEST_ENV_{}_{}".format(key, num)] = value

    return [
        testing.ExecutionInfo(xcode_config.execution_info()),
        testing.TestEnvironment(dicts.add(
            apple_common.apple_host_system_env(xcode_config),
            test_env,
        )),
        DefaultInfo(
            executable = output_script,
            runfiles = ctx.runfiles(
                files = [output_to_verify, ctx.file.verifier_script] +
                        ctx.attr._test_deps.files.to_list(),
            ),
        ),
        target_under_test[OutputGroupInfo],
    ]

# Need a cfg for a transition on target_under_test, so can't use analysistest.make.
apple_verification_test = rule(
    implementation = _apple_verification_test_impl,
    attrs = {
        "apple_cpu": attr.string(
            doc = """
A string to indicate what should be the value of the Apple --cpu flag. Defaults to `darwin_x86_64`.
""",
        ),
        "apple_generate_dsym": attr.bool(
            default = False,
            doc = """
If true, generates .dSYM debug symbol bundles for the target(s) under test.
""",
        ),
        "apple_platforms": attr.string_list(
            doc = """
List of strings representing Apple platform definitions to resolve. When set, this opts into
toolchain resolution to select the Apple SDK for Apple rules (Starlark and native). Currently it is
considered to be an error if this is set with `cpus` as both opt into different means of toolchain
resolution.
""",
        ),
        "build_settings": attr.string_dict(
            mandatory = False,
            doc = "Build settings for target under test.",
        ),
        "build_type": attr.string(
            mandatory = True,
            values = ["simulator", "device"],
            doc = """
Type of build for the target under test. Possible values are `simulator` or `device`.
""",
        ),
        "compilation_mode": attr.string(
            values = ["fastbuild", "opt", "dbg"],
            doc = """
Possible values are `fastbuild`, `dbg` or `opt`. Defaults to `fastbuild`.
https://docs.bazel.build/versions/master/user-manual.html#flag--compilation_mode
""",
            default = "fastbuild",
        ),
        "cpus": attr.string_list_dict(
            doc = """
Dictionary of command line options cpu flags (e.g. ios_multi_cpus, macos_cpus) and the list of
cpu's to use for test under target (e.g. {'ios_multi_cpus': ['arm64', 'x86_64']}) Currently it is
considered to be an error if this is set with `apple_platforms` as both opt into different means of
toolchain resolution.
""",
        ),
        "objc_enable_binary_stripping": attr.bool(
            default = False,
            doc = """
Whether to perform symbol and dead-code strippings on linked binaries. Binary
strippings will be performed if both this flag and --compilation_mode=opt are
specified.
""",
        ),
        "objc_generate_linkmap": attr.bool(
            default = False,
            doc = """
If true, generates a linkmap file for the target(s) under test.
""",
        ),
        "sanitizer": attr.string(
            default = "none",
            values = ["none", "asan", "tsan", "ubsan"],
            doc = """
Possible values are `none`, `asan`, `tsan` or `ubsan`. Defaults to `none`.
Passes a sanitizer to the target under test.
""",
        ),
        "target_features": attr.string_list(
            mandatory = False,
            doc = """
List of additional features to build for the target under testing.
""",
        ),
        "target_under_test": attr.label(
            mandatory = True,
            providers = [[AppleBinaryInfo], [AppleBundleInfo]],
            doc = "The Apple binary or Apple bundle target whose contents are to be verified.",
            cfg = apple_verification_transition,
        ),
        "verifier_script": attr.label(
            mandatory = True,
            allow_single_file = [".sh"],
            doc = """
Shell script containing the verification code. This script can expect the following environment
variables to exist:

* ARCHIVE_ROOT: The path to the unzipped `.ipa` or `.zip` archive that was the output of the
  build.
* BINARY: The path to the main bundle binary.
* BUILD_TYPE: The type of build for the target under test. Can be `simulator` or `device`.
* BUNDLE_ROOT: The directory where the bundle is located.
* CONTENT_ROOT: The directory where the bundle contents are located.
* RESOURCE_ROOT: The directory where the resource files are located.
""",
        ),
        "env": attr.string_list_dict(
            doc = """
The environmental variables to pass to the verifier script. The list of strings will be transformed
into a bash array.
""",
        ),
        "_runner_script": attr.label(
            allow_single_file = True,
            default = "//test/starlark_tests:verifier_scripts/apple_verification_test_runner.sh.template",
        ),
        "_test_deps": attr.label(
            default = "//test:apple_verification_test_deps",
        ),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
        "_xcode_config": attr.label(
            default = configuration_field(
                name = "xcode_config_label",
                fragment = "apple",
            ),
        ),
    },
    test = True,
    fragments = ["apple"],
)
