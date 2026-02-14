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

"""Bazel rules for creating macOS applications and bundles."""

load(
    "//apple/internal:macos_binary_support.bzl",
    "macos_binary_infoplist",
    "macos_command_line_launchdplist",
)
load(
    "//apple/internal:macos_rules.bzl",
    _macos_application = "macos_application",
    _macos_bundle = "macos_bundle",
    _macos_command_line_application = "macos_command_line_application",
    _macos_dylib = "macos_dylib",
    _macos_dynamic_framework = "macos_dynamic_framework",
    _macos_extension = "macos_extension",
    _macos_framework = "macos_framework",
    _macos_kernel_extension = "macos_kernel_extension",
    _macos_quick_look_plugin = "macos_quick_look_plugin",
    _macos_spotlight_importer = "macos_spotlight_importer",
    _macos_static_framework = "macos_static_framework",
    _macos_xpc_service = "macos_xpc_service",
)
load(
    "//apple/internal/testing:apple_test_assembler.bzl",
    "apple_test_assembler",
)
load(
    "//apple/internal/testing:build_test_rules.bzl",
    "apple_build_test_rule",
)
load(
    "//apple/internal/testing:macos_rules.bzl",
    _macos_internal_ui_test_bundle = "macos_internal_ui_test_bundle",
    _macos_internal_unit_test_bundle = "macos_internal_unit_test_bundle",
    _macos_ui_test = "macos_ui_test",
    _macos_unit_test = "macos_unit_test",
)

# TODO(b/118104491): Remove these re-exports and move the rule definitions into this file.
macos_quick_look_plugin = _macos_quick_look_plugin
macos_spotlight_importer = _macos_spotlight_importer
macos_xpc_service = _macos_xpc_service

def macos_application(name, **kwargs):
    # buildifier: disable=function-docstring-args
    """Packages a macOS application."""
    bundling_args = dict(kwargs)
    features = bundling_args.pop("features", [])
    features.append("link_cocoa")

    _macos_application(
        name = name,
        features = features,
        **bundling_args
    )

def macos_bundle(name, **kwargs):
    # buildifier: disable=function-docstring-args
    """Packages a macOS loadable bundle."""
    bundling_args = dict(kwargs)
    features = bundling_args.pop("features", [])
    features.append("link_cocoa")

    _macos_bundle(
        name = name,
        features = features,
        **bundling_args
    )

def macos_kernel_extension(name, **kwargs):
    # buildifier: disable=function-docstring-args
    """Packages a macOS Kernel Extension."""
    bundling_args = dict(kwargs)
    features = bundling_args.pop("features", [])
    features.append("kernel_extension")

    _macos_kernel_extension(
        name = name,
        features = features,
        **bundling_args
    )

def macos_command_line_application(name, **kwargs):
    # buildifier: disable=function-docstring-args
    """Builds a macOS command line application."""

    binary_args = dict(kwargs)

    original_deps = binary_args.pop("deps")
    binary_deps = list(original_deps)

    # If any of the Info.plist-affecting attributes is provided, create a merged
    # Info.plist target. This target also propagates an objc provider that
    # contains the linkopts necessary to add the Info.plist to the binary, so it
    # must become a dependency of the binary as well.
    base_bundle_id = binary_args.get("base_bundle_id")
    bundle_id = binary_args.get("bundle_id")
    infoplists = binary_args.get("infoplists")
    launchdplists = binary_args.get("launchdplists")
    version = binary_args.get("version")

    if base_bundle_id or bundle_id or infoplists or version:
        merged_infoplist_name = name + ".merged_infoplist"

        macos_binary_infoplist(
            name = merged_infoplist_name,
            base_bundle_id = base_bundle_id,
            bundle_id = bundle_id,
            bundle_id_suffix = binary_args.get("bundle_id_suffix"),
            infoplists = infoplists,
            minimum_os_version = binary_args.get("minimum_os_version"),
            version = version,
        )
        binary_deps.extend([":" + merged_infoplist_name])

    if launchdplists:
        merged_launchdplists_name = name + ".merged_launchdplists"

        macos_command_line_launchdplist(
            name = merged_launchdplists_name,
            launchdplists = launchdplists,
        )
        binary_deps.extend([":" + merged_launchdplists_name])

    _macos_command_line_application(
        name = name,
        deps = binary_deps,
        **binary_args
    )

def macos_dylib(name, **kwargs):
    # buildifier: disable=function-docstring-args
    """Builds a macOS dylib."""

    # Xcode will happily apply entitlements during code signing for a dylib even
    # though it doesn't have a Capabilities tab in the project settings.
    # Until there's official support for it, we'll fail if we see those attributes
    # (which are added to the rule because of the code_signing_attributes usage in
    # the rule definition).
    if "entitlements" in kwargs or "provisioning_profile" in kwargs:
        fail("macos_dylib does not support entitlements or provisioning " +
             "profiles at this time")

    binary_args = dict(kwargs)

    original_deps = binary_args.pop("deps")
    binary_deps = list(original_deps)

    # If any of the Info.plist-affecting attributes is provided, create a merged
    # Info.plist target. This target also propagates an objc provider that
    # contains the linkopts necessary to add the Info.plist to the binary, so it
    # must become a dependency of the binary as well.
    base_bundle_id = binary_args.get("base_bundle_id")
    bundle_id = binary_args.get("bundle_id")
    infoplists = binary_args.get("infoplists")
    version = binary_args.get("version")

    if base_bundle_id or bundle_id or infoplists or version:
        merged_infoplist_name = name + ".merged_infoplist"

        macos_binary_infoplist(
            name = merged_infoplist_name,
            base_bundle_id = base_bundle_id,
            bundle_id = bundle_id,
            bundle_id_suffix = binary_args.get("bundle_id_suffix"),
            infoplists = infoplists,
            minimum_os_version = binary_args.get("minimum_os_version"),
            version = version,
        )
        binary_deps.extend([":" + merged_infoplist_name])

    _macos_dylib(
        name = name,
        deps = binary_deps,
        **binary_args
    )

def macos_extension(name, **kwargs):
    # buildifier: disable=function-docstring-args
    """Packages a macOS Extension Bundle."""
    bundling_args = dict(kwargs)

    features = bundling_args.pop("features", [])
    features.append("link_cocoa")

    _macos_extension(
        name = name,
        features = features,
        **bundling_args
    )

def macos_framework(name, **kwargs):
    # buildifier: disable=function-docstring-args
    """Packages a macOS framework."""
    bundling_args = dict(kwargs)
    features = bundling_args.pop("features", [])
    features.append("link_cocoa")

    _macos_framework(
        name = name,
        features = features,
        **bundling_args
    )

def macos_static_framework(name, **kwargs):
    # buildifier: disable=function-docstring-args
    """Packages a macOS framework."""
    bundling_args = dict(kwargs)
    features = bundling_args.pop("features", [])
    features.append("link_cocoa")

    _macos_static_framework(
        name = name,
        features = features,
        **bundling_args
    )

def macos_dynamic_framework(name, **kwargs):
    # buildifier: disable=function-docstring-args
    """Packages a macOS framework."""
    bundling_args = dict(kwargs)
    features = bundling_args.pop("features", [])
    features.append("link_cocoa")

    _macos_dynamic_framework(
        name = name,
        features = features,
        **bundling_args
    )

_DEFAULT_TEST_RUNNER = str(Label("//apple/testing/default_runner:macos_default_runner"))

def macos_unit_test(name, **kwargs):
    runner = kwargs.pop("runner", _DEFAULT_TEST_RUNNER)
    apple_test_assembler.assemble(
        name = name,
        bundle_rule = _macos_internal_unit_test_bundle,
        test_rule = _macos_unit_test,
        runner = runner,
        **kwargs
    )

def macos_ui_test(name, **kwargs):
    runner = kwargs.pop("runner", _DEFAULT_TEST_RUNNER)
    apple_test_assembler.assemble(
        name = name,
        bundle_rule = _macos_internal_ui_test_bundle,
        test_rule = _macos_ui_test,
        runner = runner,
        **kwargs
    )

macos_build_test = apple_build_test_rule(
    doc = """\
Test rule to check that the given library targets (Swift, Objective-C, C++)
build for macOS.

Typical usage:

```starlark
macos_build_test(
    name = "my_build_test",
    minimum_os_version = "10.14",
    targets = [
        "//some/package:my_library",
    ],
)
```
""",
    platform_type = "macos",
)
