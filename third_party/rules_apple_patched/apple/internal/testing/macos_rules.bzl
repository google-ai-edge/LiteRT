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

"""Implementation of macOS test rules."""

load(
    "//apple:providers.bzl",
    "AppleBundleInfo",
    "MacosApplicationBundleInfo",
    "MacosFrameworkBundleInfo",
)
load(
    "//apple/internal:apple_product_type.bzl",
    "apple_product_type",
)
load(
    "//apple/internal:bundling_support.bzl",
    "bundle_id_suffix_default",
)
load(
    "//apple/internal:providers.bzl",
    "new_macosxctestbundleinfo",
)
load(
    "//apple/internal:rule_attrs.bzl",
    "rule_attrs",
)
load(
    "//apple/internal:rule_factory.bzl",
    "rule_factory",
)
load(
    "//apple/internal:transition_support.bzl",
    "transition_support",
)
load(
    "//apple/internal/aspects:framework_provider_aspect.bzl",
    "framework_provider_aspect",
)
load(
    "//apple/internal/aspects:resource_aspect.bzl",
    "apple_resource_aspect",
)
load(
    "//apple/internal/testing:apple_test_bundle_support.bzl",
    "apple_test_bundle_support",
)
load(
    "//apple/internal/testing:apple_test_rule_support.bzl",
    "apple_test_rule_support",
)

_MACOS_TEST_HOST_PROVIDERS = [[AppleBundleInfo, MacosApplicationBundleInfo]]

def _macos_ui_test_bundle_impl(ctx):
    """Implementation of macos_ui_test."""
    return apple_test_bundle_support.apple_test_bundle_impl(
        ctx = ctx,
        product_type = apple_product_type.ui_test_bundle,
    ) + [
        new_macosxctestbundleinfo(),
    ]

def _macos_unit_test_bundle_impl(ctx):
    """Implementation of macos_unit_test."""
    return apple_test_bundle_support.apple_test_bundle_impl(
        ctx = ctx,
        product_type = apple_product_type.unit_test_bundle,
    ) + [
        new_macosxctestbundleinfo(),
    ]

def _macos_ui_test_impl(ctx):
    """Implementation of macos_ui_test."""
    return apple_test_rule_support.apple_test_rule_impl(
        ctx = ctx,
        requires_dossiers = False,
        test_type = "xcuitest",
    ) + [
        new_macosxctestbundleinfo(),
    ]

def _macos_unit_test_impl(ctx):
    """Implementation of macos_unit_test."""
    return apple_test_rule_support.apple_test_rule_impl(
        ctx = ctx,
        requires_dossiers = False,
        test_type = "xctest",
    ) + [
        new_macosxctestbundleinfo(),
    ]

# Declare it with an underscore to hint that this is an implementation detail in bazel query-s.
_macos_internal_ui_test_bundle = rule_factory.create_apple_rule(
    doc = "Builds and bundles an macOS UI Test Bundle. Internal target not to be depended upon.",
    implementation = _macos_ui_test_bundle_impl,
    predeclared_outputs = {"archive": "%{name}.zip"},
    attrs = [
        rule_attrs.binary_linking_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
            extra_deps_aspects = [
                apple_resource_aspect,
                framework_provider_aspect,
            ],
            is_test_supporting_rule = True,
            requires_legacy_cc_toolchain = True,
        ),
        rule_attrs.common_bundle_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
        ),
        rule_attrs.common_tool_attrs(),
        rule_attrs.device_family_attrs(
            allowed_families = rule_attrs.defaults.allowed_families.macos,
            is_mandatory = False,
        ),
        rule_attrs.infoplist_attrs(
            default_infoplist = rule_attrs.defaults.test_bundle_infoplist,
        ),
        rule_attrs.ipa_post_processor_attrs(),
        rule_attrs.platform_attrs(
            add_environment_plist = True,
            platform_type = "macos",
        ),
        rule_attrs.signing_attrs(
            default_bundle_id_suffix = bundle_id_suffix_default.bundle_name,
            supports_capabilities = False,
            profile_extension = ".provisionprofile",
        ),
        rule_attrs.test_bundle_attrs(),
        rule_attrs.test_host_attrs(
            aspects = rule_attrs.aspects.test_host_aspects,
            is_mandatory = True,
            providers = _MACOS_TEST_HOST_PROVIDERS,
        ),
        {
            "additional_contents": attr.label_keyed_string_dict(
                allow_files = True,
                doc = """
Files that should be copied into specific subdirectories of the Contents folder in the bundle. The
keys of this dictionary are labels pointing to single files, filegroups, or targets; the
corresponding value is the name of the subdirectory of Contents where they should be placed.

The relative directory structure of filegroup contents is preserved when they are copied into the
desired Contents subdirectory.
""",
            ),
        },
        {
            "frameworks": attr.label_list(
                providers = [[AppleBundleInfo, MacosFrameworkBundleInfo]],
                doc = """
A list of framework targets (see
[`macos_framework`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-macos.md#macos_framework))
that this target depends on.
""",
            ),
        },
    ],
)

# Alias to import it.
macos_internal_ui_test_bundle = _macos_internal_ui_test_bundle

macos_ui_test = rule_factory.create_apple_test_rule(
    doc = """Builds and bundles an iOS UI `.xctest` test bundle. Runs the tests using the
provided test runner when invoked with `bazel test`.

Note: macOS UI tests are not currently supported in the default test runner.""",
    implementation = _macos_ui_test_impl,
    platform_type = "macos",
)

# Declare it with an underscore to hint that this is an implementation detail in bazel query-s.
_macos_internal_unit_test_bundle = rule_factory.create_apple_rule(
    doc = "Builds and bundles an macOS Unit Test Bundle. Internal target not to be depended upon.",
    implementation = _macos_unit_test_bundle_impl,
    predeclared_outputs = {"archive": "%{name}.zip"},
    attrs = [
        rule_attrs.binary_linking_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
            extra_deps_aspects = [
                apple_resource_aspect,
                framework_provider_aspect,
            ],
            is_test_supporting_rule = True,
            requires_legacy_cc_toolchain = True,
        ),
        rule_attrs.common_bundle_attrs(
            deps_cfg = transition_support.apple_platform_split_transition,
        ),
        rule_attrs.common_tool_attrs(),
        rule_attrs.device_family_attrs(
            allowed_families = rule_attrs.defaults.allowed_families.macos,
            is_mandatory = False,
        ),
        rule_attrs.infoplist_attrs(
            default_infoplist = rule_attrs.defaults.test_bundle_infoplist,
        ),
        rule_attrs.ipa_post_processor_attrs(),
        rule_attrs.platform_attrs(
            add_environment_plist = True,
            platform_type = "macos",
        ),
        rule_attrs.signing_attrs(
            default_bundle_id_suffix = bundle_id_suffix_default.bundle_name,
            supports_capabilities = False,
            profile_extension = ".provisionprofile",
        ),
        rule_attrs.test_bundle_attrs(),
        rule_attrs.test_host_attrs(
            aspects = rule_attrs.aspects.test_host_aspects,
            providers = _MACOS_TEST_HOST_PROVIDERS,
        ),
        {
            "additional_contents": attr.label_keyed_string_dict(
                allow_files = True,
                doc = """
Files that should be copied into specific subdirectories of the Contents folder in the bundle. The
keys of this dictionary are labels pointing to single files, filegroups, or targets; the
corresponding value is the name of the subdirectory of Contents where they should be placed.

The relative directory structure of filegroup contents is preserved when they are copied into the
desired Contents subdirectory.
""",
            ),
        },
        {
            "frameworks": attr.label_list(
                providers = [[AppleBundleInfo, MacosFrameworkBundleInfo]],
                doc = """
A list of framework targets (see
[`macos_framework`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-macos.md#macos_framework))
that this target depends on.
""",
            ),
        },
    ],
)

# Alias to import it.
macos_internal_unit_test_bundle = _macos_internal_unit_test_bundle

macos_unit_test = rule_factory.create_apple_test_rule(
    doc = """Builds and bundles a macOS unit `.xctest` test bundle. Runs the tests using the
provided test runner when invoked with `bazel test`.

`macos_unit_test` targets can work in two modes: as app or library tests. If the
`test_host` attribute is set to an `macos_application` target, the tests will
run within that application's context. If no `test_host` is provided, the tests
will run outside the context of an macOS application. Because of this, certain
functionalities might not be present (e.g. UI layout, NSUserDefaults). You can
find more information about testing for Apple platforms
[here](https://developer.apple.com/library/content/documentation/DeveloperTools/Conceptual/testing_with_xcode/chapters/03-testing_basics.html).""",
    implementation = _macos_unit_test_impl,
    platform_type = "macos",
)
