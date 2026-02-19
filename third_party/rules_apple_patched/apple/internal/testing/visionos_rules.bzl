# Copyright 2023 The Bazel Authors. All rights reserved.
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

"""Implementation of visionOS test rules."""

load(
    "//apple:providers.bzl",
    "AppleBundleInfo",
    "VisionosApplicationBundleInfo",
    "VisionosExtensionBundleInfo",
    "VisionosFrameworkBundleInfo",
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
    "new_visionosxctestbundleinfo",
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

_VISIONOS_TEST_HOST_PROVIDERS = [
    [AppleBundleInfo, VisionosApplicationBundleInfo],
    [AppleBundleInfo, VisionosExtensionBundleInfo],
]

def _visionos_ui_test_bundle_impl(ctx):
    """Implementation of visionos_ui_test."""
    return apple_test_bundle_support.apple_test_bundle_impl(
        ctx = ctx,
        product_type = apple_product_type.ui_test_bundle,
    ) + [
        new_visionosxctestbundleinfo(),
    ]

visibility("//apple/...")

def _visionos_unit_test_bundle_impl(ctx):
    """Implementation of visionos_unit_test."""
    return apple_test_bundle_support.apple_test_bundle_impl(
        ctx = ctx,
        product_type = apple_product_type.unit_test_bundle,
    ) + [
        new_visionosxctestbundleinfo(),
    ]

def _visionos_ui_test_impl(ctx):
    """Implementation of visionos_ui_test."""
    return apple_test_rule_support.apple_test_rule_impl(ctx, "xcuitest") + [
        new_visionosxctestbundleinfo(),
    ]

def _visionos_unit_test_impl(ctx):
    """Implementation of visionos_unit_test."""
    return apple_test_rule_support.apple_test_rule_impl(ctx, "xctest") + [
        new_visionosxctestbundleinfo(),
    ]

# Declare it with an underscore to hint that this is an implementation detail in bazel query-s.
_visionos_internal_ui_test_bundle = rule_factory.create_apple_rule(
    doc = "Builds and bundles an visionOS UI Test Bundle. Internal target not to be depended upon.",
    implementation = _visionos_ui_test_bundle_impl,
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
            allowed_families = rule_attrs.defaults.allowed_families.visionos,
            is_mandatory = False,
        ),
        rule_attrs.infoplist_attrs(
            default_infoplist = rule_attrs.defaults.test_bundle_infoplist,
        ),
        rule_attrs.ipa_post_processor_attrs(),
        rule_attrs.platform_attrs(
            add_environment_plist = True,
            platform_type = "visionos",
        ),
        rule_attrs.signing_attrs(
            default_bundle_id_suffix = bundle_id_suffix_default.bundle_name,
            supports_capabilities = False,
        ),
        rule_attrs.test_bundle_attrs(),
        rule_attrs.test_host_attrs(
            aspects = rule_attrs.aspects.test_host_aspects,
            is_mandatory = True,
            providers = _VISIONOS_TEST_HOST_PROVIDERS,
        ),
        {
            "frameworks": attr.label_list(
                providers = [[AppleBundleInfo, VisionosFrameworkBundleInfo]],
                doc = """
A list of framework targets (see
[`visionos_framework`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-visionos.md#visionos_framework))
that this target depends on.
""",
            ),
        },
    ],
)

# Alias to import it.
visionos_internal_ui_test_bundle = _visionos_internal_ui_test_bundle

visionos_ui_test = rule_factory.create_apple_test_rule(
    doc = """
Builds and bundles a visionOS UI `.xctest` test bundle. Runs the tests using the
provided test runner when invoked with `bazel test`.

Note: visionOS UI tests are not currently supported in the default test runner.

The following is a list of the `visionos_ui_test` specific attributes; for a list of
the attributes inherited by all test rules, please check the
[Bazel documentation](https://bazel.build/reference/be/common-definitions#common-attributes-tests).
""",
    implementation = _visionos_ui_test_impl,
    platform_type = "visionos",
)

# Declare it with an underscore so it shows up that way in queries.
_visionos_internal_unit_test_bundle = rule_factory.create_apple_rule(
    doc = "Builds and bundles an visionOS Unit Test Bundle. Internal target not to be depended upon.",
    implementation = _visionos_unit_test_bundle_impl,
    # TODO(b/288582842): Currently needed to supply a "dummy archive" for the tree artifact
    # processor. See if we can avoid needing to declare this hack for a new rule type.
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
            allowed_families = rule_attrs.defaults.allowed_families.visionos,
            is_mandatory = False,
        ),
        rule_attrs.infoplist_attrs(
            default_infoplist = rule_attrs.defaults.test_bundle_infoplist,
        ),
        rule_attrs.ipa_post_processor_attrs(),
        rule_attrs.platform_attrs(
            add_environment_plist = True,
            platform_type = "visionos",
        ),
        rule_attrs.signing_attrs(
            default_bundle_id_suffix = bundle_id_suffix_default.bundle_name,
            supports_capabilities = False,
        ),
        rule_attrs.test_bundle_attrs(),
        rule_attrs.test_host_attrs(
            aspects = rule_attrs.aspects.test_host_aspects,
            providers = _VISIONOS_TEST_HOST_PROVIDERS,
        ),
        {
            "frameworks": attr.label_list(
                providers = [[AppleBundleInfo, VisionosFrameworkBundleInfo]],
                doc = """
A list of framework targets (see
[`visionos_framework`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-visionos.md#visionos_framework))
that this target depends on.
""",
            ),
        },
    ],
)

# Alias to import it.
visionos_internal_unit_test_bundle = _visionos_internal_unit_test_bundle

visionos_unit_test = rule_factory.create_apple_test_rule(
    doc = """
Builds and bundles a visionOS Unit `.xctest` test bundle. Runs the tests using the
provided test runner when invoked with `bazel test`.

Note: visionOS unit tests are not currently supported in the default test runner.

`visionos_unit_test` targets can work in two modes: as app or library tests. If the
`test_host` attribute is set to an `visionos_application` target, the tests will run
within that application's context. If no `test_host` is provided, the tests will
run outside the context of a visionOS application. Because of this, certain
functionalities might not be present (e.g. UI layout, NSUserDefaults). You can
find more information about app and library testing for Apple platforms
[here](https://developer.apple.com/library/content/documentation/DeveloperTools/Conceptual/testing_with_xcode/chapters/03-testing_basics.html).

The following is a list of the `visionos_unit_test` specific attributes; for a list
of the attributes inherited by all test rules, please check the
[Bazel documentation](https://bazel.build/reference/be/common-definitions#common-attributes-tests).
""",
    implementation = _visionos_unit_test_impl,
    platform_type = "visionos",
)
