load(
    "//apple:capabilities.bzl",
    "apple_base_bundle_id",
    "apple_capability_set",
)
load(
    "//test/starlark_tests:common.bzl",
    "common",
)

licenses(["notice"])

package(
    default_testonly = True,
    default_visibility = ["//test/starlark_tests:__subpackages__"],
)

# Targets for base_bundle_id flows with and without shared capabilities.

apple_base_bundle_id(
    name = "base_bundle_id",
    organization_id = "com.bazel.app",
    tags = common.fixture_tags,
    variant_name = "example",
)

apple_base_bundle_id(
    name = "alt_base_bundle_id",
    organization_id = "com.altbazel.app",
    tags = common.fixture_tags,
    variant_name = "example",
)

apple_base_bundle_id(
    name = "empty_base_bundle_id",
    organization_id = "",
    tags = common.fixture_tags,
)

apple_base_bundle_id(
    name = "just_dot_base_bundle_id",
    organization_id = ".",
    tags = common.fixture_tags,
)

apple_base_bundle_id(
    name = "leading_dot_base_bundle_id",
    organization_id = ".com.bazel.app",
    tags = common.fixture_tags,
)

apple_base_bundle_id(
    name = "trailing_dot_base_bundle_id",
    organization_id = "com.bazel.app.",
    tags = common.fixture_tags,
)

apple_base_bundle_id(
    name = "double_dot_base_bundle_id",
    organization_id = "com..bazel.app",
    tags = common.fixture_tags,
)

apple_base_bundle_id(
    name = "invalid_character_bundle_id",
    organization_id = "com#bazel.app",
    tags = common.fixture_tags,
)

apple_capability_set(
    name = "base_capability_set",
    base_bundle_id = ":base_bundle_id",
    tags = common.fixture_tags,
)

apple_capability_set(
    name = "alt_capability_set",
    base_bundle_id = ":alt_base_bundle_id",
    tags = common.fixture_tags,
)

apple_capability_set(
    name = "empty_capability_set",
    tags = common.fixture_tags,
)
