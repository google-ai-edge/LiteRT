# Copyright 2022 The Bazel Authors. All rights reserved.
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

"""Common sets of attributes to be shared between the Apple rules."""

load(
    "@bazel_skylib//lib:dicts.bzl",
    "dicts",
)
load(
    "@build_bazel_apple_support//lib:apple_support.bzl",
    "apple_support",
)
load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "SwiftInfo",
)
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load(
    "//apple:common.bzl",
    "entitlements_validation_mode",
)
load(
    "//apple:providers.bzl",
    "AppleBaseBundleIdInfo",
    "AppleBundleVersionInfo",
    "ApplePlatformInfo",
    "AppleResourceBundleInfo",
    "AppleSharedCapabilityInfo",
)
load(
    "//apple/internal:apple_toolchains.bzl",
    "apple_toolchain_utils",
)
load(
    "//apple/internal:bundling_support.bzl",
    "bundle_id_suffix_default",
)
load(
    "//apple/internal/aspects:app_intents_aspect.bzl",
    "app_intents_aspect",
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
    "//apple/internal/aspects:swift_usage_aspect.bzl",
    "swift_usage_aspect",
)
load(
    "//apple/internal/testing:apple_test_bundle_support.bzl",
    "apple_test_info_aspect",
)

def _common_attrs():
    """Private attributes on all rules; these should be included in all rule attributes."""
    return dicts.add(
        {
        },
        apple_support.action_required_attrs(),
    )

def _common_tool_attrs():
    """Returns the set of attributes to support rules that need rules_apple tools and toolchains."""
    return dicts.add(
        _common_attrs(),
        apple_toolchain_utils.shared_attrs(),
    )

def _custom_transition_allowlist_attr():
    """Returns the required attribute to use Starlark defined custom transitions."""
    return {
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    }

def _app_intents_attrs(*, deps_cfg):
    """Returns a dictionary with the attribute for Apple platform rules supporting AppIntents.

    Args:
        deps_cfg: Bazel split transition to use on binary attrs, such as deps and split toolchains.
            To satisfy native Bazel linking prerequisites, `deps` and this `deps_cfg` attribute must
            use the same transition.
    """
    return {
        "app_intents": attr.label_list(
            doc = "List of dependencies implementing the AppIntents protocol.",
            cfg = deps_cfg,
            aspects = [app_intents_aspect],
            providers = [SwiftInfo],
        ),
    }

def _common_linking_api_attrs(*, deps_cfg):
    """Returns dictionary of required attributes for Bazel Apple linking APIs.

    These rule attributes are required by _link_multi_arch_binary and
    _archive_multi_arch_static_library in rules_apple/apple/internal/linking_support.bzl.

    Args:
        deps_cfg: Bazel split transition to use on binary attrs, such as deps and split toolchains.
            To satisfy native Bazel linking prerequisites, `deps` and this `deps_cfg` attribute must
            use the same transition.
    """
    return dicts.add(_common_attrs(), {
        "_cc_toolchain_forwarder": attr.label(
            cfg = deps_cfg,
            providers = [cc_common.CcToolchainInfo, ApplePlatformInfo],
            default = "//apple:default_cc_toolchain_forwarder",
        ),
        # TODO(b/251837356): Eliminate this attr when the Bazel Starlark API
        # `apple_common.get_split_build_configs(...)`` doesn't require that this attr be called
        # `_child_configuration_dummy` in Apple multiarch APIs.
        "_child_configuration_dummy": attr.label(
            cfg = deps_cfg,
            providers = [cc_common.CcToolchainInfo, ApplePlatformInfo],
            default = "//apple:default_cc_toolchain_forwarder",
        ),
    })

def _static_library_archive_attrs(*, deps_cfg):
    """Returns dictionary of required attributes for apple_common.link_multi_arch_static_library.

    Args:
        deps_cfg: Bazel split transition to use on binary attrs, such as deps and split toolchains.
            To satisfy native Bazel linking prerequisites, `deps` and this `deps_cfg` attribute must
            use the same transition.
    """
    return _common_linking_api_attrs(deps_cfg = deps_cfg)

def _binary_linking_attrs(
        *,
        deps_cfg,
        extra_deps_aspects = [],
        is_test_supporting_rule,
        requires_legacy_cc_toolchain):
    """Returns dictionary of required attributes for apple_common.link_multi_arch_binary.

    Args:
        deps_cfg: Bazel split transition to use on binary attrs, such as deps and split toolchains.
            To satisfy native Bazel linking prerequisites, `deps` and this `deps_cfg` attribute must
            use the same transition.
        extra_deps_aspects: A list of aspects to apply to deps in addition to the defaults.
            Optional.
        is_test_supporting_rule: Boolean. Indicates if this rule is intended to be used as a
            dependency of a test rule, such as a test bundle rule generated by an ios_unit_test
            macro.
        requires_legacy_cc_toolchain: Boolean. Indicates if the rule is expected to have an
            attribute referencing the old-style cc_toolchain_suite to build Apple binary rules.
    """
    deps_aspects = [
        swift_usage_aspect,
    ]
    deps_aspects.extend(extra_deps_aspects)
    extra_attrs = {}

    default_stamp = -1
    if is_test_supporting_rule:
        deps_aspects.append(apple_test_info_aspect)
        default_stamp = 0

    if requires_legacy_cc_toolchain:
        # This attribute is required by the Clang runtime libraries processing partial.
        # See utils/clang_rt_dylibs.bzl and partials/clang_rt_dylibs.bzl
        extra_attrs = dicts.add(extra_attrs, {
            "_cc_toolchain": attr.label(
                default = Label("@bazel_tools//tools/cpp:current_cc_toolchain"),
            ),
        })

    return dicts.add(
        extra_attrs,
        _common_linking_api_attrs(deps_cfg = deps_cfg),
        {
            "codesign_inputs": attr.label_list(
                doc = """
A list of dependencies targets that provide inputs that will be used by
`codesign` (referenced with `codesignopts`).
""",
            ),
            "codesignopts": attr.string_list(
                doc = """
A list of strings representing extra flags that should be passed to `codesign`.
""",
            ),
            "exported_symbols_lists": attr.label_list(
                allow_files = True,
                doc = """
A list of targets containing exported symbols lists files for the linker to control symbol
resolution.

Each file is expected to have a list of global symbol names that will remain as global symbols in
the compiled binary owned by this framework. All other global symbols will be treated as if they
were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output
file.

See the man page documentation for `ld(1)` on macOS for more details.
""",
            ),
            "linkopts": attr.string_list(
                doc = """
A list of strings representing extra flags that should be passed to the linker.
    """,
            ),
            "additional_linker_inputs": attr.label_list(
                allow_files = True,
                doc = """
A list of input files to be passed to the linker.
    """,
            ),
            "stamp": attr.int(
                default = default_stamp,
                doc = """
Enable link stamping. Whether to encode build information into the binary. Possible values:

*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt
    when their dependencies change. Use this if there are tests that depend on the build
    information.
*   `stamp = 0`: Always replace build information by constant values. This gives good build
    result caching.
*   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.
""",
                values = [-1, 0, 1],
            ),
            "deps": attr.label_list(
                aspects = deps_aspects,
                cfg = deps_cfg,
                doc = """
A list of dependent targets that will be linked into this target's binary(s). Any resources, such as
asset catalogs, that are referenced by those targets will also be transitively included in the final
bundle(s).
""",
            ),
        },
    )

def _platform_attrs(*, platform_type = None, add_environment_plist = False):
    """Returns a dictionary for rules that must know about the Apple platform being targeted.

    Args:
        platform_type: A string value to indicate the default `platform_type`. If none is given, the
            `platform_type` attribute will be assumed mandatory and must be given by the user.
        add_environment_plist: Adds the private `_environment_plist` attribute for the given
            `platform_type`. This requires that `platform_type` is defined by the rule and not the
            user; if the `platform_type` argument is not set, this will raise an internal error.
    """
    platform_attrs = {
        # Minimum OS version is treated as mandatory on all Apple rules. This should always be given
        # by the user or assigned by a macro on platform-aware rules.
        "minimum_os_version": attr.string(
            doc = """
A required string indicating the minimum OS version supported by the target, represented as a
dotted version number (for example, "9.0").
""",
            mandatory = True,
        ),
        "minimum_deployment_os_version": attr.string(
            mandatory = False,
            doc = """
A required string indicating the minimum deployment OS version supported by the target, represented
as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which
is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.
""",
        ),
    }

    if platform_type:
        # If platform_type is given when the rule is constructed, treat the platform_type attribute
        # as predefined and leave it undocumented. In this sense, the platform_type is treated as
        # immutable and an implementation detail of linking even though it is not a private attr.
        platform_attrs = dicts.add(platform_attrs, {
            "platform_type": attr.string(default = platform_type),
        })
    else:
        # Otherwise, require the user of the rule to define the platform_type.
        platform_attrs = dicts.add(platform_attrs, {
            "platform_type": attr.string(
                doc = """
The target Apple platform for which to create a binary. This dictates which SDK
is used for compilation/linking and which flag is used to determine the
architectures to target. For example, if `ios` is specified, then the output
binaries/libraries will be created combining all architectures specified by
`--ios_multi_cpus`. Options are:

*   `ios`: architectures gathered from `--ios_multi_cpus`.
*   `macos`: architectures gathered from `--macos_cpus`.
*   `tvos`: architectures gathered from `--tvos_cpus`.
*   `watchos`: architectures gathered from `--watchos_cpus`.
""",
                mandatory = True,
            ),
        })
    if add_environment_plist:
        if not platform_type:
            fail("Internal Error: An environment plist attribute requires a platform_type to be " +
                 "defined by the rule definition.")
        platform_attrs = dicts.add(platform_attrs, {
            "_environment_plist": attr.label(
                allow_single_file = True,
                default = "//apple/internal:environment_plist_{}".format(
                    platform_type,
                ),
            ),
        })
    return platform_attrs

def _test_bundle_attrs():
    """Attributes required for rules that are built to support test rules like ios_unit_test."""
    return {
        # We need to add an explicit output attribute so that the output file name from the test
        # bundle target matches the test name, otherwise, it we'd be breaking the assumption that
        # ios_unit_test(name = "Foo") creates a :Foo.zip target.
        # This is an implementation detail attribute, so it's not documented on purpose.
        "test_bundle_output": attr.output(mandatory = True),
        "_swizzle_absolute_xcttestsourcelocation": attr.label(
            default = Label(
                "@build_bazel_apple_support//lib:swizzle_absolute_xcttestsourcelocation",
            ),
        ),
    }

def _test_host_attrs(
        *,
        aspects,
        is_mandatory = False,
        providers):
    """Returns a dictionary of required attributes to handle the test host.

    Args:
        aspects: A list of aspects to apply to the test_host attribute.
        is_mandatory: Bool to indicate if the test_host should be marked as mandatory such that the
            test_host must be given explicitly by the user.
        providers: A list of lists of providers. The resolved test_host must conform to all of the
            providers mentioned in at least one of the inner lists, as the `providers` attribute on
            `attr.label` requires. Can also be expressed as a list of providers for convenience, as
            `attr.label`'s `providers` attribute allows.
    """
    return {
        "test_host": attr.label(
            aspects = aspects,
            mandatory = is_mandatory,
            providers = providers,
        ),
        "test_host_is_bundle_loader": attr.bool(
            default = True,
            doc = """
Whether the 'test_host' should be used as the -bundle_loader to allow testing
the symbols from the test host app
""",
        ),
    }

def _signing_attrs(
        *,
        default_bundle_id_suffix = bundle_id_suffix_default.no_suffix,
        supports_capabilities = True,
        profile_extension = ".mobileprovision"):
    """Returns the attribute required to support custom bundle identifiers for the given target.

    Args:
        default_bundle_id_suffix: String. A value from the bundle_id_suffix_default struct that will
            be used to indicate the default source for the bundle ID suffix, if one is composed from
            a base bundle ID.
        supports_capabilities: Boolean. Indicates if the attributes generated should support
            entitlements and capabilities for code signing.
        profile_extension: A file extension that will be required for the provisioning profile.
            Optional. This is `.mobileprovision` by default.
    """
    signing_attrs = {
        "_bundle_id_suffix_default": attr.string(
            default = default_bundle_id_suffix,
            doc = """
An internally-defined string to indicate the default suffix of the bundle ID, independent from the
user-specified `bundle_id_suffix`. This acts as a constraint to force the _signing_attrs-assigned
bundle_id_suffix_default arg to be sourced from a common struct, and allows for better handling
special cases like sourcing the bundle name or handling a no-suffix case.
""",
            values = [
                bundle_id_suffix_default.no_suffix,
                bundle_id_suffix_default.bundle_name,
                bundle_id_suffix_default.watchos_app,
                bundle_id_suffix_default.watchos2_app_extension,
            ],
        ),
        "bundle_id": attr.string(
            doc = """
The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if
the bundle ID is not intended to be composed through an assigned base bundle ID rule found within
`signed_capabilities`.
""" if supports_capabilities else """
The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if
the bundle ID is not intended to be composed through an assigned base bundle ID referenced by
`base_bundle_id`.
""",
        ),
        "bundle_id_suffix": attr.string(
            default = default_bundle_id_suffix,
            doc = """
A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from
a base bundle ID rule found within `signed_capabilities`, then this string will be appended to the
end of the bundle ID following a "." separator.
""" if supports_capabilities else """
A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from
the base bundle ID rule referenced by `base_bundle_id`, then this string will be appended to the end
of the bundle ID following a "." separator.
""",
        ),
        "provisioning_profile": attr.label(
            allow_single_file = [profile_extension],
            doc = """
The provisioning profile (`{profile_extension}` file) to use when creating the bundle. This value is
optional for simulator builds as the simulator doesn't fully enforce entitlements, but is
required for device builds.
""".format(profile_extension = profile_extension),
        ),
    }
    if supports_capabilities:
        signing_attrs = dicts.add(signing_attrs, {
            "entitlements": attr.label(
                allow_single_file = True,
                doc = """
The entitlements file required for device builds of this target. If absent, the default entitlements
from the provisioning profile will be used.

The following variables are substituted in the entitlements file: `$(CFBundleIdentifier)` with the
bundle ID of the application and `$(AppIdentifierPrefix)` with the value of the
`ApplicationIdentifierPrefix` key from the target's provisioning profile.
""",
            ),
            "entitlements_validation": attr.string(
                default = entitlements_validation_mode.loose,
                doc = """
An `entitlements_validation_mode` to control the validation of the requested entitlements against
the provisioning profile to ensure they are supported.
""",
                values = [
                    entitlements_validation_mode.error,
                    entitlements_validation_mode.warn,
                    entitlements_validation_mode.loose,
                    entitlements_validation_mode.skip,
                ],
            ),
            "shared_capabilities": attr.label_list(
                providers = [[AppleSharedCapabilityInfo]],
                doc = """
A list of shared `apple_capability_set` rules to represent the capabilities that a code sign aware
Apple bundle rule output should have. These can define the formal prefix for the target's
`bundle_id` and can further be merged with information provided by `entitlements`, if defined by any
capabilities found within the `apple_capability_set`.
""",
            ),
        })
    else:
        signing_attrs = dicts.add(signing_attrs, {
            "base_bundle_id": attr.label(
                providers = [[AppleBaseBundleIdInfo]],
                doc = """
The base bundle ID rule to dictate the form that a given bundle rule's bundle ID prefix should take.
""",
            ),
        })

    return signing_attrs

def _infoplist_attrs(*, default_infoplist = None):
    """Returns the attributes required to support a root Info.plist for the given target.

    Args:
        default_infoplist: A string representing a label to a default Info.plist. Optional. If not
            set or if set to None, the `infoplists` attribute will be considered mandatory for the
            user to set.
    """
    attr_args = {}
    if default_infoplist:
        attr_args["default"] = [Label(default_infoplist)]
    else:
        attr_args["mandatory"] = True
    return {
        "infoplists": attr.label_list(
            allow_empty = False,
            allow_files = [".plist"],
            doc = """
A list of .plist files that will be merged to form the Info.plist for this target. At least one file
must be specified. Please see
[Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling)
for what is supported.
""",
            **attr_args
        ),
    }

def _common_bundle_attrs(*, deps_cfg):
    """Returns a dictionary of rule attributes common to all rules that produce Apple bundles."""
    return {
        "bundle_name": attr.string(
            mandatory = False,
            doc = """
The desired name of the bundle (without the extension). If this attribute is not set, then the name
of the target will be used instead.
""",
        ),
        "executable_name": attr.string(
            mandatory = False,
            doc = """
The desired name of the executable, if the bundle has an executable. If this attribute is not set,
then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of
the target will be used instead.
""",
        ),
        "strings": attr.label_list(
            allow_files = [".strings"],
            doc = """
A list of `.strings` files, often localizable. These files are converted to binary plists (if they
are not already) and placed in the root of the final bundle, unless a file's immediate containing
directory is named `*.lproj`, in which case it will be placed under a directory with the same name
in the bundle.
""",
        ),
        "resources": attr.label_list(
            allow_files = True,
            aspects = [apple_resource_aspect],
            cfg = deps_cfg,
            doc = """
A list of resources or files bundled with the bundle. The resources will be stored in the
appropriate resources location within the bundle.
""",
        ),
        "version": attr.label(
            providers = [[AppleBundleVersionInfo]],
            doc = """
An `apple_bundle_version` target that represents the version for this target. See
[`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).
""",
        ),
    }

def _device_family_attrs(*, allowed_families, is_mandatory = False):
    """Returns the attributes required to define device families for resource assets.

    Args:
        allowed_families: List of strings representing valid device families to compile assets
            compatible for the given target.
        is_mandatory: Boolean. If `True`, the `families` attribute must be set by the user.
            Optional.
    """
    extra_args = {}
    if not is_mandatory:
        extra_args["default"] = allowed_families
    return {
        "families": attr.string_list(
            mandatory = is_mandatory,
            allow_empty = False,
            doc = """
A list of device families supported by this rule. At least one must be specified.
""",
            **extra_args
        ),
    }

def _extensionkit_attrs():
    """Returns the attributes required to define a target as an ExtensionKit extension."""
    return {
        "extensionkit_extension": attr.bool(
            doc = "Indicates if this target should be treated as an ExtensionKit extension.",
        ),
    }

def _ipa_post_processor_attrs():
    """Returns the attributes required to support the deprecated ipa_post_processor feature."""

    # TODO(b/36512239): Rename to "bundle_post_processor", if we have to continue supporting this.
    return {
        "ipa_post_processor": attr.label(
            allow_files = True,
            executable = True,
            cfg = "exec",
            doc = """
A tool that edits this target's archive after it is assembled but before it is signed. The tool is
invoked with a single command-line argument that denotes the path to a directory containing the
unzipped contents of the archive; this target's bundle will be the directory's only contents.

Any changes made by the tool must be made in this directory, and the tool's execution must be
hermetic given these inputs to ensure that the result can be safely cached.
""",
        ),
    }

def _app_icon_attrs(
        *,
        icon_extension = ".appiconset",
        icon_parent_extension = ".xcassets",
        supports_alternate_icons = False):
    """Returns the attribute required to define app icons for the given target.

    Args:
        icon_extension: A String representing the extension required of the directory containing the
            app icon assets. Optional. Defaults to `.appiconset`.
        icon_parent_extension: A String representing the extension required of the parent directory
            of the directory containing the app icon assets. Optional. Defaults to `.xcassets`.
        supports_alternate_icons: Bool representing if the rule supports alternate icons. False by
            default.
    """
    app_icon_attrs = {
        "app_icons": attr.label_list(
            allow_files = True,
            doc = """
Files that comprise the app icons for the application. Each file must have a containing directory
named `*.{app_icon_parent_extension}/*.{app_icon_extension}` and there may be only one such
`.{app_icon_extension}` directory in the list. In Xcode 26+ for iOS/macOS/watchOS, an `*.icon`
bundle can be provided along with the `*.{app_icon_parent_extension}` bundle to support 26 and
pre-26 Apple OS rendering.""".format(
                app_icon_extension = icon_extension,
                app_icon_parent_extension = icon_parent_extension,
            ),
        ),
    }
    if supports_alternate_icons:
        app_icon_attrs = dicts.add(app_icon_attrs, {
            "primary_app_icon": attr.string(
                doc = """
An optional String to identify the name of the primary app icon when alternate app icons have been
provided for the app. This should match both the `*.icon` bundle in iOS/macOS/watchOS 26+ and the
`*.{app_icon_parent_extension}/.appiconset` bundle's AppIcon resource in previous versions of
iOS/macOS/watchOS.
""".format(
                    app_icon_parent_extension = icon_parent_extension,
                ),
            ),
        })
    return app_icon_attrs

def _launch_images_attrs():
    """Returns the attribute required to support launch images for a given target."""
    return {
        "launch_images": attr.label_list(
            allow_files = True,
            doc = """
Files that comprise the launch images for the application. Each file must have a containing
directory named `*.xcassets/*.launchimage` and there may be only one such `.launchimage` directory
in the list.
""",
        ),
    }

def _settings_bundle_attrs():
    """Returns the attribute required to support settings bundles for a given target."""
    return {
        "settings_bundle": attr.label(
            aspects = [apple_resource_aspect],
            providers = [[AppleResourceBundleInfo], [apple_common.Objc]],
            doc = """
A resource bundle (e.g. `apple_bundle_import`) target that contains the files that make up the
application's settings bundle. These files will be copied into the root of the final application
bundle in a directory named `Settings.bundle`.
""",
        ),
    }

def _simulator_runner_template_attr():
    """Returns the attribute required to `bazel run` a *_application target with an Apple sim."""
    return {
        "_simulator_runner_template": attr.label(
            cfg = "exec",
            allow_single_file = True,
            default = Label(
                "//apple/internal/templates:apple_simulator_template",
            ),
        ),
    }

def _device_runner_template_attr():
    """Returns the attribute required to `bazel run` a *_application target on a physical device."""
    return {
        "_device_runner_template": attr.label(
            cfg = "exec",
            allow_single_file = True,
            default = Label(
                "//apple/internal/templates:apple_device_template",
            ),
        ),
    }

def _locales_to_include_attr():
    """Returns the attribute required to support configuring the explicit set of locales supported for the bundle."""
    return {
        "locales_to_include": attr.string_list(
            mandatory = False,
            doc = """
A list of locales to include in the bundle. Only *.lproj directories that are matched will be copied as a part of the build.
This value takes precedence (and is preferred) over locales defined using `--define "apple.locales_to_include=..."`.
""",
        ),
    }

# Returns the aspects required to support a test host for a given target.
_TEST_HOST_ASPECTS = [framework_provider_aspect]

# Returns the default root Info.plist required to support a test bundle rule.
_test_bundle_infoplist = "//apple/testing:DefaultTestBundlePlist"

rule_attrs = struct(
    app_icon_attrs = _app_icon_attrs,
    app_intents_attrs = _app_intents_attrs,
    aspects = struct(test_host_aspects = _TEST_HOST_ASPECTS),
    binary_linking_attrs = _binary_linking_attrs,
    common_attrs = _common_attrs,
    common_bundle_attrs = _common_bundle_attrs,
    common_tool_attrs = _common_tool_attrs,
    custom_transition_allowlist_attr = _custom_transition_allowlist_attr,
    device_family_attrs = _device_family_attrs,
    device_runner_template_attr = _device_runner_template_attr,
    extensionkit_attrs = _extensionkit_attrs,
    infoplist_attrs = _infoplist_attrs,
    ipa_post_processor_attrs = _ipa_post_processor_attrs,
    locales_to_include_attrs = _locales_to_include_attr,
    launch_images_attrs = _launch_images_attrs,
    platform_attrs = _platform_attrs,
    settings_bundle_attrs = _settings_bundle_attrs,
    signing_attrs = _signing_attrs,
    simulator_runner_template_attr = _simulator_runner_template_attr,
    static_library_archive_attrs = _static_library_archive_attrs,
    test_bundle_attrs = _test_bundle_attrs,
    test_host_attrs = _test_host_attrs,
    defaults = struct(
        allowed_families = struct(
            ios = ["iphone", "ipad"],
            macos = ["mac"],
            tvos = ["tv"],
            visionos = ["vision"],
            watchos = ["watch"],
        ),
        test_bundle_infoplist = _test_bundle_infoplist,
    ),
)
