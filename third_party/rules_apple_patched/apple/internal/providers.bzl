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

"""Defines providers and related types used throughout the bundling rules.

These providers are part of the public API of the bundling rules. The symbols are re-exported in
public space to allow for writing rules that can reference the contents of these providers, but the
initialization via each provider's "raw initializer" is gated to the internal rules implementation.

Public initializers must be defined in apple:providers.bzl instead of apple/internal:providers.bzl.
These should build from the raw initializer where possible, but not export it, to allow for a safe
boundary with well-defined public APIs for broader usage.
"""

def _make_banned_init(*, preferred_public_factory = None, provider_name):
    """Generates a lambda with a fail(...) for providers to dictate preferred initializer patterns.

    Args:
        preferred_public_factory: Optional. An `apple_provider` prefixed public factory method for
            users of the provider to call instead, if one exists.
        provider_name: The name of the provider to reference in error messaging.

    Returns:
        A lambda with a fail(...) for providers that can't be publicly initialized, or which must
        recommend an alternative public interface.
    """
    if preferred_public_factory:
        return lambda *kwargs: fail("""
{provider} is a provider that must be initialized through apple_provider.{preferred_public_factory}
""".format(
            provider = provider_name,
            preferred_public_factory = preferred_public_factory,
        ))
    return lambda *kwargs: fail("""
%s is not a provider that is intended to be publicly initialized.

Please file an issue with the Apple BUILD rules if you would like a public API for this provider.
""" % provider_name)

AppleBaseBundleIdInfo, new_applebasebundleidinfo = provider(
    doc = "Provides the base bundle ID prefix for an Apple rule.",
    fields = {
        "base_bundle_id": """
`String`. The bundle ID prefix, composed from an organization ID and an optional variant name.
""",
    },
    init = _make_banned_init(provider_name = "AppleBaseBundleIdInfo"),
)

AppleBinaryInfo, new_applebinaryinfo = provider(
    doc = """
Provides information about an Apple binary target.

This provider propagates general information about an Apple binary that is not
specific to any particular binary type.
""",
    fields = {
        "binary": """
`File`. The binary (executable, dynamic library, etc.) file that the target represents.
""",
        "infoplist": """
`File`. The complete (binary-formatted) `Info.plist` embedded in the binary.
""",
        "product_type": """
`String`. The dot-separated product type identifier associated with the binary (for example,
`com.apple.product-type.tool`).
""",
    },
    init = _make_banned_init(provider_name = "AppleBinaryInfo"),
)

AppleBundleInfo, new_applebundleinfo = provider(
    doc = """
This provider propagates general information about an Apple bundle that is not
specific to any particular bundle type. It is propagated by most bundling
rules (applications, extensions, frameworks, test bundles, and so forth).
""",
    fields = {
        "archive": "`File`. The archive that contains the built bundle.",
        "archive_root": """
`String`. The file system path (relative to the workspace root) where the signed
bundle was constructed (before archiving). Other rules **should not** depend on
this field; it is intended to support IDEs that want to read that path from the
provider to avoid performance issues from unzipping the output archive.
""",
        "binary": """
`File`. The binary (executable, dynamic library, etc.) that was bundled. The
physical file is identical to the one inside the bundle except that it is
always unsigned, so note that it is _not_ a path to the binary inside your
output bundle. The primary purpose of this field is to provide a way to access
the binary directly at analysis time; for example, for code coverage.
""",
        "bundle_extension": """
`String`. The bundle extension.
""",
        "bundle_id": """
`String`. The bundle identifier (i.e., `CFBundleIdentifier` in
`Info.plist`) of the bundle.
""",
        "bundle_name": """
`String`. The name of the bundle, without the extension.
""",
        "entitlements": "`File`. Entitlements file used, if any.",
        "executable_name": """
`string`. The name of the executable that was bundled.
""",
        "extension_safe": """
`Boolean`. True if the target propagating this provider was
compiled and linked with -application-extension, restricting it to
extension-safe APIs only.
""",
        "infoplist": """
`File`. The complete (binary-formatted) `Info.plist` file for the bundle.
""",
        "minimum_deployment_os_version": """
`string`. The minimum deployment OS version (as a dotted version
number like "9.0") that this bundle was built to support. This is different from
`minimum_os_version`, which is effective at compile time. Ensure version
specific APIs are guarded with `available` clauses.
""",
        "minimum_os_version": """
`String`. The minimum OS version (as a dotted version
number like "9.0") that this bundle was built to support.
""",
        "platform_type": """
`String`. The platform type for the bundle (i.e. `ios` for iOS bundles).
""",
        "product_type": """
`String`. The dot-separated product type identifier associated
with the bundle (for example, `com.apple.product-type.application`).
""",
        "uses_swift": """
Boolean. True if Swift is used by the target propagating this
provider. This does not consider embedded bundles; for example, an
Objective-C application containing a Swift extension would have this field
set to true for the extension but false for the application.
""",
    },
    init = _make_banned_init(provider_name = "AppleBundleInfo"),
)

AppleBundleVersionInfo, new_applebundleversioninfo = provider(
    doc = "Provides versioning information for an Apple bundle.",
    fields = {
        "version_file": """
Required. A `File` containing JSON-formatted text describing the version number information
propagated by the target.

It contains two keys:

*   `build_version`, which corresponds to `CFBundleVersion`.

*   `short_version_string`, which corresponds to `CFBundleShortVersionString`.
""",
    },
    init = _make_banned_init(
        provider_name = "AppleBundleVersionInfo",
        preferred_public_factory = "make_apple_bundle_version_info(...)",
    ),
)

def make_apple_bundle_version_info(*, version_file):
    """Creates a new instance of the `AppleBundleVersionInfo` provider.

    Args:
        version_file: Required. See the docs on `AppleBundleVersionInfo`.

    Returns:
        A new `AppleBundleVersionInfo` provider based on the supplied arguments.
    """
    if type(version_file) != "File":
        fail("""
Error: Expected "version_file" to be of type "File".

Received unexpected type "{actual_type}".
""".format(actual_type = type(version_file)))

    return new_applebundleversioninfo(version_file = version_file)

AppleCodesigningDossierInfo, new_applecodesigningdossierinfo = provider(
    doc = "Provides information around the use of a code signing dossier.",
    fields = {
        "dossier": """
A `File` reference to the code signing dossier zip that acts as a direct dependency of the given
target if one was generated.
""",
    },
    init = _make_banned_init(provider_name = "AppleCodesigningDossierInfo"),
)

AppleDebugOutputsInfo, new_appledebugoutputsinfo = provider(
    """
Holds debug outputs of an Apple binary rule.

This provider is DEPRECATED. Preferably use `AppleDsymBundleInfo` instead.

The only field is `output_map`, which is a dictionary of:
  `{ arch: { "dsym_binary": File, "linkmap": File }`

Where `arch` is any Apple architecture such as "arm64" or "armv7".
""",
    fields = ["outputs_map"],
    init = _make_banned_init(provider_name = "AppleDebugOutputsInfo"),
)

AppleDsymBundleInfo, new_appledsymbundleinfo = provider(
    doc = "Provides information for an Apple dSYM bundle.",
    fields = {
        "direct_dsyms": """
`List` containing `File` references to each of the dSYM bundles that act as direct dependencies of
the given target if any were generated.
""",
        "transitive_dsyms": """
`depset` containing `File` references to each of the dSYM bundles that act as transitive
dependencies of the given target if any were generated.
""",
    },
    init = _make_banned_init(provider_name = "AppleDsymBundleInfo"),
)

_AppleDynamicFrameworkInfo = provider(
    doc = "Contains information about an Apple dynamic framework.",
    fields = {
        "framework_dirs": """\
The framework path names used as link inputs in order to link against the
dynamic framework.
""",
        "framework_files": """\
The full set of artifacts that should be included as inputs to link against the
dynamic framework.
""",
        "binary": "The dylib binary artifact of the dynamic framework.",
        "cc_info": """\
A `CcInfo` which contains information about the transitive dependencies linked
into the binary.
""",
    },
)

# TODO: Remove when we drop 7.x
AppleDynamicFrameworkInfo = getattr(
    apple_common,
    "AppleDynamicFramework",
    _AppleDynamicFrameworkInfo,
)

# TODO: Remove when we drop 7.x
def new_appledynamicframeworkinfo(**kwargs):
    legacy_initializer = getattr(
        apple_common,
        "new_dynamic_framework_provider",
        None,
    )
    if legacy_initializer:
        return legacy_initializer(**kwargs)

    return AppleDynamicFrameworkInfo(**kwargs)

_AppleExecutableBinaryInfo = provider(
    doc = """
Contains the executable binary output that was built using
`link_multi_arch_binary` with the `executable` binary type.
""",
    fields = {
        # TODO: Remove when we drop 7.x
        "objc": """\
apple_common.Objc provider used for legacy linking behavior.
""",
        "binary": """\
The executable binary artifact output by `link_multi_arch_binary`.
""",
        "cc_info": """\
A `CcInfo` which contains information about the transitive dependencies linked
into the binary.
""",
    },
)

AppleExecutableBinaryInfo = getattr(apple_common, "AppleExecutableBinary", _AppleExecutableBinaryInfo)

# TODO: Use common init pattern when we drop 7.x
def new_appleexecutablebinaryinfo(**kwargs):
    legacy_initializer = getattr(apple_common, "new_executable_binary_provider", None)
    if legacy_initializer:
        return legacy_initializer(**kwargs)

    return AppleExecutableBinaryInfo(**kwargs)

AppleExtraOutputsInfo, new_appleextraoutputsinfo = provider(
    doc = """
Provides information about extra outputs that should be produced from the build.

This provider propagates supplemental files that should be produced as outputs
even if the bundle they are associated with is not a direct output of the rule.
For example, an application that contains an extension will build both targets
but only the application will be a rule output. However, if dSYM bundles are
also being generated, we do want to produce the dSYMs for *both* application and
extension as outputs of the build, not just the dSYMs of the explicit target
being built (the application).
""",
    fields = {
        "files": """
`depset` of `File`s. These files will be propagated from embedded bundles (such
as frameworks and extensions) to the top-level bundle (such as an application)
to ensure that they are explicitly produced as outputs of the build.
""",
    },
    init = _make_banned_init(provider_name = "AppleExtraOutputsInfo"),
)

AppleFrameworkBundleInfo, new_appleframeworkbundleinfo = provider(
    doc = """
Denotes a target is an Apple framework bundle.

This provider does not reference 3rd party or precompiled frameworks.
Propagated by Apple framework rules: `ios_framework`, and `tvos_framework`.
""",
    fields = {},
    init = _make_banned_init(provider_name = "AppleFrameworkBundleInfo"),
)

AppleFrameworkImportInfo, new_appleframeworkimportinfo = provider(
    doc = """
Provider that propagates information about 3rd party imported framework targets.

Propagated by framework and XCFramework import rules: `apple_dynamic_framework_import`,
`apple_dynamic_xcframework_import`, `apple_static_framework_import`, and
`apple_static_xcframework_import`
""",
    fields = {
        "framework_imports": """
`depset` of `File`s that represent framework imports that need to be bundled in the top level
application bundle under the Frameworks directory.
""",
        "dsym_imports": """
Depset of Files that represent dSYM imports that need to be processed to
provide .symbols files for packaging into the .ipa file if requested in the
build with --define=apple.package_symbols=(yes|true|1).
""",
        "build_archs": """
`depset` of `String`s that represent binary architectures reported from the current build.
""",
        "debug_info_binaries": """
Depset of Files that represent framework binaries and dSYM binaries that
provide debug info.
""",
    },
    init = _make_banned_init(provider_name = "AppleFrameworkImportInfo"),
)

def merge_apple_framework_import_info(apple_framework_import_infos):
    """Merges multiple `AppleFrameworkImportInfo` into one.

    Args:
        apple_framework_import_infos: List of `AppleFrameworkImportInfo` to be merged.

    Returns:
        Result of merging all the received framework infos.
    """
    transitive_debug_info_binaries = []
    transitive_dsyms = []
    transitive_sets = []
    build_archs = []

    for framework_info in apple_framework_import_infos:
        if hasattr(framework_info, "debug_info_binaries"):
            transitive_debug_info_binaries.append(framework_info.debug_info_binaries)
        if hasattr(framework_info, "dsym_imports"):
            transitive_dsyms.append(framework_info.dsym_imports)
        if hasattr(framework_info, "framework_imports"):
            transitive_sets.append(framework_info.framework_imports)
        build_archs.append(framework_info.build_archs)

    return new_appleframeworkimportinfo(
        debug_info_binaries = depset(transitive = transitive_debug_info_binaries),
        dsym_imports = depset(transitive = transitive_dsyms),
        framework_imports = depset(transitive = transitive_sets),
        build_archs = depset(transitive = build_archs),
    )

ApplePlatformInfo, new_appleplatforminfo = provider(
    doc = "Provides information for the currently selected Apple platforms.",
    fields = {
        "target_os": """
`String` representing the selected Apple OS.
""",
        "target_arch": """
`String` representing the selected target architecture or cpu type.
""",
        "target_environment": """
`String` representing the selected target environment (e.g. "device", "simulator").
""",
    },
    init = _make_banned_init(provider_name = "ApplePlatformInfo"),
)

AppleResourceBundleInfo, new_appleresourcebundleinfo = provider(
    doc = """
Denotes that a target is an Apple resource bundle.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an Apple resource bundle
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is an Apple resource bundle should use this provider to describe that
requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "AppleResourceBundleInfo"),
)

AppleResourceInfo, new_appleresourceinfo = provider(
    doc = "Provider that propagates buckets of resources that are differentiated by type.",
    # @unsorted-dict-items
    fields = {
        "alternate_icons": "Alternate icons to be included in the App bundle.",
        "asset_catalogs": "Resources that need to be embedded into Assets.car.",
        "datamodels": "Datamodel files.",
        "framework": "Apple framework bundle from `ios_framework` and `tvos_framework` targets.",
        "infoplists": """Plist files to be merged and processed. Plist files that should not be \
merged into the root Info.plist should be propagated in `plists`. Because of this, infoplists \
should only be bucketed with the `bucketize_typed` method.""",
        "metals": """Metal Shading Language source files to be compiled into a single .metallib file
and bundled at the top level.""",
        "mlmodels": "Core ML model files that should be processed and bundled at the top level.",
        "plists": "Resource Plist files that should not be merged into Info.plist",
        "pngs": """PNG images which are not bundled in an .xcassets folder or an .icon folder in \
Xcode 26+.""",
        "processed": "Typed resources that have already been processed.",
        "storyboards": "Storyboard files.",
        "strings": "Localization strings files.",
        "texture_atlases": "Texture atlas files.",
        "unprocessed": "Generic resources not mapped to the other types.",
        "xcstrings": "String catalog files.",
        "xibs": "XIB Interface files.",
        "owners": """`depset` of (resource, owner) pairs.""",
        "processed_origins": """`depset` of (processed resource, resource list) pairs.""",
        "unowned_resources": """`depset` of unowned resources.""",
    },
    init = _make_banned_init(provider_name = "AppleResourceInfo"),
)

AppleSharedCapabilityInfo, new_applesharedcapabilityinfo = provider(
    doc = "Provides information on a mergeable set of shared capabilities.",
    fields = {
        "base_bundle_id": """
`String`. The bundle ID prefix, composed from an organization ID and an optional variant name.
""",
    },
    init = _make_banned_init(provider_name = "AppleSharedCapabilityInfo"),
)

AppleStaticXcframeworkBundleInfo, new_applestaticxcframeworkbundleinfo = provider(
    doc = """
Denotes that a target is a static library XCFramework.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an XCFramework bundle
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is an XCFramework should use this provider to describe that
requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "AppleStaticXcframeworkBundleInfo"),
)

AppleTestInfo, new_appletestinfo = provider(
    doc = """
Provider that test targets propagate to be used for IDE integration.

This includes information regarding test source files, transitive include paths,
transitive module maps, and transitive Swift modules. Test source files are
considered to be all of which belong to the first-level dependencies on the test
target.
""",
    fields = {
        "includes": """
`depset` of `String`s representing transitive include paths which are needed by
IDEs to be used for indexing the test sources.
""",
        "module_maps": """
`depset` of `File`s representing module maps which are needed by IDEs to be used
for indexing the test sources.
""",
        "module_name": """
`String` representing the module name used by the test's sources. This is only
set if the test only contains a single top-level Swift dependency. This may be
used by an IDE to identify the Swift module (if any) used by the test's sources.
""",
        "non_arc_sources": """
`depset` of `File`s containing non-ARC sources from the test's immediate
deps.
""",
        "sources": """
`depset` of `File`s containing sources and headers from the test's immediate deps.
""",
        "swift_modules": """
`depset` of `File`s representing transitive swift modules which are needed by
IDEs to be used for indexing the test sources.
""",
        "test_bundle": "The artifact representing the XCTest bundle for the test target.",
        "test_host": """
The artifact representing the test host for the test target, if the test requires a test host.
""",
        "deps": """
`depset` of `String`s representing the labels of all immediate deps of the test.
Only source files from these deps will be present in `sources`. This may be used
by IDEs to differentiate a test target's transitive module maps from its direct
module maps, as including the direct module maps may break indexing for the
source files of the immediate deps.
""",
    },
    init = _make_banned_init(provider_name = "AppleTestInfo"),
)

AppleTestRunnerInfo, new_appletestrunnerinfo = provider(
    doc = """
Provider that runner targets must propagate.

In addition to the fields, all the runfiles that the runner target declares will be added to the
test rules runfiles.
""",
    fields = {
        "execution_requirements": """
Optional dictionary that represents the specific hardware requirements for this test.
""",
        "execution_environment": """
Optional dictionary with the environment variables that are to be set in the test action, and are
not propagated into the XCTest invocation. These values will _not_ be added into the %(test_env)s
substitution, but will be set in the test action.
""",
        "test_environment": """
Optional dictionary with the environment variables that are to be propagated into the XCTest
invocation. These values will be included in the %(test_env)s substitution and will _not_ be set in
the test action.
""",
        "test_runner_template": """
Required template file that contains the specific mechanism with which the tests will be run. The
*_ui_test and *_unit_test rules will substitute the following values:
    * %(test_host_path)s:   Path to the app being tested.
    * %(test_bundle_path)s: Path to the test bundle that contains the tests.
    * %(test_env)s:         Environment variables for the XCTest invocation (e.g FOO=BAR,BAZ=QUX).
    * %(test_type)s:        The test type, whether it is unit or UI.
""",
    },
    init = _make_banned_init(
        provider_name = "AppleTestRunnerInfo",
        preferred_public_factory = "make_apple_test_runner_info(...)",
    ),
)

def make_apple_test_runner_info(**kwargs):
    """Creates a new instance of the AppleTestRunnerInfo provider.

    Args:
        **kwargs: A set of keyword arguments expected to match the fields of `AppleTestRunnerInfo`.
            See the documentation for `AppleTestRunnerInfo` for what these must be.

    Returns:
        A new `AppleTestRunnerInfo` provider based on the supplied arguments.
    """
    if "test_runner_template" not in kwargs or not kwargs["test_runner_template"]:
        fail("""
Error: Could not find the required argument "test_runner_template" needed to build an
AppleTestRunner provider.

Received the following arguments for make_apple_test_runner_info: {kwargs}
""".format(kwargs = ", ".join(kwargs.keys())))

    return new_appletestrunnerinfo(**kwargs)

AppleXcframeworkBundleInfo, new_applexcframeworkbundleinfo = provider(
    doc = """
Denotes that a target is an XCFramework.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an XCFramework bundle
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is an XCFramework should use this provider to describe that
requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "AppleXcframeworkBundleInfo"),
)

IosApplicationBundleInfo, new_iosapplicationbundleinfo = provider(
    doc = """
Denotes that a target is an iOS application.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an iOS application bundle
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is an iOS application should use this provider to describe that
requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "IosApplicationBundleInfo"),
)

IosAppClipBundleInfo, new_iosappclipbundleinfo = provider(
    doc = """
Denotes that a target is an iOS app clip.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an iOS app clip bundle (and
not some other Apple bundle). Rule authors who wish to require that a dependency
is an iOS app clip should use this provider to describe that requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "IosAppClipBundleInfo"),
)

IosExtensionBundleInfo, new_iosextensionbundleinfo = provider(
    doc = """
Denotes that a target is an iOS application extension.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an iOS application
extension bundle (and not some other Apple bundle). Rule authors who wish to
require that a dependency is an iOS application extension should use this
provider to describe that requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "IosExtensionBundleInfo"),
)

IosFrameworkBundleInfo, new_iosframeworkbundleinfo = provider(
    doc = """
Denotes that a target is an iOS dynamic framework.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an iOS dynamic framework
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is an iOS dynamic framework should use this provider to describe
that requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "IosFrameworkBundleInfo"),
)

IosStaticFrameworkBundleInfo, new_iosstaticframeworkbundleinfo = provider(
    doc = """
Denotes that a target is an iOS static framework.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an iOS static framework
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is an iOS static framework should use this provider to describe
that requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "IosStaticFrameworkBundleInfo"),
)

IosImessageApplicationBundleInfo, new_iosimessageapplicationbundleinfo = provider(
    doc = """
Denotes that a target is an iOS iMessage application.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an iOS iMessage application
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is an iOS iMessage application should use this provider to describe
that requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "IosImessageApplicationBundleInfo"),
)

IosImessageExtensionBundleInfo, new_iosimessageextensionbundleinfo = provider(
    doc = """
Denotes that a target is an iOS iMessage extension.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an iOS iMessage extension
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is an iOS iMessage extension should use this provider to describe
that requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "IosImessageExtensionBundleInfo"),
)

IosXcTestBundleInfo, new_iosxctestbundleinfo = provider(
    doc = """
Denotes a target that is an iOS .xctest bundle.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an iOS .xctest bundle (and
not some other Apple bundle). Rule authors who wish to require that a dependency
is an iOS .xctest bundle should use this provider to describe that requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "IosXcTestBundleInfo"),
)

MacosApplicationBundleInfo, new_macosapplicationbundleinfo = provider(
    doc = """
Denotes that a target is a macOS application.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a macOS application bundle
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is a macOS application should use this provider to describe that
requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "MacosApplicationBundleInfo"),
)

MacosBundleBundleInfo, new_macosbundlebundleinfo = provider(
    doc = """
Denotes that a target is a macOS loadable bundle.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a macOS loadable bundle
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is a macOS loadable bundle should use this provider to describe that
requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "MacosBundleBundleInfo"),
)

MacosExtensionBundleInfo, new_macosextensionbundleinfo = provider(
    doc = """
Denotes that a target is a macOS application extension.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a macOS application
extension bundle (and not some other Apple bundle). Rule authors who wish to
require that a dependency is a macOS application extension should use this
provider to describe that requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "MacosExtensionBundleInfo"),
)

MacosKernelExtensionBundleInfo, new_macoskernelextensionbundleinfo = provider(
    doc = """
Denotes that a target is a macOS kernel extension.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a macOS kernel extension
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is a macOS kernel extension should use this provider to describe that
requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "MacosKernelExtensionBundleInfo"),
)

MacosQuickLookPluginBundleInfo, new_macosquicklookpluginbundleinfo = provider(
    doc = """
Denotes that a target is a macOS Quick Look Generator bundle.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a macOS Quick Look generator
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is a macOS Quick Look generator should use this provider to describe
that requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "MacosQuickLookPluginBundleInfo"),
)

MacosSpotlightImporterBundleInfo, new_macosspotlightimporterbundleinfo = provider(
    doc = """
Denotes that a target is a macOS Spotlight Importer bundle.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a macOS Spotlight importer
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is a macOS Spotlight importer should use this provider to describe that
requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "MacosSpotlightImporterBundleInfo"),
)

MacosXPCServiceBundleInfo, new_macosxpcservicebundleinfo = provider(
    doc = """
Denotes that a target is a macOS XPC Service bundle.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a macOS XPC service
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is a macOS XPC service should use this provider to describe that
requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "MacosXPCServiceBundleInfo"),
)

MacosXcTestBundleInfo, new_macosxctestbundleinfo = provider(
    doc = """
Denotes a target that is a macOS .xctest bundle.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a macOS .xctest bundle
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is a macOS .xctest bundle should use this provider to describe that
requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "MacosXcTestBundleInfo"),
)

TvosApplicationBundleInfo, new_tvosapplicationbundleinfo = provider(
    doc = """
Denotes that a target is a tvOS application.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a tvOS application bundle
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is a tvOS application should use this provider to describe that
requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "TvosApplicationBundleInfo"),
)

TvosExtensionBundleInfo, new_tvosextensionbundleinfo = provider(
    doc = """
Denotes that a target is a tvOS application extension.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a tvOS application
extension bundle (and not some other Apple bundle). Rule authors who wish to
require that a dependency is a tvOS application extension should use this
provider to describe that requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "TvosExtensionBundleInfo"),
)

TvosFrameworkBundleInfo, new_tvosframeworkbundleinfo = provider(
    doc = """
Denotes that a target is a tvOS dynamic framework.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a tvOS dynamic framework
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is a tvOS dynamic framework should use this provider to describe
that requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "TvosFrameworkBundleInfo"),
)

TvosStaticFrameworkBundleInfo, new_tvosstaticframeworkbundleinfo = provider(
    doc = """
Denotes that a target is a tvOS static framework.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a tvOS static framework
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is a tvOS static framework should use this provider to describe
that requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "TvosStaticFrameworkBundleInfo"),
)

TvosXcTestBundleInfo, new_tvosxctestbundleinfo = provider(
    doc = """
Denotes a target that is a tvOS .xctest bundle.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a tvOS .xctest bundle (and
not some other Apple bundle). Rule authors who wish to require that a dependency
is a tvOS .xctest bundle should use this provider to describe that requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "TvosXcTestBundleInfo"),
)

VisionosApplicationBundleInfo, new_visionosapplicationbundleinfo = provider(
    doc = """
Denotes that a target is a visionOS application.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a visionOS application
bundle (and not some other Apple bundle). Rule authors who wish to require that a
dependency is a visionOS application should use this provider to describe that
requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "VisionosApplicationBundleInfo"),
)

VisionosExtensionBundleInfo, new_visionosextensionbundleinfo = provider(
    doc = """
Denotes that a target is an visionOS application extension.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an iOS application
extension bundle (and not some other Apple bundle). Rule authors who wish to
require that a dependency is an iOS application extension should use this
provider to describe that requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "VisionosExtensionBundleInfo"),
)

VisionosFrameworkBundleInfo, new_visionosframeworkbundleinfo = provider(
    doc = """
Denotes that a target is visionOS dynamic framework.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a visionOS dynamic framework
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is a visionOS dynamic framework should use this provider to describe
that requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "VisionosFrameworkBundleInfo"),
)

VisionosStaticFrameworkBundleInfo, new_visionosstaticframeworkbundleinfo = provider(
    doc = """
Denotes that a target is an visionOS static framework.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a visionOS static framework
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is a visionOS static framework should use this provider to describe
that requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "VisionosStaticFrameworkBundleInfo"),
)

VisionosXcTestBundleInfo, new_visionosxctestbundleinfo = provider(
    doc = """
Denotes a target that is a visionOS .xctest bundle.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a visionOS .xctest bundle
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is a visionOS .xctest bundle  should use this provider to describe
that requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "VisionosXcTestBundleInfo"),
)

WatchosApplicationBundleInfo, new_watchosapplicationbundleinfo = provider(
    doc = """
Denotes that a target is a watchOS application.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a watchOS application
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is a watchOS application should use this provider to describe that
requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "WatchosApplicationBundleInfo"),
)

WatchosExtensionBundleInfo, new_watchosextensionbundleinfo = provider(
    doc = """
Denotes that a target is a watchOS application extension.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a watchOS application
extension bundle (and not some other Apple bundle). Rule authors who wish to
require that a dependency is a watchOS application extension should use this
provider to describe that requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "WatchosExtensionBundleInfo"),
)

WatchosXcTestBundleInfo, new_watchosxctestbundleinfo = provider(
    doc = """
Denotes a target that is a watchOS .xctest bundle.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a watchOS .xctest bundle (and
not some other Apple bundle). Rule authors who wish to require that a dependency
is a watchOS .xctest bundle should use this provider to describe that requirement.
""",
    fields = {},
    init = _make_banned_init(provider_name = "WatchosXcTestBundleInfo"),
)
