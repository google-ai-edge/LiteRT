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

"""Support for describing rule implementations.

The main purpose of this file is to have a central location that fully describes how a rule
implementation should behave, based only on the platform and product type. In previous
implementations of the rules, these information would be encoded in multiple private attributes.
With this approach, both rule definition and implementation infrastructure can access the same
parameters that affect both the attributes and the implementation logic of the rules.
"""

load(
    "//apple/internal:apple_product_type.bzl",
    "apple_product_type",
)
load(
    "//apple/internal:bundle_package_type.bzl",
    "bundle_package_type",
)

# Options to declare signing behavior and exceptions.
#
# Args:
#     none: No additional exceptions to code signing.
#     sign_with_provisioning_profile: Sign the target's bundle/binary with a provisioning profile
#         and no entitlements if a provisioning profile was specified at its level.
#     skip_signing: Do not sign binaries and frameworks for this target.
_CODESIGNING_EXCEPTIONS = struct(
    none = None,
    sign_with_provisioning_profile = "sign_with_provisioning_profile",
    skip_signing = "skip_signing",
)

def _describe_bundle_locations(
        archive_relative = "",
        bundle_relative_contents = "",
        contents_relative_app_clips = "AppClips",
        contents_relative_binary = "",
        contents_relative_extensions = "Extensions",
        contents_relative_frameworks = "Frameworks",
        contents_relative_plugins = "PlugIns",
        contents_relative_resources = "",
        contents_relative_watch = "Watch",
        contents_relative_xpc_service = "XPCServices"):
    """Creates a descriptor of locations for different types of artifacts within an Apple bundle."""
    return struct(
        archive_relative = archive_relative,
        bundle_relative_contents = bundle_relative_contents,
        contents_relative_app_clips = contents_relative_app_clips,
        contents_relative_binary = contents_relative_binary,
        contents_relative_extensions = contents_relative_extensions,
        contents_relative_frameworks = contents_relative_frameworks,
        contents_relative_plugins = contents_relative_plugins,
        contents_relative_resources = contents_relative_resources,
        contents_relative_watch = contents_relative_watch,
        contents_relative_xpc_service = contents_relative_xpc_service,
    )

def _describe_rule_type(
        additional_infoplist_values = None,
        allowed_device_families = None,
        allows_locale_trimming = False,
        binary_infoplist = True,
        bundle_extension = None,
        bundle_locations = None,
        bundle_package_type = None,
        codesigning_exceptions = _CODESIGNING_EXCEPTIONS.none,
        expose_non_archive_relative_output = False,
        product_type = None,
        requires_pkginfo = False,
        requires_signing_for_device = True,
        rpaths = [],
        skip_simulator_signing_allowed = True,
        stub_binary_path = None):
    """Creates a rule descriptor struct containing all the platform and product specific configs.

    Args:
        additional_infoplist_values: Dictionary of additional values to set into the rule's
            Info.plist.
        allowed_device_families: If given, the list of device families that this rule supports.
        allows_locale_trimming: Whether the rule supports trimming `.lproj` localizations.
        binary_infoplist: Whether the Info.plist output should be in binary form.
        bundle_package_type: Four-character code representing the bundle type.
        bundle_extension: Extension for the Apple bundle inside the archive.
        bundle_locations: Struct with expected bundle locations for different types of artifacts.
        codesigning_exceptions: A value from _CODESIGNING_EXCEPTIONS to determine conditions for
            code signing, if exceptions should be made.
        expose_non_archive_relative_output: Whether or not to expose an output archive that ignores
            the `archive_relative` bundle location, to permit embedding within another target. Has no
            effect if `archive_relative` is empty.
        product_type: The product type for this rule.
        requires_pkginfo: Whether the PkgInfo file should be included inside the rule's bundle.
        requires_signing_for_device: Whether signing is required when building for devices (as
            opposed to simulators).
        rpaths: List of rpaths to add to the linker.
        skip_simulator_signing_allowed: Whether this rule is allowed to skip signing when building
            for the simulator.
        stub_binary_path: Xcode SDK root relative path to the stub binary to copy as this rule's
            binary artifact.

    Returns:
        A struct with fields that describe the configuration for a specific bundling rule.
    """

    if not bundle_locations:
        bundle_locations = _describe_bundle_locations()

    return struct(
        additional_infoplist_values = additional_infoplist_values,
        allowed_device_families = allowed_device_families,
        allows_locale_trimming = allows_locale_trimming,
        binary_infoplist = binary_infoplist,
        bundle_extension = bundle_extension,
        bundle_locations = bundle_locations,
        bundle_package_type = bundle_package_type,
        codesigning_exceptions = codesigning_exceptions,
        expose_non_archive_relative_output = expose_non_archive_relative_output,
        product_type = product_type,
        requires_pkginfo = requires_pkginfo,
        requires_signing_for_device = requires_signing_for_device,
        rpaths = rpaths,
        skip_simulator_signing_allowed = skip_simulator_signing_allowed,
        stub_binary_path = stub_binary_path,
    )

_DEFAULT_MACOS_BUNDLE_LOCATIONS = _describe_bundle_locations(
    bundle_relative_contents = "Contents",
    contents_relative_binary = "MacOS",
    contents_relative_resources = "Resources",
)

# Descriptors for all possible platform/product type combinations.
# TODO(b/248317958): Migrate rpaths to args on the linking_support methods.
_RULE_TYPE_DESCRIPTORS = {
    "ios": {
        # ios_application
        apple_product_type.application: _describe_rule_type(
            allowed_device_families = ["iphone", "ipad"],
            allows_locale_trimming = True,
            bundle_extension = ".app",
            bundle_locations = _describe_bundle_locations(archive_relative = "Payload"),
            bundle_package_type = bundle_package_type.application,
            product_type = apple_product_type.application,
            requires_pkginfo = True,
            rpaths = [
                # Application binaries live in Application.app/Application
                # Frameworks are packaged in Application.app/Frameworks
                "@executable_path/Frameworks",
            ],
        ),
        # ios_app_clip
        apple_product_type.app_clip: _describe_rule_type(
            allowed_device_families = ["iphone", "ipad"],
            allows_locale_trimming = True,
            bundle_extension = ".app",
            bundle_locations = _describe_bundle_locations(archive_relative = "Payload"),
            bundle_package_type = bundle_package_type.application,
            expose_non_archive_relative_output = True,
            product_type = apple_product_type.app_clip,
            requires_pkginfo = True,
            rpaths = [
                # AppClip binary located at AppClip.app/AppClip
                # Frameworks are packaged in AppClip.app/Frameworks
                "@executable_path/Frameworks",
            ],
        ),
        # ios_extension (NSExtension)
        apple_product_type.app_extension: _describe_rule_type(
            allowed_device_families = ["iphone", "ipad"],
            allows_locale_trimming = True,
            bundle_extension = ".appex",
            bundle_package_type = bundle_package_type.extension_or_xpc,
            product_type = apple_product_type.app_extension,
            rpaths = [
                # Extension binaries live in Application.app/PlugIns/Extension.appex/Extension
                # Frameworks are packaged in Application.app/PlugIns/Extension.appex/Frameworks
                # or Application.app/Frameworks
                "@executable_path/Frameworks",
                "@executable_path/../../Frameworks",
            ],
        ),
        # ios_extension (ExtensionKit)
        apple_product_type.extensionkit_extension: _describe_rule_type(
            allowed_device_families = ["iphone", "ipad"],
            allows_locale_trimming = True,
            bundle_extension = ".appex",
            bundle_package_type = bundle_package_type.extension_or_xpc,
            product_type = apple_product_type.extensionkit_extension,
            rpaths = [
                # ExtensionKit binaries live in Application.app/Extensions/Extension.appex/Extension
                # Frameworks are packaged in Application.app/Frameworks
                "@executable_path/../../Frameworks",
            ],
        ),
        # ios_framework
        apple_product_type.framework: _describe_rule_type(
            allowed_device_families = ["iphone", "ipad"],
            bundle_extension = ".framework",
            bundle_package_type = bundle_package_type.framework,
            codesigning_exceptions = _CODESIGNING_EXCEPTIONS.sign_with_provisioning_profile,
            product_type = apple_product_type.framework,
            rpaths = [
                # Framework binaries live in
                # Application.app/Frameworks/Framework.framework/Framework or
                # Application.app/PlugIns/Extension.appex/Framework.framework/Framework
                # Frameworks are packaged in Application.app/Frameworks or
                # Application.app/PlugIns/Extension.appex/Frameworks
                "@executable_path/Frameworks",
            ],
        ),
        # ios_imessage_application
        apple_product_type.messages_application: _describe_rule_type(
            additional_infoplist_values = {"LSApplicationLaunchProhibited": True},
            allows_locale_trimming = True,
            allowed_device_families = ["iphone", "ipad"],
            bundle_extension = ".app",
            bundle_package_type = bundle_package_type.application,
            bundle_locations = _describe_bundle_locations(archive_relative = "Payload"),
            product_type = apple_product_type.messages_application,
            stub_binary_path = "../../../Library/Application Support/" +
                               "MessagesApplicationStub/MessagesApplicationStub",
        ),
        # ios_imessage_extension
        apple_product_type.messages_extension: _describe_rule_type(
            allowed_device_families = ["iphone", "ipad"],
            allows_locale_trimming = True,
            bundle_extension = ".appex",
            bundle_package_type = bundle_package_type.extension_or_xpc,
            product_type = apple_product_type.messages_extension,
            rpaths = [
                # Extension binaries live in Application.app/PlugIns/Extension.appex/Extension
                # Frameworks are packaged in Application.app/PlugIns/Extension.appex/Frameworks
                # or Application.app/Frameworks
                "@executable_path/Frameworks",
                "@executable_path/../../Frameworks",
            ],
        ),
        # ios_stickerpack_extension
        apple_product_type.messages_sticker_pack_extension: _describe_rule_type(
            additional_infoplist_values = {"LSApplicationIsStickerProvider": "YES"},
            allowed_device_families = ["iphone", "ipad"],
            bundle_extension = ".appex",
            bundle_package_type = bundle_package_type.extension_or_xpc,
            product_type = apple_product_type.messages_sticker_pack_extension,
            stub_binary_path = "../../../Library/Application Support/" +
                               "MessagesApplicationExtensionStub/MessagesApplicationExtensionStub",
        ),
        # ios_static_framework
        apple_product_type.static_framework: _describe_rule_type(
            allowed_device_families = ["iphone", "ipad"],
            bundle_extension = ".framework",
            codesigning_exceptions = _CODESIGNING_EXCEPTIONS.skip_signing,
            product_type = apple_product_type.static_framework,
        ),
        # ios_ui_test
        apple_product_type.ui_test_bundle: _describe_rule_type(
            allowed_device_families = ["iphone", "ipad"],
            bundle_extension = ".xctest",
            bundle_package_type = bundle_package_type.bundle,
            product_type = apple_product_type.ui_test_bundle,
            requires_signing_for_device = False,
            rpaths = [
                # Test binaries live in Application.app/PlugIns/Test.xctest/Test
                # Frameworks are packaged in Application.app/Frameworks and in
                # Application.app/PlugIns/Test.xctest/Frameworks
                "@executable_path/Frameworks",
                "@loader_path/Frameworks",
            ],
            skip_simulator_signing_allowed = False,
        ),
        # ios_unit_test
        apple_product_type.unit_test_bundle: _describe_rule_type(
            allowed_device_families = ["iphone", "ipad"],
            bundle_extension = ".xctest",
            bundle_package_type = bundle_package_type.bundle,
            product_type = apple_product_type.unit_test_bundle,
            requires_signing_for_device = False,
            rpaths = [
                # Test binaries live in Application.app/PlugIns/Test.xctest/Test
                # Frameworks are packaged in Application.app/Frameworks and in
                # Application.app/PlugIns/Test.xctest/Frameworks
                "@executable_path/Frameworks",
                "@loader_path/Frameworks",
            ],
            skip_simulator_signing_allowed = False,
        ),
    },
    "macos": {
        # macos_application
        apple_product_type.application: _describe_rule_type(
            allowed_device_families = ["mac"],
            allows_locale_trimming = True,
            bundle_extension = ".app",
            bundle_locations = _DEFAULT_MACOS_BUNDLE_LOCATIONS,
            bundle_package_type = bundle_package_type.application,
            product_type = apple_product_type.application,
            requires_pkginfo = True,
            requires_signing_for_device = False,
            rpaths = [
                # Application binaries live in Application.app/Contents/MacOS/Application
                # Frameworks are packaged in Application.app/Contents/Frameworks
                "@executable_path/../Frameworks",
            ],
        ),
        # macos_command_line_application
        apple_product_type.tool: _describe_rule_type(
            allowed_device_families = ["mac"],
            bundle_extension = "",
            product_type = apple_product_type.tool,
            requires_signing_for_device = False,
        ),
        # macos_dylib
        apple_product_type.dylib: _describe_rule_type(
            allowed_device_families = ["mac"],
            bundle_extension = "",
            product_type = apple_product_type.dylib,
            requires_signing_for_device = False,
        ),
        # macos_extension (NSExtension)
        apple_product_type.app_extension: _describe_rule_type(
            allowed_device_families = ["mac"],
            allows_locale_trimming = True,
            bundle_extension = ".appex",
            bundle_locations = _DEFAULT_MACOS_BUNDLE_LOCATIONS,
            bundle_package_type = bundle_package_type.extension_or_xpc,
            product_type = apple_product_type.app_extension,
            requires_signing_for_device = False,
            rpaths = [
                # Extension binaries live in
                # Application.app/Contents/PlugIns/Extension.appex/Contents/MacOS/Extension
                # Frameworks are packaged in
                # Application.app/Contents/PlugIns/Extension.appex/Contents/Frameworks
                # or Application.app/Contents/Frameworks
                "@executable_path/../Frameworks",
                "@executable_path/../../../../Frameworks",
            ],
        ),
        # macos_extension (ExtensionKit)
        apple_product_type.extensionkit_extension: _describe_rule_type(
            allowed_device_families = ["mac"],
            allows_locale_trimming = True,
            bundle_extension = ".appex",
            bundle_locations = _DEFAULT_MACOS_BUNDLE_LOCATIONS,
            bundle_package_type = bundle_package_type.extension_or_xpc,
            product_type = apple_product_type.extensionkit_extension,
            requires_signing_for_device = False,
            rpaths = [
                # ExtensionKit binaries live in
                # Application.app/Contents/Extensions/Extension.appex/Contents/MacOS/Extension
                # Frameworks are packaged in Application.app/Contents/Frameworks
                "@executable_path/../../../../Frameworks",
            ],
        ),
        # macos_quick_look_plugin
        apple_product_type.quicklook_plugin: _describe_rule_type(
            allowed_device_families = ["mac"],
            bundle_extension = ".qlgenerator",
            bundle_locations = _DEFAULT_MACOS_BUNDLE_LOCATIONS,
            bundle_package_type = bundle_package_type.bundle,
            product_type = apple_product_type.quicklook_plugin,
            requires_signing_for_device = False,
        ),
        # macos_bundle
        apple_product_type.bundle: _describe_rule_type(
            allowed_device_families = ["mac"],
            bundle_extension = ".bundle",
            bundle_locations = _DEFAULT_MACOS_BUNDLE_LOCATIONS,
            bundle_package_type = bundle_package_type.bundle,
            product_type = apple_product_type.bundle,
            requires_signing_for_device = False,
            rpaths = [
                # Bundle binaries are loaded from the executable location and application binaries
                # live in Application.app/Contents/MacOS/Application
                # Frameworks are packaged in Application.app/Contents/Frameworks
                "@executable_path/../Frameworks",
            ],
        ),
        # macos_kernel_extension
        apple_product_type.kernel_extension: _describe_rule_type(
            allowed_device_families = ["mac"],
            binary_infoplist = False,
            bundle_extension = ".kext",
            bundle_locations = _DEFAULT_MACOS_BUNDLE_LOCATIONS,
            bundle_package_type = bundle_package_type.kernel_extension,
            product_type = apple_product_type.kernel_extension,
            requires_signing_for_device = False,
        ),
        # macos_spotlight_importer
        apple_product_type.spotlight_importer: _describe_rule_type(
            allowed_device_families = ["mac"],
            bundle_extension = ".mdimporter",
            bundle_locations = _DEFAULT_MACOS_BUNDLE_LOCATIONS,
            bundle_package_type = bundle_package_type.extension_or_xpc,
            product_type = apple_product_type.spotlight_importer,
            requires_signing_for_device = False,
        ),
        # macos_xpc_service
        apple_product_type.xpc_service: _describe_rule_type(
            allowed_device_families = ["mac"],
            bundle_extension = ".xpc",
            bundle_locations = _DEFAULT_MACOS_BUNDLE_LOCATIONS,
            bundle_package_type = bundle_package_type.extension_or_xpc,
            product_type = apple_product_type.xpc_service,
            requires_signing_for_device = False,
            requires_pkginfo = True,
            rpaths = [
                # XPC Application binaries live in
                # Application.app/Contents/XCPServices/XPCService.xpc/Contents/MacOS/XPCService
                # Frameworks are packaged in Application.app/Contents/XCPServices/XPCService.xpc/Contents/Frameworks
                "@executable_path/../Frameworks",
                # Shared frameworks are packaged in Application.app/Contents/Frameworks
                "@executable_path/../../../../Frameworks",
            ],
        ),
        # macos_ui_test
        apple_product_type.ui_test_bundle: _describe_rule_type(
            allowed_device_families = ["mac"],
            bundle_extension = ".xctest",
            bundle_locations = _DEFAULT_MACOS_BUNDLE_LOCATIONS,
            bundle_package_type = bundle_package_type.bundle,
            product_type = apple_product_type.ui_test_bundle,
            requires_signing_for_device = False,
            rpaths = [
                # Test binaries live in
                # Application.app/Contents/PlugIns/Test.xctest/Contents/MacOS/Test
                # Frameworks are packaged in Application.app/Contents/Frameworks and in
                # Application.app/Contents/PlugIns/Test.xctest/Contents/Frameworks
                "@executable_path/../Frameworks",
                "@loader_path/../Frameworks",
            ],
        ),
        # macos_unit_test
        apple_product_type.unit_test_bundle: _describe_rule_type(
            allowed_device_families = ["mac"],
            bundle_extension = ".xctest",
            bundle_locations = _DEFAULT_MACOS_BUNDLE_LOCATIONS,
            bundle_package_type = bundle_package_type.bundle,
            product_type = apple_product_type.unit_test_bundle,
            requires_signing_for_device = False,
            rpaths = [
                # Test binaries live in
                # Application.app/Contents/PlugIns/Test.xctest/Contents/MacOS/Test
                # Frameworks are packaged in Application.app/Contents/Frameworks and in
                # Application.app/Contents/PlugIns/Test.xctest/Contents/Frameworks
                "@executable_path/../Frameworks",
                "@loader_path/../Frameworks",
            ],
        ),
        # macos_framework
        apple_product_type.framework: _describe_rule_type(
            allowed_device_families = ["mac"],
            bundle_extension = ".framework",
            bundle_package_type = bundle_package_type.framework,
            codesigning_exceptions = _CODESIGNING_EXCEPTIONS.sign_with_provisioning_profile,
            product_type = apple_product_type.framework,
            rpaths = [
                # Application binaries  Application.app/Contents/MacOS/Application
                # Frameworks            Application.app/Contents/Frameworks
                # XPCService Frameworks Application.app/Contents/XCPServices/XPCService.xpc/Contents/Frameworks
                # PlugIns Frameworks    Application.app/Contents/PlugIns/XXX.plugin/Contents/Frameworks
                "@executable_path/../Frameworks",
                "@executable_path/../../../../Frameworks",
            ],
        ),
        # macos_static_framework
        apple_product_type.static_framework: _describe_rule_type(
            allowed_device_families = ["mac"],
            bundle_extension = ".framework",
            codesigning_exceptions = _CODESIGNING_EXCEPTIONS.skip_signing,
            product_type = apple_product_type.static_framework,
        ),
    },
    "tvos": {
        # tvos_application
        apple_product_type.application: _describe_rule_type(
            allowed_device_families = ["tv"],
            allows_locale_trimming = True,
            bundle_extension = ".app",
            bundle_locations = _describe_bundle_locations(archive_relative = "Payload"),
            bundle_package_type = bundle_package_type.application,
            product_type = apple_product_type.application,
            requires_pkginfo = True,
            rpaths = [
                # Application binaries live in Application.app/Application
                # Frameworks are packaged in Application.app/Frameworks
                "@executable_path/Frameworks",
            ],
        ),
        # tvos_extension (NSExtension)
        apple_product_type.app_extension: _describe_rule_type(
            allowed_device_families = ["tv"],
            allows_locale_trimming = True,
            bundle_extension = ".appex",
            bundle_package_type = bundle_package_type.extension_or_xpc,
            product_type = apple_product_type.app_extension,
            rpaths = [
                # Extension binaries live in Application.app/PlugIns/Extension.appex/Extension
                # Frameworks are packaged in Application.app/PlugIns/Extension.appex/Frameworks
                # or Application.app/Frameworks
                "@executable_path/Frameworks",
                "@executable_path/../../Frameworks",
            ],
        ),
        # tvos_extension (ExtensionKit)
        apple_product_type.extensionkit_extension: _describe_rule_type(
            allowed_device_families = ["tv"],
            allows_locale_trimming = True,
            bundle_extension = ".appex",
            bundle_package_type = bundle_package_type.extension_or_xpc,
            product_type = apple_product_type.extensionkit_extension,
            rpaths = [
                # Extension binaries live in Application.app/Extensions/Extension.appex/Extension
                # Frameworks are packaged in Application.app/Frameworks
                "@executable_path/../../Frameworks",
            ],
        ),
        # tvos_framework
        apple_product_type.framework: _describe_rule_type(
            allowed_device_families = ["tv"],
            bundle_extension = ".framework",
            bundle_package_type = bundle_package_type.framework,
            codesigning_exceptions = _CODESIGNING_EXCEPTIONS.sign_with_provisioning_profile,
            product_type = apple_product_type.framework,
            rpaths = [
                # Framework binaries live in
                # Application.app/Frameworks/Framework.framework/Framework or
                # Application.app/PlugIns/Extension.appex/Framework.framework/Framework
                # Frameworks are packaged in Application.app/Frameworks or
                # Application.app/PlugIns/Extension.appex/Frameworks
                "@executable_path/Frameworks",
            ],
        ),
        # tvos_static_framework
        apple_product_type.static_framework: _describe_rule_type(
            allowed_device_families = ["tv"],
            bundle_extension = ".framework",
            codesigning_exceptions = _CODESIGNING_EXCEPTIONS.skip_signing,
            product_type = apple_product_type.static_framework,
        ),
        # tvos_ui_test
        apple_product_type.ui_test_bundle: _describe_rule_type(
            allowed_device_families = ["tv"],
            bundle_extension = ".xctest",
            bundle_package_type = bundle_package_type.bundle,
            product_type = apple_product_type.ui_test_bundle,
            requires_signing_for_device = False,
            rpaths = [
                # Test binaries live in Application.app/PlugIns/Test.xctest/Test
                # Frameworks are packaged in Application.app/Frameworks and in
                # Application.app/PlugIns/Test.xctest/Frameworks
                "@executable_path/Frameworks",
                "@loader_path/Frameworks",
            ],
            skip_simulator_signing_allowed = False,
        ),
        # tvos_unit_test
        apple_product_type.unit_test_bundle: _describe_rule_type(
            allowed_device_families = ["tv"],
            bundle_extension = ".xctest",
            bundle_package_type = bundle_package_type.bundle,
            product_type = apple_product_type.unit_test_bundle,
            requires_signing_for_device = False,
            rpaths = [
                # Test binaries live in Application.app/PlugIns/Test.xctest/Test
                # Frameworks are packaged in Application.app/Frameworks and in
                # Application.app/PlugIns/Test.xctest/Frameworks
                "@executable_path/Frameworks",
                "@loader_path/Frameworks",
            ],
            skip_simulator_signing_allowed = False,
        ),
    },
    "visionos": {
        # visionos_application (single target application)
        apple_product_type.application: _describe_rule_type(
            allowed_device_families = ["vision"],
            allows_locale_trimming = True,
            bundle_extension = ".app",
            bundle_locations = _describe_bundle_locations(archive_relative = "Payload"),
            bundle_package_type = bundle_package_type.application,
            product_type = apple_product_type.application,
            requires_pkginfo = True,
            rpaths = [
                # Application binaries live in Application.app/Application
                "@executable_path/Frameworks",
            ],
        ),
        # visionos_extension
        apple_product_type.app_extension: _describe_rule_type(
            allowed_device_families = ["vision"],
            allows_locale_trimming = True,
            bundle_extension = ".appex",
            bundle_package_type = bundle_package_type.extension_or_xpc,
            product_type = apple_product_type.app_extension,
            rpaths = [
                # Extension binaries live in Application.app/PlugIns/Extension.appex/Extension
                # Frameworks are packaged in Application.app/PlugIns/Extension.appex/Frameworks
                # or Application.app/Frameworks
                "@executable_path/Frameworks",
                "@executable_path/../../Frameworks",
            ],
        ),
        # visionos_framework
        apple_product_type.framework: _describe_rule_type(
            allowed_device_families = ["vision"],
            bundle_extension = ".framework",
            bundle_package_type = bundle_package_type.framework,
            codesigning_exceptions = _CODESIGNING_EXCEPTIONS.sign_with_provisioning_profile,
            product_type = apple_product_type.framework,
            rpaths = [
                # Framework binaries live in
                # Application.app/Frameworks/Framework.framework/Framework or
                # Application.app/PlugIns/Extension.appex/Framework.framework/Framework
                # Frameworks are packaged in Application.app/Frameworks or
                # Application.app/PlugIns/Extension.appex/Frameworks
                "@executable_path/Frameworks",
            ],
        ),
        # visionos_static_framework
        apple_product_type.static_framework: _describe_rule_type(
            allowed_device_families = ["vision"],
            bundle_extension = ".framework",
            codesigning_exceptions = _CODESIGNING_EXCEPTIONS.skip_signing,
            product_type = apple_product_type.static_framework,
        ),
        # visionos_ui_test
        apple_product_type.ui_test_bundle: _describe_rule_type(
            allowed_device_families = ["vision"],
            bundle_extension = ".xctest",
            bundle_package_type = bundle_package_type.bundle,
            product_type = apple_product_type.ui_test_bundle,
            requires_signing_for_device = False,
            rpaths = [
                # Test binaries live in Application.app/PlugIns/Test.xctest/Test
                # Frameworks are packaged in Application.app/Frameworks and in
                # Application.app/PlugIns/Test.xctest/Frameworks
                "@executable_path/Frameworks",
                "@loader_path/Frameworks",
            ],
            skip_simulator_signing_allowed = False,
        ),
        # visionos_unit_test
        apple_product_type.unit_test_bundle: _describe_rule_type(
            allowed_device_families = ["vision"],
            bundle_extension = ".xctest",
            bundle_package_type = bundle_package_type.bundle,
            product_type = apple_product_type.unit_test_bundle,
            requires_signing_for_device = False,
            rpaths = [
                # Test binaries live in Application.app/PlugIns/Test.xctest/Test
                # Frameworks are packaged in Application.app/Frameworks and in
                # Application.app/PlugIns/Test.xctest/Frameworks
                "@executable_path/Frameworks",
                "@loader_path/Frameworks",
            ],
            skip_simulator_signing_allowed = False,
        ),
    },
    "watchos": {
        # watchos_application (single target application)
        apple_product_type.application: _describe_rule_type(
            additional_infoplist_values = {"WKApplication": True},
            allowed_device_families = ["watch"],
            allows_locale_trimming = True,
            bundle_extension = ".app",
            bundle_package_type = bundle_package_type.application,
            product_type = apple_product_type.application,
            requires_pkginfo = True,
            rpaths = [
                # Application binaries live in Application.app/Application
                "@executable_path/Frameworks",
            ],
        ),
        # watchos_application (watchOS 2 extension-based app bundle)
        apple_product_type.watch2_application: _describe_rule_type(
            additional_infoplist_values = {"WKWatchKitApp": True},
            allowed_device_families = ["watch"],
            allows_locale_trimming = True,
            bundle_extension = ".app",
            bundle_package_type = bundle_package_type.application,
            product_type = apple_product_type.watch2_application,
            requires_pkginfo = True,
            stub_binary_path = "Library/Application Support/WatchKit/WK",
        ),
        # watchos_extension (watchOS 2 app extension)
        apple_product_type.watch2_extension: _describe_rule_type(
            allowed_device_families = ["watch"],
            allows_locale_trimming = True,
            bundle_extension = ".appex",
            bundle_package_type = bundle_package_type.extension_or_xpc,
            product_type = apple_product_type.watch2_extension,
            rpaths = [
                # Extension binaries live in Application.app/PlugIns/Extension.appex/Extension
                # Frameworks are packaged in Application.app/PlugIns/Extension.appex/Frameworks
                # or Application.app/Frameworks
                "@executable_path/Frameworks",
                "@executable_path/../../Frameworks",
            ],
        ),
        # watchos_framework
        apple_product_type.framework: _describe_rule_type(
            allowed_device_families = ["watch"],
            bundle_extension = ".framework",
            bundle_package_type = bundle_package_type.framework,
            codesigning_exceptions = _CODESIGNING_EXCEPTIONS.sign_with_provisioning_profile,
            product_type = apple_product_type.framework,
            rpaths = [
                # Framework binaries live in
                # Application.app/Frameworks/Framework.framework/Framework or
                # Application.app/PlugIns/Extension.appex/Framework.framework/Framework
                # Frameworks are packaged in Application.app/Frameworks or
                # Application.app/PlugIns/Extension.appex/Frameworks
                "@executable_path/Frameworks",
            ],
        ),
        # watchos_static_framework
        apple_product_type.static_framework: _describe_rule_type(
            allowed_device_families = ["watch"],
            bundle_extension = ".framework",
            codesigning_exceptions = _CODESIGNING_EXCEPTIONS.skip_signing,
            product_type = apple_product_type.static_framework,
        ),
        # watchos_ui_test
        apple_product_type.ui_test_bundle: _describe_rule_type(
            allowed_device_families = ["watch"],
            bundle_extension = ".xctest",
            bundle_package_type = bundle_package_type.bundle,
            product_type = apple_product_type.ui_test_bundle,
            requires_signing_for_device = False,
            rpaths = [
                # Test binaries live in Application.app/PlugIns/Test.xctest/Test
                # Frameworks are packaged in Application.app/Frameworks and in
                # Application.app/PlugIns/Test.xctest/Frameworks
                "@executable_path/Frameworks",
                "@loader_path/Frameworks",
            ],
            skip_simulator_signing_allowed = False,
        ),
        # watchos_unit_test
        apple_product_type.unit_test_bundle: _describe_rule_type(
            allowed_device_families = ["watch"],
            bundle_extension = ".xctest",
            bundle_package_type = bundle_package_type.bundle,
            product_type = apple_product_type.unit_test_bundle,
            requires_signing_for_device = False,
            rpaths = [
                # Test binaries live in Application.app/PlugIns/Test.xctest/Test
                # Frameworks are packaged in Application.app/Frameworks and in
                # Application.app/PlugIns/Test.xctest/Frameworks
                "@executable_path/Frameworks",
                "@loader_path/Frameworks",
            ],
            skip_simulator_signing_allowed = False,
        ),
    },
}

def _rule_descriptor(*, platform_type, product_type):
    """Returns the rule descriptor for the given platform and product types.

    This method fails if the platform and product combination is invalid.

    Args:
        platform_type: Platform of the rule (e.g. "macos").
        product_type: Product type of the rule (e.g. apple_product_type.application).

    Returns:
        The rule descriptor that describes the rule for the given platform and product types.
    """
    rule_descriptor = _RULE_TYPE_DESCRIPTORS[platform_type].get(product_type)
    if not rule_descriptor:
        fail(
            "Platform type '{platform_type}' does not support product type '{product_type}'".format(
                platform_type = platform_type,
                product_type = product_type,
            ),
        )
    return rule_descriptor

rule_support = struct(
    rule_descriptor = _rule_descriptor,
    codesigning_exceptions = _CODESIGNING_EXCEPTIONS,
)
