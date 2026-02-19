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

"""Support for product types used by Apple bundling rules.

This should be internal to the Apple rules, used to control/configure the
type of rule being created and thus its descriptor to control behaviors.
"""

# Product type identifiers used to describe various bundle types.
#
# The "product type" is a concept used internally by Xcode (the strings themselves
# are visible inside the `.pbxproj` file) that describes properties of the bundle,
# such as its default extension.
#
# Additionally, products like iMessage applications and sticker packs in iOS 10
# require a stub executable instead of a user-defined binary and additional values
# injected into their `Info.plist` files. These behaviors are also captured in the
# product type identifier. The product types currently supported are:
#
# * `application`: A basic iOS, macOS, or tvOS application. This is the default
#   product type for those targets; it can be overridden with a more specific
#   product type if needed.
# * `app_extension`: A basic iOS, macOS, watchOS or tvOS application extension.
#   This is the default product type for those targets; it can be overridden with
#   a more specific product type if needed.
# * `bundle`: A loadable macOS bundle. This is the default product type for
#   `macos_bundle` targets; it can be overridden with a more specific product type
#   if needed.
# * `dylib`: A dynamically-loadable library. This is the default product type for
#   `macos_dylib`; it does not need to be set explicitly (and cannot be changed).
# * `extensionkit_extension`: A basic iOS, macOS, or tvOS ExtensionKit extension.
#   These are `.appex` bundles located in an application's `Extensions`
#   subdirectory. Third-party specs to define the integration points for third-
#   party ExtensionKit extensions can be defined through `.appextensionpoint` XML
#   plists on macOS.
# * `framework`: A basic dynamic framework. This is the default product type for
#   those targets; it does not need to be set explicitly (and cannot be changed).
# * `kernel_extension`: A macOS kernel extension. This product type should be used
#   with a `macos_bundle` target to create such a plug-in; the built bundle will
#   have the extension `.kext`.
# * `messages_application`: An application that integrates with the Messages
#   app (iOS 10 and above). This application must include an `ios_extension`
#   with the `messages_extension` or `messages_sticker_pack_extension` product
#   type (or both extensions). This product type does not contain a user-provided
#   binary.
# * `messages_extension`: An extension that integrates custom code/behavior into
#   a Messages application. This product type should contain a user-provided
#   binary.
# * `messages_sticker_pack_extension`: An extension that defines custom sticker
#   packs for the Messages app. This product type does not contain a
#   user-provided binary.
# * `quicklook_plugin`: A macOS Quick Look generator plug-in. This is the default
#   product type for `macos_quick_look_plugin` targets; it does not need to be set
#   explicitly (and cannot be changed).
# * `spotlight_importer`: A macOS Spotlight importer plug-in. This product type
#   should be used with a `macos_bundle` target to create such a plug-in; the
#   built bundle will have the extension `.mdimporter`.
# * `static_framework`: An iOS static framework, which is a `.framework` bundle
#   that contains resources and headers but a static library instead of a dynamic
#   library.
# * `tool`: A command-line tool. This is the default product type for
#   `macos_command_line_application`; it does not need to be set explicitly (and
#   cannot be changed).
# * `ui_test_bundle`: A UI testing bundle (.xctest). This is the default product
#   type for those targets; it does not need to be set explicitly (and cannot be
#   changed).
# * `unit_test_bundle`: A unit test bundle (.xctest). This is the default product
#   type for those targets; it does not need to be set explicitly (and cannot be
#   changed).
# * `watch2_application`: A watchOS 2+ application. This is the default product
#   type for those targets; it does not need to be set explicitly (and cannot be
#   changed).
# * `watch2_extension`: A watchOS 2+ application extension. This is the default
#   product type for those targets; it does not need to be set explicitly (and
#   cannot be changed).
# * `xpc_service`: A macOS XPC service. This product type should be used with a
#   `macos_application` target to create such a service; the built bundle will
#   have the extension `.xpc`.
apple_product_type = struct(
    application = "com.apple.product-type.application",
    app_clip = "com.apple.product-type.application.on-demand-install-capable",
    app_extension = "com.apple.product-type.app-extension",
    bundle = "com.apple.product-type.bundle",
    dylib = "com.apple.product-type.library.dynamic",
    extensionkit_extension = "com.apple.product-type.extensionkit-extension",
    framework = "com.apple.product-type.framework",
    kernel_extension = "com.apple.product-type.kernel-extension",
    messages_application = "com.apple.product-type.application.messages",
    messages_extension = "com.apple.product-type.app-extension.messages",
    messages_sticker_pack_extension = (
        "com.apple.product-type.app-extension.messages-sticker-pack"
    ),
    quicklook_plugin = "com.apple.product-type.quicklook-plugin",
    spotlight_importer = "com.apple.product-type.spotlight-importer",
    static_framework = "com.apple.product-type.framework.static",
    tool = "com.apple.product-type.tool",
    ui_test_bundle = "com.apple.product-type.bundle.ui-testing",
    unit_test_bundle = "com.apple.product-type.bundle.unit-test",
    watch2_application = "com.apple.product-type.application.watchapp2",
    watch2_extension = "com.apple.product-type.watchkit2-extension",
    xpc_service = "com.apple.product-type.xpc-service",
)
