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

"""Partial implementation for app assets validation."""

load(
    "@bazel_skylib//lib:partial.bzl",
    "partial",
)
load(
    "//apple/internal:apple_product_type.bzl",
    "apple_product_type",
)

# Standard icon extensions as of Xcode 26 for most Apple platforms (iOS, macOS, watchOS).
_STANDARD_ICONS = [".appiconset/", ".icon/"]

# Valid icon extensions for specific product types that have exceptional requirements, independent
# of platform.
_VALID_ICON_EXTENSIONS_FOR_PRODUCT_TYPE = {
    apple_product_type.messages_extension: [".stickersiconset/"],
    apple_product_type.messages_sticker_pack_extension: [
        # Replacement for appiconset.
        ".stickersiconset/",
        # The stickers. Ideally we wouldn't list these here, but `app_assets_validation_partial`
        # is called with `sticker_assets` instead of just the app icons.
        ".stickerpack/",
        ".sticker/",
        ".stickersequence/",
    ],
}

# Comprehensive list of all valid icon extensions for each platform. These cover apps, extensions,
# app clips, and bundle types that can use icons.
_VALID_ICON_EXTENSIONS_FOR_PLATFORM = {
    "ios": _STANDARD_ICONS,
    "macos": _STANDARD_ICONS,
    "tvos": [".brandassets/"],
    "watchos": _STANDARD_ICONS,
    "visionos": [".solidimagestack/"],
}

def _app_assets_validation_partial_impl(
        *,
        app_icons,
        launch_images,
        platform_prerequisites,
        product_type):
    """Implementation for the app assets processing partial."""

    # actool.bzl has the most comprehensive validations since it evaluates the final set of files
    # before they are sent to actool. We only check here that the user is sending files that look
    # like they could be app icons via an attribute named `app_icons`, and likewise for launch
    # images.

    if app_icons:
        valid_icon_extensions = (
            _VALID_ICON_EXTENSIONS_FOR_PRODUCT_TYPE.get(product_type, None) or
            _VALID_ICON_EXTENSIONS_FOR_PLATFORM[str(platform_prerequisites.platform_type)]
        )
        for resource in app_icons:
            resource_short_path = resource.short_path
            possible_valid_icon = False
            for valid_icon_extension in valid_icon_extensions:
                if valid_icon_extension in resource_short_path:
                    possible_valid_icon = True
                    break
            if (not possible_valid_icon and
                not resource_short_path.endswith(".xcassets/Contents.json") and
                not resource_short_path.endswith(".xcstickers/Contents.json")):
                fail("""
Found in app_icons a file that cannot be used as an app icon:
{resource_short_path}

Valid icon bundles for this target have the following extensions: {valid_icon_extensions}
""".format(
                    resource_short_path = resource_short_path,
                    valid_icon_extensions = valid_icon_extensions,
                ))

    if launch_images:
        for resource in launch_images:
            resource_short_path = resource.short_path
            if (not ".launchimage/" in resource_short_path and
                not resource_short_path.endswith(".xcassets/Contents.json")):
                fail("""
Found in launch_images a file that cannot be used as a launch image:
{resource_short_path}

All launch images must be in a directory named '*.launchimage' within an '*.xcassets' directory.
""".format(
                    resource_short_path = resource_short_path,
                ))

    return struct()

def app_assets_validation_partial(
        *,
        app_icons = [],
        launch_images = [],
        platform_prerequisites,
        product_type):
    """Constructor for the app assets validation partial.

    This partial validates the given app_icons and launch_images are correct for the current
    product type.

    Args:
        app_icons: List of files that represents the App icons.
        launch_images: List of files that represent the launch images.
        platform_prerequisites: Struct containing information on the platform being targeted.
        product_type: Product type identifier used to describe the current bundle type.

    Returns:
        A partial that validates app assets.
    """
    return partial.make(
        _app_assets_validation_partial_impl,
        app_icons = app_icons,
        launch_images = launch_images,
        platform_prerequisites = platform_prerequisites,
        product_type = product_type,
    )
