load("@bazel_skylib//rules:build_test.bzl", "build_test")
load(
    "//apple:ios.bzl",
    "ios_imessage_application",
    "ios_sticker_pack_extension",
)
load(
    "//apple:versioning.bzl",
    "apple_bundle_version",
)

licenses(["notice"])

apple_bundle_version(
    name = "StickerAppVersion",
    build_version = "1.0",
)

ios_sticker_pack_extension(
    name = "StickersExtension",
    bundle_id = "com.example.StickersApp.StickersExtension",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = ["StickersExtension-Info.plist"],
    minimum_os_version = "11.0",
    sticker_assets = glob(["Stickers.xcstickers/**"]),
    version = ":StickerAppVersion",
)

ios_imessage_application(
    name = "StickersApp",
    app_icons = glob(["StickersApp.xcassets/**"]),
    bundle_id = "com.example.StickersApp",
    extension = ":StickersExtension",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = ["StickersApp-Info.plist"],
    minimum_os_version = "11.0",
    version = ":StickerAppVersion",
)

# Not normally needed, just done for rules_apple's examples so a
# 'bazel test examples/...' ensures all Examples still build.
build_test(
    name = "ExamplesBuildTest",
    targets = [
        ":StickersApp",
        ":StickersExtension",
    ],
)
