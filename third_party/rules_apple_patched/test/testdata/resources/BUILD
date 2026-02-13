load(
    "//apple:resources.bzl",
    "apple_bundle_import",
    "apple_precompiled_resource_bundle",
    "apple_resource_bundle",
)

# Public only because these are used by the integration tests from generated
# workspaces. Please no not depend on them as they can change at any time.
package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files([
    "intent.intentdefinition",
    "launch_screen_ios.storyboard",
    "launch_screen_ios.xib",
    "nonlocalized.plist",
    "nonlocalized.strings",
    "nonlocalized_resource.txt",
    "sample.mlmodel",
    "sample.mlpackage",
    "sample.png",
    "storyboard_ios.storyboard",
    "view_ios.xib",
    "storyboard_macos.storyboard",
    "view_macos.xib",
])

filegroup(
    name = "intent_srcs",
    srcs = [
        "BUILD",
        "intent.intentdefinition",
    ],
)

# A convenience target that can be passed into the data attribute of an Apple
# shell test to make the mlmodel file available to builds under test.
filegroup(
    name = "mlmodel_srcs",
    srcs = [
        "BUILD",
        "sample.mlmodel",
        "sample.mlpackage",
    ],
)

# A convenience target that can be passed into the data attribute of an Apple
# shell test to make the iOS-compatible resources available to builds under
# test.
filegroup(
    name = "resource_data_deps_ios",
    srcs = [
        "BUILD",
        "launch_screen_ios.storyboard",
        "launch_screen_ios.xib",
        "storyboard_ios.storyboard",
        "view_ios.xib",
        ":app_icons_ios",
        ":app_icons_ios_with_bad_size",
        ":assets",
        ":assets2",
        ":imageset_missing_child",
        ":imageset_with_invalid_json",
        ":imageset_with_unassigned_child",
        ":launch_images_ios",
        ":localized_storyboards_ios",
        ":localized_xibs_ios",
        ":message_ext_app_icon_ios",
        ":settings_bundle_ios_files",
        ":sticker_pack_ios",
    ],
)

filegroup(
    name = "resource_data_deps_platform_independent",
    srcs = [
        "BUILD",
        "Info-template.plist",
        "nonlocalized.plist",
        "nonlocalized.strings",
        "nonlocalized_resource.txt",
        "sample.png",
        ":basic_bundle_files",
        ":localized_generic_resources",
        ":localized_plists",
        ":localized_strings",
        ":mapping_model",
        ":structured",
        ":unversioned_datamodel",
        ":versioned_datamodel",
    ],
)

filegroup(
    name = "texture_atlas_data_deps",
    srcs = [
        "BUILD",
        ":star_atlas_files",
    ],
)

filegroup(
    name = "app_icons_ios",
    srcs = glob(["app_icons_ios.xcassets/**"]),
)

filegroup(
    name = "app_icons_with_alts_ios",
    srcs = glob(["app_icons_with_alts_ios.xcassets/**"]),
)

filegroup(
    name = "alternate_icons_ios",
    srcs = glob(["alticons/**"]),
)

filegroup(
    name = "assets",
    srcs = glob(["assets.xcassets/**"]),
)

filegroup(
    name = "assets2",
    srcs = glob(["assets2.xcassets/**"]),
)

apple_bundle_import(
    name = "basic_bundle",
    bundle_imports = [":basic_bundle_files"],
)

filegroup(
    name = "basic_bundle_files",
    srcs = glob(["basic.bundle/**"]),
)

genrule(
    name = "generate_strings",
    outs = ["generated.strings"],
    cmd = "echo '\"generated_string\" = \"I like turtles!\";' > $@",
)

genrule(
    name = "generate_xcstrings",
    outs = ["generated.xcstrings"],
    cmd = """
    cat <<EOF > $@
{
  "sourceLanguage": "en",
  "strings": {
    "generated_string": {
      "localizations": {
        "en": {
          "stringUnit": {
            "state": "translated",
            "value": "I like turtles!"
          }
        }
      }
    }
  },
  "version" : "1.0"
}
EOF
""",
)

genrule(
    name = "generate_structured_strings",
    outs = ["structured/generated.strings"],
    cmd = "echo '\"generated_structured_string\" = \"I like turtles too!\";' > $@",
)

genrule(
    name = "generate_structured_xcstrings",
    outs = ["structured/generated.xcstrings"],
    cmd = """
    cat <<EOF > $@
{
  "sourceLanguage": "en",
  "strings": {
    "generated_structured_string": {
      "localizations": {
        "en": {
          "stringUnit": {
            "state": "translated",
            "value": "I like turtles too!"
          }
        }
      }
    }
  },
  "version" : "1.0"
}
EOF
""",
)

genrule(
    name = "generate_storyboard_copy",
    srcs = ["storyboard_ios.storyboard"],
    outs = ["storyboard_ios_copy.storyboard"],
    cmd = "cp $< $@",
)

genrule(
    name = "generate_macos_storyboard_copy",
    srcs = ["storyboard_macos.storyboard"],
    outs = ["storyboard_macos_copy.storyboard"],
    cmd = "cp $< $@",
)

apple_resource_bundle(
    name = "bundle_library_macos",
    bundle_name = "bundle_library_macos",
    # TODO: Bazel seems to crash with an exception if given datamodels
    # and targeting macOS. So until objc_bundle_library is redone in
    # starlark we might just have to live with this.
    infoplists = [
        "Info-template.plist",
    ],
    resources = [
        # TODO: asset_catalogs
        ":basic_bundle",
        ":mapping_model",
        #    ":unversioned_datamodel",
        #    ":versioned_datamodel",
        "nonlocalized.plist",
        ":localized_plists",
        "sample.png",
        "nonlocalized.strings",
        "generated.strings",
        ":localized_strings",
        "nonlocalized_resource.txt",
        ":localized_generic_resources",
        # TODO: storyboards
        # TODO: xibs
    ],
    structured_resources = [
        ":generate_structured_strings",
        ":structured",
    ],
)

# This is just like :bundle_library_* but with a subset of the resources. The
# subset are all the resources that don't require Apple's processing tools
# (actool, ibtool, etc.) which some times fail for "reasons". This then
# provides a bundle that does get some processing, but is less likely to cause
# build/test flake. It is useful for some of the simpler tests looking at
# resource deduping, etc.
apple_resource_bundle(
    name = "simple_bundle_library",
    bundle_name = "simple_bundle_library",
    infoplists = [
        "Info-template.plist",
    ],
    resources = [
        "generated.strings",
        "nonlocalized.plist",
        "nonlocalized.strings",
        "nonlocalized_resource.txt",
        "sample.png",
        ":basic_bundle",
        ":localized_generic_resources",
        ":localized_plists",
        ":localized_strings",
    ],
    structured_resources = [
        ":generate_structured_strings",
        ":structured",
    ],
)

apple_precompiled_resource_bundle(
    name = "simple_precompiled_bundle_library",
    bundle_id = "org.bazel.simple-precompiled-bundle-library",
    bundle_name = "simple_precompiled_bundle_library",
    infoplists = [
        "Info-template.plist",
    ],
    resources = [
        "generated.strings",
        "nonlocalized.plist",
        "nonlocalized.strings",
        "nonlocalized_resource.txt",
        "sample.png",
        ":basic_bundle",
        ":localized_generic_resources",
        ":localized_plists",
        ":localized_strings",
    ],
    structured_resources = [
        ":generate_structured_strings",
        ":structured",
    ],
)

filegroup(
    name = "launch_images_ios",
    srcs = glob(["launch_images_ios.xcassets/**"]),
)

filegroup(
    name = "localized_generic_resources",
    srcs = glob(["*.lproj/*.txt"]),
)

filegroup(
    name = "localized_plists",
    srcs = glob(["*.lproj/*.plist"]),
)

filegroup(
    name = "localized_storyboards_ios",
    srcs = glob(["*.lproj/*_ios.storyboard"]),
)

filegroup(
    name = "localized_storyboards_macos",
    srcs = glob(["*.lproj/*_macos.storyboard"]),
)

filegroup(
    name = "localized_strings",
    srcs = glob(["*.lproj/*.strings"]),
)

filegroup(
    name = "localized_xcstrings",
    srcs = glob(["*.xcstrings"]),
)

filegroup(
    name = "localized_xibs_ios",
    srcs = glob(["*.lproj/*_ios.xib"]),
)

filegroup(
    name = "localized_xibs_macos",
    srcs = glob(["*.lproj/*_macos.xib"]),
)

filegroup(
    name = "mapping_model",
    srcs = glob(["mapping_model.xcmappingmodel/**"]),
)

apple_bundle_import(
    name = "settings_bundle_ios",
    bundle_imports = [":settings_bundle_ios_files"],
)

filegroup(
    name = "settings_bundle_ios_files",
    srcs = glob(["settings_ios.bundle/**"]),
)

filegroup(
    name = "star_atlas_files",
    srcs = glob(["star.atlas/**"]),
)

filegroup(
    name = "imageset_missing_child",
    srcs = glob(["imageset_missing_child.xcassets/**"]),
)

filegroup(
    name = "imageset_with_unassigned_child",
    srcs = glob(["imageset_with_unassigned_child.xcassets/**"]),
)

filegroup(
    name = "app_icons_ios_with_bad_size",
    srcs = glob(["app_icons_ios_with_bad_size.xcassets/**"]),
)

filegroup(
    name = "imageset_with_invalid_json",
    srcs = glob(["imageset_with_invalid_json.xcassets/**"]),
)

filegroup(
    name = "sticker_pack_ios",
    srcs = glob(["sticker_pack_ios.xcstickers/**"]),
)

filegroup(
    name = "message_ext_app_icon_ios",
    srcs = glob(["message_ext_app_icon_ios.xcassets/**"]),
)

filegroup(
    name = "structured",
    srcs = glob(["structured/**"]),
)

filegroup(
    name = "unversioned_datamodel",
    srcs = glob(["unversioned_datamodel.xcdatamodel/**"]),
)

filegroup(
    name = "versioned_datamodel",
    srcs = glob(["versioned_datamodel.xcdatamodeld/**"]),
)

filegroup(
    name = "metal_files",
    srcs = [
        ":metal_hdrs",
        ":metal_srcs",
    ],
)

filegroup(
    name = "metal_srcs",
    srcs = glob(["*.metal"]),
)

filegroup(
    name = "metal_hdrs",
    srcs = ["shader_defines.h"],
)
