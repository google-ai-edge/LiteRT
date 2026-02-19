"""
Rule for packaging a bundle into a .xcarchive.
"""

load(
    "//apple:providers.bzl",
    "AppleBundleInfo",
    "AppleDsymBundleInfo",
)
load(
    "//apple/internal:providers.bzl",
    "new_applebinaryinfo",
)
load(
    "//apple/internal/providers:apple_debug_info.bzl",
    "AppleDebugInfo",
)

def _xcarchive_impl(ctx):
    """
    Implementation for xcarchive.

    This rule uses the providers from the bundle target to re-package it into a .xcarchive.
    The .xcarchive is a directory that contains the .app bundle, dSYM and other metadata.
    """
    bundle_info = ctx.attr.bundle[AppleBundleInfo]
    dsym_info = ctx.attr.bundle[AppleDsymBundleInfo]
    debug_info = ctx.attr.bundle[AppleDebugInfo]
    xcarchive = ctx.actions.declare_directory("%s.xcarchive" % bundle_info.bundle_name)

    arguments = ctx.actions.args()
    arguments.add("--info_plist", bundle_info.infoplist.path)
    arguments.add("--bundle", bundle_info.archive.path)
    arguments.add("--output", xcarchive.path)

    arguments.add_all(
        dsym_info.transitive_dsyms,
        before_each = "--dsym",
        expand_directories = False,
    )

    linkmaps = debug_info.linkmaps.to_list()
    for linkmap in linkmaps:
        arguments.add("--linkmap", linkmap.path)

    ctx.actions.run(
        inputs = depset(
            [
                bundle_info.archive,
                bundle_info.infoplist,
            ] + linkmaps,
            transitive = [dsym_info.transitive_dsyms],
        ),
        outputs = [xcarchive],
        executable = ctx.executable._make_xcarchive,
        arguments = [arguments],
        mnemonic = "XCArchive",
    )

    # Limiting the contents of AppleBinaryInfo to what is necessary for testing and validation.
    xcarchive_binary_info = new_applebinaryinfo(
        binary = xcarchive,
        infoplist = None,
        product_type = None,
    )

    return [
        DefaultInfo(files = depset([xcarchive])),
        xcarchive_binary_info,
    ]

xcarchive = rule(
    implementation = _xcarchive_impl,
    attrs = {
        "bundle": attr.label(
            providers = [
                AppleBundleInfo,
                AppleDsymBundleInfo,
                AppleDebugInfo,
            ],
            doc = """\
The label to a target to re-package into a .xcarchive. For example, an
`ios_application` target.
            """,
        ),
        "_make_xcarchive": attr.label(
            default = Label("//tools/xcarchivetool:make_xcarchive"),
            executable = True,
            cfg = "exec",
            doc = """\
An executable binary that can re-package a bundle into a .xcarchive.
            """,
        ),
    },
    doc = """\
Re-packages an Apple bundle into a .xcarchive.

This rule uses the providers from the bundle target to construct the required
metadata for the .xcarchive.

Example:

````starlark
load("//apple:xcarchive.bzl", "xcarchive")

ios_application(
    name = "App",
    bundle_id = "com.example.my.app",
    ...
)

xcarchive(
    name = "App.xcarchive",
    bundle = ":App",
)
````
    """,
)
