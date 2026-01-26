"""Definitions for py_wheel platforms."""

load("@bazel_skylib//lib:selects.bzl", "selects")
load("//litert:litert.bzl", "MANYLINUX_LEVEL")

def get_wheel_platform_name():
    selects.config_setting_group(
        name = "linux_x86_64",
        match_all = [
            "@platforms//os:linux",
            "@platforms//cpu:x86_64",
        ],
    )

    selects.config_setting_group(
        name = "linux_arm64",
        match_all = [
            "@platforms//os:linux",
            "@platforms//cpu:arm64",
        ],
    )

    selects.config_setting_group(
        name = "macos_arm64",
        match_all = [
            "@platforms//os:macos",
            "@platforms//cpu:arm64",
        ],
    )

    selects.config_setting_group(
        name = "windows_x86_64",
        match_all = [
            "@platforms//os:windows",
            "@platforms//cpu:x86_64",
        ],
    )

    selects.config_setting_group(
        name = "windows_arm64",
        match_all = [
            "@platforms//os:windows",
            "@platforms//cpu:arm64",
        ],
    )

    return select({
        ":linux_x86_64": MANYLINUX_LEVEL + "_x86_64",
        ":linux_arm64": MANYLINUX_LEVEL + "_aarch64",
        ":macos_arm64": "macosx_12_0_arm64",
        ":windows_x86_64": "win_amd64",
        ":windows_arm64": "win_arm64",
        "//conditions:default": "none",
    })
