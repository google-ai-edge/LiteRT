load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("//litert/python:pytype.bzl", "pytype_strict_binary")

package(
    default_visibility = ["//visibility:public"],
)

pytype_strict_binary(
    name = "sdist_wrapper",
    srcs = [
        "sdist_wrapper.py",
    ],
    main = "sdist_wrapper.py",
    deps = [
        "@pypi_setuptools//:pkg",
        "@pypi_wheel//:pkg",
    ],
)

# LiteRT prebuilt archive URL + per-file SHA-256 (JSON from
# `sdist_wrapper.py pin`), set by ci/build_pip_package_with_bazel.sh when it
# builds the SDK sdists. Empty when not pinned (local development builds).
string_flag(
    name = "litert_prebuilts_pin",
    build_setting_default = "",
)
