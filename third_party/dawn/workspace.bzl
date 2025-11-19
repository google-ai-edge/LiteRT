"""This file is used to load the dawn library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "dawn",
        add_prefix = "dawn",
        urls = [
            "https://github.com/google/dawn/archive/v20250713.025201.tar.gz",
        ],
        build_file = "//third_party/dawn:BUILD",
    )
