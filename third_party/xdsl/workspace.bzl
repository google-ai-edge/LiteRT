"""This file is used to load the xdsl library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "xdsl",
        build_file = "@//third_party/xdsl:xdsl.BUILD",
        strip_prefix = "xdsl-0.28.0/xdsl",
        urls = [
            "https://github.com/xdslproject/xdsl/archive/refs/tags/v0.28.0.tar.gz",
        ],
    )
