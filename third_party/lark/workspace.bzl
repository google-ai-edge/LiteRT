"""This file is used to load the lark library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "lark",
        build_file = "@//third_party/lark:lark.BUILD",
        add_prefix = "lark-1.3.1/lark",
        urls = [
            "https://github.com/lark-parser/lark/archive/refs/tags/1.3.1.tar.gz",
        ],
    )
