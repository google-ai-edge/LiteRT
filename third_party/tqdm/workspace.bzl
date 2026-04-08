"""This file is used to load the tqdm library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "tqdm",
        build_file = "@//third_party/tqdm:tqdm.BUILD",
        sha256 = "8a66e36475bcfca29b4808d61ee73591f8d92d273899de60360ced4d68364d3a",
        strip_prefix = "tqdm-4.67.3",
        urls = [
            "https://github.com/tqdm/tqdm/archive/refs/tags/v4.67.3.tar.gz",
        ],
    )
