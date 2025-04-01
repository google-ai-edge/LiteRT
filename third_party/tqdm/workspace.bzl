"""This file is used to load the tqdm library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "tqdm",
        build_file = "@//third_party/tqdm:tqdm.BUILD",
        add_prefix = "tqdm",
        urls = [
            "https://third-party-mirror.googlesource.com/tqdm/+archive/d593e871a6b3fcc21ca5281aebda0feee0e8732e.tar.gz",
        ],
    )
