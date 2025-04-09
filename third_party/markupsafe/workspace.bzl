"""This file is used to load the markupsafe library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "markupsafe",
        build_file = "@//third_party/markupsafe:markupsafe.BUILD",
        sha256 = "0f83b6d1bf6fa65546221d42715034e7e654845583a84906c5936590f9a7ad8f",
        strip_prefix = "markupsafe-2.1.1/src/markupsafe",
        urls = [
            "https://github.com/pallets/markupsafe/archive/2.1.1.tar.gz",
        ],
    )
