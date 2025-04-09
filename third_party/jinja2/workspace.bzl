"""This file is used to load the jinja2 library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "jinja2",
        build_file = "@//third_party/jinja2:jinja2.BUILD",
        url = "https://github.com/pallets/jinja/archive/3.0.1.tar.gz",
        strip_prefix = "jinja-3.0.1/src",
        sha256 = "1e37a6f86c29fa8ace108ea72b41d2d5c5bd67d79be14bfeca3ba6eb37d789de",
    )
