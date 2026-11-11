"""This file is used to load the dawn library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "dawn",
        build_file = Label("//third_party/dawn:dawn.BUILD"),
        # Backports api_cpp.h C++20 constructs (std::span, templated lambdas,
        # requires-clauses) to C++17.
        patches = [Label("//third_party/dawn:cpp17_compat.patch")],
        patch_args = ["-p1"],
        strip_prefix = "dawn-20260720.160313",
        urls = [
            "https://github.com/google/dawn/archive/v20260720.160313.tar.gz",
        ],
    )
