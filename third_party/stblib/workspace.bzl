"""This file is used to load the stblib library."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "stblib",
        remote = "https://github.com/nothings/stb",
        commit = "c0c982601f40183e74d84a61237e968dca08380e",
        build_file = "@//third_party/stblib:stblib.BUILD",
    )
