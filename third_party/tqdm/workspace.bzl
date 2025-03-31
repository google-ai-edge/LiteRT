"""This file is used to load the tqdm library."""

load("@org_tensorflow//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "tqdm",
        build_file = "@//third_party/tqdm:tqdm.BUILD",
        sha256 = "e089e5207c36522f28178fe121220c29317afe2995355dda9c33aed1893c5fad",
        strip_prefix = "tqdm-4.67.1",
        urls = tf_mirror_urls(
            "https://github.com/tqdm/tqdm/archive/refs/tags/v4.67.1.tar.gz",
        ),
    )
