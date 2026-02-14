# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Setting up your development environment

To enforce a consistent code style through our code base, we use `buildifier`
from the [bazelbuild/buildtools](https://github.com/bazelbuild/buildtools) to
format `BUILD` and `*.bzl` files. We also use `buildifier --lint=warn` to check
for common issues.

You can download `buildifier` from
[bazelbuild/buildtools Releases Page](https://github.com/bazelbuild/buildtools/releases).

Bazel's CI is configured to ensure that files in pull requests are formatted
correctly and that there are no lint issues.

## Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google.com/conduct/).
