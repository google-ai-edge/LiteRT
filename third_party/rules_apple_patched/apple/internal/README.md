# Internal Apple bundling logic

The `.bzl` files in this directory are internal implementation details for the
Apple bundling rules. They should not be imported directly by users; instead,
import the platform-specific rules in `//apple:ios.bzl`, `//apple:tvos.bzl`,
and so forth.
