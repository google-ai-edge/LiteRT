load("@rules_python//python:py_binary.bzl", "py_binary")

py_binary(
    name = "local_provisioning_profile_finder",
    srcs = ["local_provisioning_profile_finder.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    # Used by the rule implementations, so it needs to be public; but
    # should be considered an implementation detail of the rules and
    # not used by other things.
    visibility = ["//visibility:public"],
)
