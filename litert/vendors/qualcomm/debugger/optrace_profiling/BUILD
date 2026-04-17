load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//litert:__subpackages__"],
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    data = [
        "assets/config.json",
        "assets/config_viewer.json",
        "assets/htp_ext_config.json",
    ],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@org_tensorflow//third_party/py/numpy",
    ],
)
