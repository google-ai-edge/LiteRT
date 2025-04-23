# Description: Jinja2 library.

package(default_visibility = ["//visibility:public"])

license(
    name = "license",
    package_name = "jinja2",
)

licenses(["notice"])

py_library(
    name = "jinja2",
    srcs = glob(["jinja2/*.py"]),
    visibility = ["//visibility:public"],
    deps = ["@markupsafe"],
)
