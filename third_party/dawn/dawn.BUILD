# Build targets for open-source Dawn repository.

load("@xla//third_party/rules_python/python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "dawn_json_generator",
    srcs = [
        "generator/dawn_json_generator.py",
        "generator/generator_lib.py",
    ],
    imports = [
        "generator",
    ],
    deps = [
        "@jinja2",
    ],
)

genrule(
    name = "generate_webgpu_cpp",
    srcs = glob([
        "generator/templates/**/*",
    ]) + [
        "src/dawn/dawn.json",
        "src/dawn/dawn_wire.json",
    ],
    outs = [
        "include/dawn/webgpu_cpp.h",
        "include/dawn/webgpu_cpp_print.h",
        "include/webgpu/webgpu_cpp_chained_struct.h",
    ],
    cmd = "$(location :dawn_json_generator) " +
          "--dawn-json $(location src/dawn/dawn.json) " +
          "--wire-json $(location src/dawn/dawn_wire.json) " +
          "--template-dir $$(dirname $(location generator/templates/api_cpp.h)) " +
          "--targets cpp_headers " +
          "--output-dir $(RULEDIR)",
    tools = [":dawn_json_generator"],
)

genrule(
    name = "generate_webgpu",
    srcs = glob([
        "generator/templates/**/*",
    ]) + [
        "src/dawn/dawn.json",
        "src/dawn/dawn_wire.json",
    ],
    outs = [
        "include/dawn/webgpu.h",
        "include/dawn/dawn_proc_table.h",
    ],
    cmd = "$(location :dawn_json_generator) " +
          "--dawn-json $(location src/dawn/dawn.json) " +
          "--wire-json $(location src/dawn/dawn_wire.json) " +
          "--template-dir $$(dirname $(location generator/templates/api.h)) " +
          "--targets headers " +
          "--output-dir $(RULEDIR)",
    tools = [":dawn_json_generator"],
)

cc_library(
    name = "webgpu_headers",
    hdrs = [
        ":generate_webgpu",
        ":generate_webgpu_cpp",
    ] + glob([
        "include/**/*.h",
        "src/**/*.h",
    ]),
    includes = [
        "include",
        "src",
    ],
)

cc_library(
    name = "dawn_headers",
    hdrs = [
        ":generate_webgpu",
    ] + glob([
        "include/**/*.h",
    ]),
    includes = [
        "include",
    ],
)

cc_library(
    name = "dawncpp_headers",
    hdrs = [
        ":generate_webgpu_cpp",
    ] + glob([
        "include/**/*.h",
    ]),
    includes = [
        "include",
    ],
)

cc_library(
    name = "dawn_native",
    hdrs = [
        ":generate_webgpu",
        ":generate_webgpu_cpp",
    ] + glob([
        "include/**/*.h",
        "src/**/*.h",
    ]),
    includes = [
        "include",
        "src",
    ],
    deps = [
        ":webgpu_headers",
    ],
)

cc_library(
    name = "webgpu_dawn",
    hdrs = [
        ":generate_webgpu",
        ":generate_webgpu_cpp",
    ] + glob([
        "include/**/*.h",
        "src/**/*.h",
    ]),
    includes = [
        "include",
        "src",
    ],
    deps = [
        ":dawn_native",
        ":webgpu_headers",
    ],
)

cc_library(
    name = "libdawn_proc",
    hdrs = [
        ":generate_webgpu",
    ] + glob([
        "include/**/*.h",
    ]),
    includes = [
        "include",
    ],
)
