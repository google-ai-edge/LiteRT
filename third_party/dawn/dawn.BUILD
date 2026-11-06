# Build targets for open-source Dawn repository.

load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "dawn_json_generator",
    srcs = [
        "generator/dawn_json_generator.py",
        "generator/generator_lib.py",
        "generator/webgpu_docs_utility.py",
    ],
    imports = [
        "generator",
    ],
    # Disable implicit __init__.py creation in runfiles so @jinja2 is not shadowed.
    legacy_create_init = 0,
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
    deps = [
        "@com_google_absl//absl/types:span",
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
    deps = [
        "@com_google_absl//absl/types:span",
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

genrule(
    name = "generate_webgpu_dawn_stub",
    srcs = ["include/dawn/webgpu.h"],
    outs = ["webgpu_dawn_stub.cc"],
    cmd = "sed -n -E 's/.*WGPU_EXPORT .* (wgpu[A-Za-z0-9_]+)\\(.*/extern \"C\" __declspec(dllexport) void \\1() {}/p' $(location include/dawn/webgpu.h) > $@",
    target_compatible_with = ["@platforms//os:windows"],
)

cc_binary(
    name = "stub/webgpu_dawn.dll",
    srcs = [":webgpu_dawn_stub.cc"],
    linkshared = 1,
    target_compatible_with = ["@platforms//os:windows"],
)

filegroup(
    name = "webgpu_dawn_interface_lib",
    srcs = [":stub/webgpu_dawn.dll"],
    output_group = "interface_library",
    target_compatible_with = ["@platforms//os:windows"],
)

cc_import(
    name = "webgpu_dawn_dll_import",
    interface_library = ":webgpu_dawn_interface_lib",
    system_provided = 1,
    target_compatible_with = ["@platforms//os:windows"],
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
    linkopts = select({
        "@platforms//os:windows": [
            "/DELAYLOAD:webgpu_dawn.dll",
            "delayimp.lib",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":dawn_native",
        ":webgpu_headers",
    ] + select({
        "@platforms//os:windows": [":webgpu_dawn_dll_import"],
        "//conditions:default": [],
    }),
)

genrule(
    name = "generate_dawn_version",
    outs = ["include/dawn/dawn_version.h"],
    cmd = "printf '%s\\n' " +
          "'#ifndef INCLUDE_DAWN_DAWN_VERSION_H_' " +
          "'#define INCLUDE_DAWN_DAWN_VERSION_H_' " +
          "'#include <array>' " +
          "'#include <cstdint>' " +
          "'namespace dawn {' " +
          "'static constexpr std::array<uint8_t, 20> kDawnVersion = {};' " +
          "'}  // namespace dawn' " +
          "'#endif  // INCLUDE_DAWN_DAWN_VERSION_H_' > $@",
)

genrule(
    name = "generate_dawn_proc",
    srcs = glob([
        "generator/templates/**/*",
    ]) + [
        "src/dawn/dawn.json",
        "src/dawn/dawn_wire.json",
    ],
    outs = [
        "src/dawn/dawn_proc.cpp",
        "src/dawn/dawn_thread_dispatch_proc.cpp",
    ],
    cmd = "$(location :dawn_json_generator) " +
          "--dawn-json $(location src/dawn/dawn.json) " +
          "--wire-json $(location src/dawn/dawn_wire.json) " +
          "--template-dir $$(dirname $(location generator/templates/dawn_proc.cpp)) " +
          "--targets proc " +
          "--output-dir $(RULEDIR)",
    tools = [":dawn_json_generator"],
)

cc_library(
    name = "libdawn_proc",
    srcs = [
        "src/utils/log.cc",
        ":generate_dawn_proc",
        ":generate_dawn_version",
    ],
    hdrs = [
        "src/dawn/common/Compiler.h",
        ":generate_webgpu",
    ] + glob([
        "include/**/*.h",
        "src/utils/*.h",
    ]),
    copts = ["-DDAWN_DISABLE_LOGGING"],
    includes = [
        ".",
        "include",
        "src",
    ],
)
