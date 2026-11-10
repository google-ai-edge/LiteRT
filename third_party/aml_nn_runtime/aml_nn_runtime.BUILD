# BUILD for @aml_npu_nn_runtime (nn_runtime after strip_prefix / LITERT_AML_NN_RUNTIME).
#
# Layout (GitHub amlnn-toolkit/.../nn_runtime):
#   nnsdk/include/...          (+ android/lib64|lib32, linux/...)
#   aml_compiler_core/include/ (+ libs/{android,linux,yocto}/...)
#
# nnsdk: headers only (dlopen libnnsdk.so).
# aml_compiler_core: headers + cc_import libaml_compiler_core.so.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# -------------------- nnsdk (headers; optional packaging .so) --------------------
cc_library(
    name = "nnsdk_headers",
    hdrs = glob(
        ["nnsdk/include/**/*.h"],
        allow_empty = True,
    ),
    includes = [
        "nnsdk/include",
    ],
)

# GitHub: nnsdk/android/lib64|lib32; also accept arm64-v8a / armeabi-v7a if present.
filegroup(
    name = "nnsdk_android_so_arm64",
    srcs = glob(
        [
            "nnsdk/android/lib64/*.so",
            "nnsdk/android/arm64-v8a/*.so",
        ],
        allow_empty = True,
    ),
)

filegroup(
    name = "nnsdk_android_so_arm",
    srcs = glob(
        [
            "nnsdk/android/lib32/*.so",
            "nnsdk/android/armeabi-v7a/*.so",
        ],
        allow_empty = True,
    ),
)

# -------------------- aml_compiler_core (link + package) --------------------
cc_library(
    name = "headers",
    hdrs = glob(
        ["aml_compiler_core/include/**/*.h"],
        allow_empty = True,
    ),
    includes = ["aml_compiler_core/include"],
    include_prefix = "aml_npu/compiler_core",
    strip_include_prefix = "aml_compiler_core/include",
)

cc_import(
    name = "_so_android_arm64",
    shared_library = "aml_compiler_core/libs/android/arm64-v8a/libaml_compiler_core.so",
)

cc_import(
    name = "_so_android_armv7",
    shared_library = "aml_compiler_core/libs/android/armeabi-v7a/libaml_compiler_core.so",
)

cc_import(
    name = "_so_linux_x86_64",
    shared_library = "aml_compiler_core/libs/linux/x86_64/libaml_compiler_core.so",
)

cc_import(
    name = "_so_linux_aarch64",
    shared_library = "aml_compiler_core/libs/linux/aarch64/libaml_compiler_core.so",
)

cc_library(
    name = "aml_compiler_core_lib",
    deps = [":headers"] + select({
        "@//third_party/aml_nn_runtime:aml_abi_arm64": [":_so_android_arm64"],
        "@//third_party/aml_nn_runtime:aml_abi_armv7": [":_so_android_armv7"],
        "@//third_party/aml_nn_runtime:aml_abi_x86_64": [":_so_linux_x86_64"],
        "//conditions:default": [":_so_android_arm64"],
    }),
)

alias(
    name = "aml_compiler_core",
    actual = ":aml_compiler_core_lib",
)

filegroup(
    name = "aml_compiler_core_so_arm64",
    srcs = ["aml_compiler_core/libs/android/arm64-v8a/libaml_compiler_core.so"],
)

filegroup(
    name = "aml_compiler_core_so_arm",
    srcs = ["aml_compiler_core/libs/android/armeabi-v7a/libaml_compiler_core.so"],
)

filegroup(
    name = "aml_compiler_core_so_x86",
    srcs = ["aml_compiler_core/libs/linux/x86_64/libaml_compiler_core.so"],
)

filegroup(
    name = "aml_compiler_core_so_linux_aarch64",
    srcs = ["aml_compiler_core/libs/linux/aarch64/libaml_compiler_core.so"],
)

filegroup(
    name = "aml_compiler_core_so_yocto_aarch64",
    srcs = glob(
        ["aml_compiler_core/libs/yocto/aarch64/libaml_compiler_core.so"],
        allow_empty = True,
    ),
)

# Packaging: compiler_adla next to compiler_core on GitHub (not linked into LiteRT).
filegroup(
    name = "compiler_adla_android_so_arm64",
    srcs = glob(
        ["aml_compiler_core/libs/android/arm64-v8a/libcompiler_adla.so"],
        allow_empty = True,
    ),
)

filegroup(
    name = "compiler_adla_android_so_arm",
    srcs = glob(
        ["aml_compiler_core/libs/android/armeabi-v7a/libcompiler_adla.so"],
        allow_empty = True,
    ),
)

filegroup(
    name = "compiler_adla_so_x86",
    srcs = glob(
        ["aml_compiler_core/libs/linux/x86_64/libcompiler_adla.so"],
        allow_empty = True,
    ),
)

filegroup(
    name = "compiler_adla_so_linux_aarch64",
    srcs = glob(
        [
            "aml_compiler_core/libs/linux/aarch64/libcompiler_adla.so",
            "aml_compiler_core/libs/yocto/aarch64/libcompiler_adla.so",
        ],
        allow_empty = True,
    ),
)

# Convenience: everything typically needed under jni/<abi>/ besides the JNI .so.
filegroup(
    name = "android_aar_native_arm64",
    srcs = [
        ":aml_compiler_core_so_arm64",
        ":compiler_adla_android_so_arm64",
        ":nnsdk_android_so_arm64",
    ],
)

filegroup(
    name = "android_aar_native_arm",
    srcs = [
        ":aml_compiler_core_so_arm",
        ":compiler_adla_android_so_arm",
        ":nnsdk_android_so_arm",
    ],
)

exports_files(glob(["aml_compiler_core/libs/**/*.so"], allow_empty = True))
