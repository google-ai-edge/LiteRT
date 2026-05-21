package(default_visibility = ["//visibility:public"])

licenses(["reciprocal"])

exports_files(["LICENSE"])  # LICENCE from the original repository

cc_library(
    name = "qnn_lib_headers",
    hdrs = glob(
        [
            "include/QNN/*.h",
            "include/QNN/CPU/*.h",
            "include/QNN/HTP/*.h",
            "include/QNN/System/*.h",
            "include/QNN/IR/*.h",
            "include/QNN/DSP/*.h",
            "include/QNN/Saver/*.h",
        ],
        exclude = [
            "include/QNN/HTA/**/*.h",
            "include/QNN/TFLiteDelegate/**/*.h",
        ],
    ),
    includes = [
        "include/QNN/",
    ],
    visibility = ["//visibility:public"],
)

alias(
    name = "libQnnHtp.so",
    actual = select({
        "@platforms//os:android": "lib/aarch64-android/libQnnHtp.so",
        "//conditions:default": "lib/x86_64-linux-clang/libQnnHtp.so",
    }),
)

alias(
    name = "libQnnSystem.so",
    actual = select({
        "@platforms//os:android": "lib/aarch64-android/libQnnSystem.so",
        "//conditions:default": "lib/x86_64-linux-clang/libQnnSystem.so",
    }),
)

alias(
    name = "libQnnIr.so",
    actual = select({
        "@platforms//os:android": "lib/aarch64-android/libQnnIr.so",
        "//conditions:default": "lib/x86_64-linux-clang/libQnnIr.so",
    }),
)

alias(
    name = "libQnnSaver.so",
    actual = select({
        "@platforms//os:android": "lib/aarch64-android/libQnnSaver.so",
        "//conditions:default": "lib/x86_64-linux-clang/libQnnSaver.so",
    }),
)

alias(
    name = "libQnnHtpPrepare.so",
    actual = select({
        "@platforms//os:android": "lib/aarch64-android/libQnnHtpPrepare.so",
        "//conditions:default": "lib/x86_64-linux-clang/libQnnHtpPrepare.so",
    }),
)

alias(
    name = "libQnnHtpV75Stub.so",
    actual = "lib/aarch64-android/libQnnHtpV75Stub.so",
)

alias(
    name = "libQnnHtpV75Skel.so",
    actual = "lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so",
)

alias(
    name = "libQnnHtpV79Stub.so",
    actual = "lib/aarch64-android/libQnnHtpV79Stub.so",
)

alias(
    name = "libQnnHtpV79Skel.so",
    actual = "lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so",
)

exports_files(glob(["**/*.so"]))
