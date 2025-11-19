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

exports_files(glob(["**/*.so"]))
