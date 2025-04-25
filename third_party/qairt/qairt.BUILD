package(default_visibility = ["//visibility:public"])

licenses(["reciprocal"])

exports_files(["LICENSE"])  # LICENCE from the original repository

cc_library(
    name = "qnn_lib_headers",
    hdrs = glob(
        [
            "latest/include/QNN/*.h",
            "latest/include/QNN/CPU/*.h",
            "latest/include/QNN/HTP/*.h",
            "latest/include/QNN/System/*.h",
        ],
        exclude = [
            "latest/include/QNN/HTA/**/*.h",
            "latest/include/QNN/TFLiteDelegate/**/*.h",
        ],
    ),
    includes = [
        "latest/include/QNN/",
    ],
    visibility = ["//visibility:public"],
)
