package(default_visibility = ["//visibility:public"])

licenses(["reciprocal"])

exports_files(["LICENSE"])  # LICENCE from the original repository

[
    cc_library(
        name = "%s_%s_headers" % (version, arch_name),
        hdrs = glob(
            [
                "%s/%s/include/neuron/api/*.h" % (version, arch_name),
            ],
        ),
        includes = [
            "%s/%s/include/" % (version, arch_name),
        ],
    )
    for version, arch_name in [
        ("v9_latest", "host"),
        ("v8_latest", "host"),
        ("v7_latest", "host"),
    ]
]

exports_files(srcs = glob(["**/*.so"]))
