package(default_visibility = ["//visibility:public"])

licenses(["reciprocal"])

exports_files(["LICENSE"])  # LICENCE from the original repository

[
    cc_library(
        name = "%s_%s_headers" % (version, arch_name),
        defines = [
            'LITERT_NEURO_PILOT_V9_HOST_LIB_DIR=\\"external/neuro_pilot/v9_latest/host/lib\\"',
            'LITERT_NEURO_PILOT_V8_HOST_LIB_DIR=\\"external/neuro_pilot/v8_latest/host/lib\\"',
            'LITERT_NEURO_PILOT_V7_HOST_LIB_DIR=\\"external/neuro_pilot/v7_latest/host/lib\\"',
        ],
        includes = [
            "%s/%s/include/" % (version, arch_name),
        ],
        linkopts = select({
            "@platforms//os:android": [],
            "//conditions:default": [
                "-Wl,-rpath=%s" % ":".join([
                    "external/neuro_pilot/%s/%s/lib" % (version, arch_name),
                    "external/neuro_pilot/v9_latest/host/lib",
                    "external/neuro_pilot/v8_latest/host/lib",
                    "external/neuro_pilot/v7_latest/host/lib",
                ]),
            ],
        }),
        textual_hdrs = glob(
            [
                "%s/%s/include/neuron/api/*.h" % (version, arch_name),
            ],
            allow_empty = True,
        ),
    )
    for version, arch_name in [
        ("v9_latest", "host"),
        ("v8_latest", "host"),
        ("v7_latest", "host"),
    ]
]

alias(
    name = "neuron_host_headers",
    actual = ":v8_latest_host_headers",
)

alias(
    name = "latest_host_headers",
    actual = ":neuron_host_headers",
)

[
    filegroup(
        name = "v%s_libneuron_adapter_so" % major,
        srcs = glob(
            ["v%s_latest/host/lib/libneuron_adapter.so" % major],
            allow_empty = True,
        ),
    )
    for major in [
        "9",
        "8",
        "7",
    ]
]

[
    filegroup(
        name = "v%s_libcxx_so" % major,
        srcs = glob(
            ["v%s_latest/host/lib/libc++.so.1" % major],
            allow_empty = True,
        ),
    )
    for major in [
        "9",
        "8",
        "7",
    ]
]

[
    filegroup(
        name = "v%s_libunwind_so" % major,
        srcs = glob(
            ["v%s_latest/host/lib/libunwind.so.1" % major],
            allow_empty = True,
        ),
    )
    for major in [
        "9",
        "8",
        "7",
    ]
]

[
    filegroup(
        name = "v%s_host_libs" % major,
        srcs = [
            ":v%s_libcxx_so" % major,
            ":v%s_libneuron_adapter_so" % major,
            ":v%s_libunwind_so" % major,
        ],
    )
    for major in [
        "9",
        "8",
        "7",
    ]
]

filegroup(
    name = "common_host_libs",
    srcs = glob(
        [
            "neuron_sdk/host/lib/libatomic.so.1",
            "neuron_sdk/host/lib/libc++abi.so.1",
            "neuron_sdk/host/lib/libgcc_s.so.1",
        ],
        allow_empty = True,
    ),
)

alias(
    name = "libneuron_adapter.so",
    actual = ":v8_libneuron_adapter_so",
)

alias(
    name = "libc++.so.1",
    actual = ":v8_libcxx_so",
)

alias(
    name = "neuron_host_libs",
    actual = ":v8_host_libs",
)

exports_files(srcs = glob(["**/*.so"]))
