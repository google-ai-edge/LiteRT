cc_test(
    name = "add_relu_test",
    srcs = [
        "add_relu_test.cc",
    ],
    deps = [
        "//litert/cc:litert_buffer_ref",
        "//litert/c:litert_model",
        "//litert/test:matchers",
        "//litert/test/generators:graph_helpers",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core:tensor_pool",
        "//litert/vendors/qualcomm/core/utils:qnn_model",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/qnn_backend_test:qnn_backend_creator",
        "//litert/vendors/qualcomm/qnn_backend_test:utils",
        "@com_google_googletest//:gtest_main",
    ],
)
