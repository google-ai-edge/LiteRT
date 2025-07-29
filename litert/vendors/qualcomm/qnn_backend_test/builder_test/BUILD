cc_test(
    name = "relu_test",
    srcs = [
        "relu_test.cc",
    ],
    deps = [
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core:tensor_pool",
        "//litert/vendors/qualcomm/core/builders:relu_op_builder",
        "//litert/vendors/qualcomm/core/utils:qnn_model",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/qnn_backend_test:qnn_backend_creator",
        "@com_google_googletest//:gtest_main",
    ],
)
