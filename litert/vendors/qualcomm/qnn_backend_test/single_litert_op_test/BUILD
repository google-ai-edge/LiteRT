cc_test(
    name = "add_relu_test",
    srcs = [
        "add_relu_test.cc",
    ],
    deps = [
        "//litert/vendors/qualcomm/qnn_backend_test:qnn_backend_creator",
        "//litert/core/model:model_graph_testonly",
        "//litert/core/model:model_testonly",
        "//litert/core/util:flatbuffer_tools_testonly",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core:tensor_pool",
        "//litert/vendors/qualcomm/core/utils:qnn_model",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/qnn_backend_test:utils",
        "@com_google_googletest//:gtest_main",
    ],
)
