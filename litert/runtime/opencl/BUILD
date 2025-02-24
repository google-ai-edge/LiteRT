package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

cc_library(
    name = "cl_command_queue",
    srcs = [
        "cl_command_queue.cc",
    ],
    hdrs = [
        "cl_command_queue.h",
    ],
    deps = [
        ":cl_context",
        ":cl_device",
        ":opencl_wrapper",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/opencl_headers",
    ],
)

cc_library(
    name = "cl_device",
    srcs = [
        "cl_device.cc",
    ],
    hdrs = [
        "cl_device.h",
    ],
    deps = [
        ":opencl_wrapper",
        "//third_party/absl/status",
        "//third_party/absl/strings:str_format",
        "//third_party/opencl_headers",
    ],
)

cc_library(
    name = "cl_context",
    srcs = [
        "cl_context.cc",
    ],
    hdrs = [
        "cl_context.h",
    ],
    deps = [
        ":cl_device",
        ":opencl_wrapper",
        "//third_party/absl/status",
        "//third_party/absl/strings",
        "//third_party/opencl_headers",
    ],
)

cc_library(
    name = "opencl_wrapper",
    srcs = [
        "opencl_wrapper.cc",
    ],
    hdrs = [
        "opencl_wrapper.h",
    ],
    visibility = [
        "//third_party/odml/infra:__subpackages__",
        "//litert:__subpackages__",
    ],
    deps = [
        "//third_party/absl/status",
        "//third_party/absl/strings",
        "//third_party/opencl_headers",
    ],
)

cc_library(
    name = "buffer",
    srcs = [
        "buffer.cc",
    ],
    hdrs = [
        "buffer.h",
    ],
    deps = [
        ":cl_command_queue",
        ":cl_context",
        ":opencl_wrapper",
        "//third_party/absl/status",
        "//third_party/absl/strings",
        "//third_party/absl/types:span",
        "//third_party/opencl_headers",
    ],
)

cc_test(
    name = "buffer_test",
    srcs = ["buffer_test.cc"],
    # require GPU to run OpenCL tests.
    tags = [
        "requires-gpu-nvidia",
    ],
    deps = [
        ":buffer",
        ":cl_command_queue",
        ":cl_context",
        ":cl_device",
        ":opencl_wrapper",
        "//testing/base/public:gunit_main",
        "//third_party/absl/types:span",
    ],
)
