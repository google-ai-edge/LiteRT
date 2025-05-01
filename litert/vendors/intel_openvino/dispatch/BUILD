load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

litert_dynamic_lib(
    name = "dispatch_api",
    srcs = [
        "dispatch_api.cc",
    ],
    hdrs = [
    ],
    export_litert_only = True,
    linkopts = select({
        "//conditions:default": [],
    }),
    linkstatic = 1,
    shared_lib_name = "dispatch_api_so",
    so_name = "libLiteRtDispatch_IntelOpenvino.so",
    tags = [
        "nobuilder",
    ],
    visibility = ["//litert:__subpackages__"],
    deps = [

        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/c:litert_model",
        "//litert/c:litert_tensor_buffer",
        "//litert/cc:litert_expected",
        "//litert/core/util:tensor_type_util",
        "//litert/vendors/c:litert_dispatch_c_api",
    ],
)
