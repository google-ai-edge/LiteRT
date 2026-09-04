load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "hooks",
    srcs = [
        "hooks_utils.cc",
        "power_trace_hook.cc",
        "tpu_tile_hook.cc",
    ],
    hdrs = [
        "hooks_utils.h",
        "power_trace_hook.h",
        "tpu_tile_hook.h",
    ],
    deps = [
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_metrics_header",
        "//litert/c/internal:litert_logging",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//litert/vendors/google_tensor/dispatch:dispatch_api_headers",
        "//platforms/darwinn/devtools/power_stats",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)
