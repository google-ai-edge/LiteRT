load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

td_library(
    name = "litert_td_files",
    srcs = [
        "attributes.td",
        "canonicalize.td",
        "dialect.td",
        "ops.td",
        "type_constraints.td",
        "types.td",
    ],
    deps = [
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "attributes_inc_gen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "attributes.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "attributes.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "attributes.td",
    deps = [":litert_td_files"],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "dialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect.td",
    deps = [":litert_td_files"],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = {
        "ops.h.inc": ["-gen-op-decls"],
        "ops.cc.inc": ["-gen-op-defs"],
        "ops_interface.h.inc": ["-gen-op-interface-decls"],
        "ops_interface.cc.inc": ["-gen-op-interface-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ops.td",
    deps = [":litert_td_files"],
)

gentbl_cc_library(
    name = "types_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
                "-typedefs-dialect=litert",
            ],
            "types.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "-typedefs-dialect=litert",
            ],
            "types.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "types.td",
    deps = [":litert_td_files"],
)

gentbl_cc_library(
    name = "canonicalize_inc_gen",
    tbl_outs = {"canonicalize.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "canonicalize.td",
    deps = [":litert_td_files"],
)

cc_library(
    name = "dialect",
    srcs = [
        "attributes.cc",
        "dialect.cc",
        "ops.cc",
        "traits.cc",
        "types.cc",
    ],
    hdrs = [
        "attributes.h",
        "dialect.h",
        "ops.h",
        "traits.h",
        "types.h",
    ],
    deps = [
        ":attributes_inc_gen",
        ":callback_resource",
        ":canonicalize_inc_gen",
        ":dialect_inc_gen",
        ":lazy_blob_manager",
        ":lazy_resource_blob",
        ":ops_inc_gen",
        ":tensor_parser",
        ":tensor_printer",
        ":types_inc_gen",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "types_test",
    srcs = ["types_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":dialect",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "lazy_dense_elements_attr_test",
    srcs = ["lazy_dense_elements_attr_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":dialect",
        "//litert/compiler/mlir/dialects/litert:lazy_resource_blob",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tensor_parser",
    hdrs = ["tensor_parser.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "tensor_printer",
    hdrs = ["tensor_printer.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_test(
    name = "tensor_printer_test",
    srcs = ["tensor_printer_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":dialect",
        ":tensor_printer",
        "//perftools/profiles/collector/heap:alloc_recorder",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "tensor_printer_benchmark",
    srcs = ["tensor_printer_benchmark.cc"],
    tags = ["no_oss"],
    deps = [
        ":tensor_printer",
        "//perftools/profiles/collector/heap:alloc_recorder",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "traits",
    srcs = ["traits.cc"],
    hdrs = ["traits.h"],
    deps = [
        ":dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "lazy_resource_blob",
    srcs = ["lazy_resource_blob.cc"],
    hdrs = ["lazy_resource_blob.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:fingerprint",
    ],
)

cc_test(
    name = "lazy_resource_blob_test",
    srcs = ["lazy_resource_blob_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":lazy_resource_blob",
        "//perftools/profiles/collector/heap:alloc_recorder",
        "//third_party/tcmalloc:malloc_extension",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "lazy_blob_manager",
    srcs = ["lazy_blob_manager.cc"],
    hdrs = ["lazy_blob_manager.h"],
    # TODO(aarfaian): figure out what's going on with thread-safety-analysis complaining about llvm::sys::SmartRWMutex
    copts = ["-Wno-thread-safety-analysis"],
    deps = [
        ":lazy_resource_blob",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/log",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "callback_resource",
    srcs = ["callback_resource.cc"],
    hdrs = ["callback_resource.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
