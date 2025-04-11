#!/bin/bash

print_usage() {
    echo "Usage: $0 <LiteRT Root>"
    echo "  example:"
    echo "    $0 \$LiteRT/"
}

if [ "$#" -eq 1 ]; then
    if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
        print_usage
        exit 0
    fi
else 
    print_usage
    exit 1
fi

litert_root="$1"
pushd "${litert_root}"

BUILD_TARGETS=(
    "//litert/vendors/qualcomm/core/backends:all"
    "//litert/vendors/qualcomm/core/builders:all"
    "//litert/vendors/qualcomm/core/schema:all"
    "//litert/vendors/qualcomm/core/utils:all"
    "//litert/vendors/qualcomm/core/wrappers:all"
    "//litert/vendors/qualcomm/core:all"
    "//litert/vendors/qualcomm/compiler/IR:all"
    "//litert/vendors/qualcomm/compiler/legalizations:all"
    "//litert/vendors/qualcomm/compiler:all"
    "//litert/vendors/qualcomm/dispatch:all"
    "//litert/vendors/qualcomm/tools:all"
    "//litert/vendors/qualcomm:all"
)

build_result=""
for TARGET in "${BUILD_TARGETS[@]}"; do
    echo "Building $TARGET"
    bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility "$TARGET" # --copt=-Werror=unused-variable

    res=$?
    if [ $res -ne 0 ]; then
        build_result="$build_result\nFAIL    $TARGET"
    else
        build_result="$build_result\nSUCCESS $TARGET"
    fi
done

TEST_TARGETS=(
    "//litert/vendors/qualcomm/core/utils:utils_test"
    "//litert/vendors/qualcomm/core/wrappers/tests:op_wrapper_test"
    "//litert/vendors/qualcomm/core/wrappers/tests:tensor_wrapper_test"
    "//litert/vendors/qualcomm/core/wrappers/tests:param_wrapper_test"
    "//litert/vendors/qualcomm/core/wrappers/tests:quantize_params_wrapper_test"
    "//litert/vendors/qualcomm/core:tensor_pool_test"
    "//litert/vendors/qualcomm/compiler:qnn_compiler_plugin_test"
    "//litert/vendors/qualcomm/dispatch:_dispatch_api_qualcomm_test"
    "//litert/vendors/qualcomm:qnn_manager_test"
)

qairt_folder="${litert_root}/third_party/qairt/latest"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$qairt_folder/lib/x86_64-linux-clang/

test_result=""
for TARGET in "${TEST_TARGETS[@]}"; do
    echo "Building $TARGET"
    bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility "$TARGET" # --copt=-Werror=unused-variable

    build_res=$?
    if [ $build_res -ne 0 ]; then
        test_result="$test_result\nFAIL    $TARGET"
    else
        binary_path=bazel-bin/$(echo "$TARGET" | sed 's|//||; s|:|/|')
        echo "Running $binary_path"
        "$binary_path"
        run_res=$?
        if [ $run_res -ne 0 ]; then
            test_result="$test_result\nFAIL    $TARGET"
        else
            test_result="$test_result\nSUCCESS $TARGET"
        fi
    fi
done

popd "${litert_root}"

# summary result
printf "==== Build Summary ====\n"
printf "If there is any build failed, please check log before and fix it.\n"
printf "$build_result\n"
printf "\n"


printf "==== Test Summary ====\n"
printf "If there is any test failed, please check log before and fix it.\n"
printf "$test_result\n"
printf "\n"
