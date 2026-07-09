cmake_minimum_required(VERSION 3.20)

project(litert_fuzzers LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Some vendored TFLite dependencies still declare very old CMake policy
# versions. Keep the standalone fuzzer project buildable with CMake 4.x.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
# This project is not installable. Disabling dependency install exports also
# avoids export-set errors from the vendored RE2/GoogleTest CMake projects.
set(CMAKE_SKIP_INSTALL_RULES ON)

set(LITERT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.."
    CACHE PATH "LiteRT source directory")
set(FUZZTEST_SOURCE_DIR "/data/bt/os/fuzztest"
    CACHE PATH "FuzzTest source directory")

option(LITERT_FUZZ_SANITIZERS
       "Enable AddressSanitizer and UndefinedBehaviorSanitizer"
       ON)
option(LITERT_FUZZ_ENABLE_XNNPACK
       "Build XNNPACK delegate fuzz targets"
       ON)

set(FUZZTEST_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(FUZZTEST_BUILD_FLATBUFFERS OFF CACHE BOOL "" FORCE)
set(FUZZTEST_FUZZING_MODE ON CACHE BOOL "" FORCE)

if(LITERT_FUZZ_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# FuzzTest's CMake builds the framework first, then applies sanitizer-coverage
# flags to the target project. Without these flags the FUZZ_TESTs still run with
# random inputs, but corpus growth is not coverage-guided through TFLite code.
include("${FUZZTEST_SOURCE_DIR}/cmake/FuzzTestFlagSetup.cmake")
fuzztest_setup_fuzzing_flags()

# TFLite's CMake project supplies the tensorflow-lite target used by the
# fuzzer. Keep its optional test suite disabled; this project builds only the
# selected fuzz target.
set(TFLITE_KERNEL_TEST OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_XNNPACK ${LITERT_FUZZ_ENABLE_XNNPACK} CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_GPU OFF CACHE BOOL "" FORCE)
add_subdirectory(
  "${LITERT_SOURCE_DIR}/tflite"
  "${CMAKE_CURRENT_BINARY_DIR}/tflite"
  EXCLUDE_FROM_ALL)

add_subdirectory(
  "${FUZZTEST_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/fuzztest"
  EXCLUDE_FROM_ALL)

set(FUZZ_SUPPORT_SOURCES
  "${LITERT_SOURCE_DIR}/tflite/kernels/fuzzing/fuzzer_quota_allocator.cc"
  "${LITERT_SOURCE_DIR}/tflite/kernels/fuzzing/one_op_fuzz_model.cc")

set(PAD_FUZZ_SOURCES
  ${FUZZ_SUPPORT_SOURCES}
  "${LITERT_SOURCE_DIR}/tflite/kernels/fuzzing/pad_fuzz_test.cc")

function(add_tflite_fuzzer target_name)
  add_executable(${target_name} ${ARGN})
  target_link_libraries(${target_name} PRIVATE
    fuzztest::fuzztest
    fuzztest::fuzztest_gtest_main)
endfunction()

add_tflite_fuzzer(tflite_pad_fuzz_test ${PAD_FUZZ_SOURCES})
target_link_libraries(tflite_pad_fuzz_test PRIVATE
  tensorflow-lite)

add_tflite_fuzzer(tflite_mirror_pad_fuzz_test
  ${FUZZ_SUPPORT_SOURCES}
  "${LITERT_SOURCE_DIR}/tflite/kernels/fuzzing/mirror_pad_fuzz_test.cc")
target_link_libraries(tflite_mirror_pad_fuzz_test PRIVATE
  tensorflow-lite)

add_tflite_fuzzer(tflite_stablehlo_pad_fuzz_test
  ${FUZZ_SUPPORT_SOURCES}
  "${LITERT_SOURCE_DIR}/tflite/kernels/fuzzing/stablehlo_pad_fuzz_test.cc")
target_link_libraries(tflite_stablehlo_pad_fuzz_test PRIVATE
  tensorflow-lite)

add_tflite_fuzzer(tflite_convolution_fuzz_test
  ${FUZZ_SUPPORT_SOURCES}
  "${LITERT_SOURCE_DIR}/tflite/kernels/fuzzing/convolution_fuzz_test.cc")
target_link_libraries(tflite_convolution_fuzz_test PRIVATE
  tensorflow-lite)

if(LITERT_FUZZ_ENABLE_XNNPACK)
  add_tflite_fuzzer(tflite_pad_xnnpack_fuzz_test ${PAD_FUZZ_SOURCES})
  target_compile_definitions(tflite_pad_xnnpack_fuzz_test PRIVATE
    TFLITE_PAD_FUZZ_ENABLE_XNNPACK)
  # The fuzzer calls the delegate C API directly. Put the delegate archive
  # before tensorflow-lite so its references back into TFLite resolve with
  # linkers that do a single left-to-right scan of static libraries.
  target_link_libraries(tflite_pad_xnnpack_fuzz_test PRIVATE
    xnnpack-delegate
    tensorflow-lite)
endif()

enable_testing()
add_test(
  NAME tflite_pad_fuzz_test
  COMMAND tflite_pad_fuzz_test
          --fuzz=PadFuzzTest.PadNeverCrashes)
add_test(
  NAME tflite_pad_ref_fuzz_test
  COMMAND tflite_pad_fuzz_test
          --fuzz=PadFuzzTest.PadReferenceNeverCrashes)
add_test(
  NAME tflite_mirror_pad_fuzz_test
  COMMAND tflite_mirror_pad_fuzz_test
          --fuzz=MirrorPadFuzzTest.MirrorPadNeverCrashes)
add_test(
  NAME tflite_stablehlo_pad_fuzz_test
  COMMAND tflite_stablehlo_pad_fuzz_test
          --fuzz=StablehloPadFuzzTest.StablehloPadNeverCrashes)
add_test(
  NAME tflite_convolution_fuzz_test
  COMMAND tflite_convolution_fuzz_test
          --fuzz=ConvolutionFuzzTest.Conv2DIsSafeAndContractCompliant)
if(LITERT_FUZZ_ENABLE_XNNPACK)
  add_test(
    NAME tflite_pad_xnnpack_fuzz_test
    COMMAND tflite_pad_xnnpack_fuzz_test
            --fuzz=PadFuzzTest.PadXnnpackNeverCrashes)
endif()
