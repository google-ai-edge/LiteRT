# Third-party dependencies for LiteRT

include(FetchContent)
include(ExternalProject)

# Abseil
FetchContent_Declare(
    absl
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG        20240722.0
)

# FlatBuffers
FetchContent_Declare(
    flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG        v24.3.25
)

# GoogleTest (for testing)
if(LITERT_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
endif()

# XNNPACK (for CPU acceleration)
FetchContent_Declare(
    xnnpack
    GIT_REPOSITORY https://github.com/google/XNNPACK.git
    GIT_TAG        2024-11-20
)

# pthreadpool (dependency of XNNPACK)
FetchContent_Declare(
    pthreadpool
    GIT_REPOSITORY https://github.com/Maratyszcza/pthreadpool.git
    GIT_TAG        master
)

# cpuinfo (dependency of XNNPACK)
FetchContent_Declare(
    cpuinfo
    GIT_REPOSITORY https://github.com/pytorch/cpuinfo.git
    GIT_TAG        main
)

# FP16 (dependency of XNNPACK)
FetchContent_Declare(
    fp16
    GIT_REPOSITORY https://github.com/Maratyszcza/fp16.git
    GIT_TAG        master
)

# FXdiv (dependency of XNNPACK)
FetchContent_Declare(
    fxdiv
    GIT_REPOSITORY https://github.com/Maratyszcza/fxdiv.git
    GIT_TAG        master
)

# Configure Abseil
set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_ENABLE_INSTALL ON)

# Configure FlatBuffers
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL ON CACHE BOOL "" FORCE)

# Configure GoogleTest
if(LITERT_BUILD_TESTS)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
endif()

# Configure XNNPACK and its dependencies
set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(XNNPACK_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(PTHREADPOOL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PTHREADPOOL_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(CPUINFO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(FP16_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FP16_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)

# Make dependencies available
FetchContent_MakeAvailable(absl)
FetchContent_MakeAvailable(flatbuffers)

if(LITERT_BUILD_TESTS)
    FetchContent_MakeAvailable(googletest)
endif()

# Configure XNNPACK dependencies
FetchContent_MakeAvailable(pthreadpool)
FetchContent_MakeAvailable(cpuinfo)
FetchContent_MakeAvailable(fp16)
FetchContent_MakeAvailable(fxdiv)
FetchContent_MakeAvailable(xnnpack)

# Create alias targets for consistency
if(NOT TARGET absl::base)
    add_library(absl::base ALIAS absl_base)
endif()

if(NOT TARGET absl::strings)
    add_library(absl::strings ALIAS absl_strings)
endif()

if(NOT TARGET absl::flags)
    add_library(absl::flags ALIAS absl_flags)
endif()

if(NOT TARGET absl::flags_parse)
    add_library(absl::flags_parse ALIAS absl_flags_parse)
endif()

if(NOT TARGET absl::log)
    add_library(absl::log ALIAS absl_log)
endif()

if(NOT TARGET absl::status)
    add_library(absl::status ALIAS absl_status)
endif()

if(NOT TARGET absl::statusor)
    add_library(absl::statusor ALIAS absl_statusor)
endif()

# Platform-specific dependencies
if(ANDROID)
    # Android-specific libraries
    find_library(ANDROID_LOG_LIB log)
    find_library(ANDROID_LIB android)
endif()

if(LITERT_ENABLE_GPU)
    # OpenCL for GPU support
    find_package(OpenCL)
    if(OpenCL_FOUND)
        set(LITERT_HAS_OPENCL ON)
    endif()
endif()

# Export variables for use in other CMakeLists.txt files
set(LITERT_THIRD_PARTY_INCLUDES
    ${absl_SOURCE_DIR}
    ${flatbuffers_SOURCE_DIR}/include
    ${xnnpack_SOURCE_DIR}/include
    PARENT_SCOPE
)

set(LITERT_THIRD_PARTY_LIBS
    absl::strings
    absl::flags
    absl::flags_parse
    absl::log
    absl::status
    absl::statusor
    absl::str_format
    absl::span
    absl::cleanup
    absl::flat_hash_map
    absl::flat_hash_set
    flatbuffers
    XNNPACK
    PARENT_SCOPE
)

if(ANDROID)
    list(APPEND LITERT_THIRD_PARTY_LIBS ${ANDROID_LOG_LIB} ${ANDROID_LIB})
endif()

if(LITERT_HAS_OPENCL)
    list(APPEND LITERT_THIRD_PARTY_LIBS OpenCL::OpenCL)
endif()

if(NOT WIN32)
    list(APPEND LITERT_THIRD_PARTY_LIBS dl)
endif()

# Set variables in parent scope
set(LITERT_THIRD_PARTY_INCLUDES ${LITERT_THIRD_PARTY_INCLUDES} PARENT_SCOPE)
set(LITERT_THIRD_PARTY_LIBS ${LITERT_THIRD_PARTY_LIBS} PARENT_SCOPE)