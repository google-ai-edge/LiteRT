# Simplified third-party dependencies for LiteRT (essential only)

include(FetchContent)

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

# Make dependencies available
FetchContent_MakeAvailable(absl)
FetchContent_MakeAvailable(flatbuffers)

if(LITERT_BUILD_TESTS)
    FetchContent_MakeAvailable(googletest)
endif()

# Platform-specific dependencies
if(ANDROID)
    # Android-specific libraries
    find_library(ANDROID_LOG_LIB log)
    find_library(ANDROID_LIB android)
endif()

if(LITERT_ENABLE_GPU)
    # OpenCL for GPU support (optional)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        set(LITERT_HAS_OPENCL ON)
    endif()
endif()

# Export variables for use in other CMakeLists.txt files
set(LITERT_THIRD_PARTY_INCLUDES
    ${absl_SOURCE_DIR}
    ${flatbuffers_SOURCE_DIR}/include
    CACHE INTERNAL ""
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

# Set variables in cache for global access
set(LITERT_THIRD_PARTY_LIBS ${LITERT_THIRD_PARTY_LIBS} CACHE INTERNAL "")