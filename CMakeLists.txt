cmake_minimum_required(VERSION 3.20)

# Set the project name and version
project(LiteRT 
    VERSION 1.0.0
    DESCRIPTION "LiteRT: A High-Performance ML Runtime System"
    LANGUAGES C CXX)

# Set build type to Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build options
option(LITERT_ENABLE_QUALCOMM "Enable Qualcomm NPU support" OFF)
option(LITERT_ENABLE_MEDIATEK "Enable MediaTek NPU support" OFF)
option(LITERT_ENABLE_GOOGLE_TENSOR "Enable Google Tensor support" OFF)
option(LITERT_ENABLE_GPU "Enable GPU acceleration" ON)
option(LITERT_BUILD_TOOLS "Build LiteRT tools" ON)
option(LITERT_BUILD_TESTS "Build LiteRT tests" OFF)

# Platform detection
if(ANDROID)
    set(LITERT_PLATFORM_ANDROID ON)
elseif(WIN32)
    set(LITERT_PLATFORM_WINDOWS ON)
elseif(APPLE)
    set(LITERT_PLATFORM_APPLE ON)
else()
    set(LITERT_PLATFORM_LINUX ON)
endif()

# Compiler flags
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Common compiler flags
if(MSVC)
    add_compile_options(/W4 /std:c++17)
    add_compile_definitions(WIN32_LEAN_AND_MEAN NOGDI _USE_MATH_DEFINES)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    # Enable experimental C11 atomics
    add_compile_options(/experimental:c11atomics)
else()
    add_compile_options(-Wall -Wextra)
    if(ANDROID)
        add_compile_options(-w)  # Suppress warnings on Android as per .bazelrc
    endif()
    add_compile_options(-fPIC)
endif()

# Android-specific configuration
if(ANDROID)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-libstdc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++")
    add_compile_definitions(ANDROID)
endif()

# Global compile definitions
add_compile_definitions(
    FLATBUFFERS_LOCALE_INDEPENDENT=0
    GRPC_BAZEL_BUILD
)

# Feature-specific compile definitions
if(LITERT_ENABLE_GPU)
    add_compile_definitions(LITERT_ENABLE_GPU=1)
else()
    add_compile_definitions(LITERT_ENABLE_GPU=0)
endif()

if(LITERT_ENABLE_QUALCOMM)
    add_compile_definitions(LITERT_ENABLE_QUALCOMM=1)
endif()

if(LITERT_ENABLE_MEDIATEK)
    add_compile_definitions(LITERT_ENABLE_MEDIATEK=1)
endif()

if(LITERT_ENABLE_GOOGLE_TENSOR)
    add_compile_definitions(LITERT_ENABLE_GOOGLE_TENSOR=1)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# Find required packages
find_package(Threads REQUIRED)

# Set up third-party dependencies
include(cmake/third_party_simple.cmake)

# Add subdirectories in dependency order
add_subdirectory(litert/c)
add_subdirectory(litert/cc)
add_subdirectory(litert/core)
add_subdirectory(litert/runtime)

# Add vendor support if enabled
if(LITERT_ENABLE_QUALCOMM)
    add_subdirectory(litert/vendors/qualcomm)
endif()

if(LITERT_ENABLE_MEDIATEK)
    add_subdirectory(litert/vendors/mediatek)
endif()

if(LITERT_ENABLE_GOOGLE_TENSOR)
    add_subdirectory(litert/vendors/google_tensor)
endif()

# Add tools if enabled
if(LITERT_BUILD_TOOLS)
    add_subdirectory(litert/tools)
endif()

# Add tests if enabled
if(LITERT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(litert/test)
endif()

# Installation
install(DIRECTORY litert/c/ 
    DESTINATION include/litert/c
    FILES_MATCHING PATTERN "*.h")

install(DIRECTORY litert/cc/
    DESTINATION include/litert/cc
    FILES_MATCHING PATTERN "*.h")

# Print configuration summary
message(STATUS "")
message(STATUS "LiteRT Configuration Summary:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Target Platform: ${CMAKE_SYSTEM_NAME}")
if(ANDROID)
    message(STATUS "  Android ABI: ${ANDROID_ABI}")
    message(STATUS "  Android Platform: ${ANDROID_PLATFORM}")
endif()
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Tools: ${LITERT_BUILD_TOOLS}")
message(STATUS "  Build Tests: ${LITERT_BUILD_TESTS}")
message(STATUS "  Qualcomm Support: ${LITERT_ENABLE_QUALCOMM}")
message(STATUS "  MediaTek Support: ${LITERT_ENABLE_MEDIATEK}")
message(STATUS "  Google Tensor Support: ${LITERT_ENABLE_GOOGLE_TENSOR}")
message(STATUS "  GPU Support: ${LITERT_ENABLE_GPU}")
message(STATUS "")