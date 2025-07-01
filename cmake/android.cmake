# Android build configuration for LiteRT
# This file configures CMake for Android cross-compilation

# Set Android-specific variables if not already set
if(NOT ANDROID_NDK)
    if(DEFINED ENV{ANDROID_NDK_ROOT})
        set(ANDROID_NDK $ENV{ANDROID_NDK_ROOT})
    elseif(DEFINED ENV{ANDROID_NDK})
        set(ANDROID_NDK $ENV{ANDROID_NDK})
    else()
        message(FATAL_ERROR "ANDROID_NDK not found. Please set ANDROID_NDK_ROOT or ANDROID_NDK environment variable")
    endif()
endif()

# Default Android settings to match Bazel's android_arm64 config
if(NOT ANDROID_ABI)
    set(ANDROID_ABI "arm64-v8a")
endif()

if(NOT ANDROID_PLATFORM)
    set(ANDROID_PLATFORM "android-21")  # Minimum API level for arm64-v8a
endif()

if(NOT ANDROID_STL)
    set(ANDROID_STL "c++_static")
endif()

# Set Android toolchain
set(CMAKE_TOOLCHAIN_FILE ${ANDROID_NDK}/build/cmake/android.toolchain.cmake)

# Android-specific compiler flags (matching .bazelrc)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")  # Suppress warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -std=c++17")  # Suppress warnings and set C++17

# Android-specific linker flags
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-libstdc++")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++")

# Force static linking for Android (matching Bazel's android config)
set(BUILD_SHARED_LIBS OFF)

# Android-specific definitions
add_definitions(-DANDROID)

# Set CPU optimization flags for ARM64
if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
endif()

# Print Android configuration
message(STATUS "")
message(STATUS "Android Configuration:")
message(STATUS "  NDK: ${ANDROID_NDK}")
message(STATUS "  ABI: ${ANDROID_ABI}")
message(STATUS "  Platform: ${ANDROID_PLATFORM}")
message(STATUS "  STL: ${ANDROID_STL}")
message(STATUS "  Toolchain: ${CMAKE_TOOLCHAIN_FILE}")
message(STATUS "")