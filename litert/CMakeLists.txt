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

cmake_minimum_required(VERSION 3.20)
project(LiteRT VERSION 1.4.0 LANGUAGES CXX C)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(LITERT_BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(LITERT_BUILD_TOOLS "Build LiteRT tools" ON)
option(LITERT_BUILD_TESTS "Build LiteRT tests" OFF)
option(LITERT_USE_GPU "Enable GPU support" ON)
option(LITERT_USE_NPU "Enable NPU support" ON)

# Find TFLite package - we depend on it being built first
set(TFLITE_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/../tflite_build" CACHE PATH "Path to TFLite build directory")
set(TFLITE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tflite" CACHE PATH "Path to TFLite source directory")

# Get TensorFlow source directory from TFLite build
if(EXISTS "${TFLITE_BUILD_DIR}")
    include("${TFLITE_BUILD_DIR}/tensorflow_src_dir.cmake" OPTIONAL)
endif()

# Set default TensorFlow source directory if not found
if(NOT TENSORFLOW_SOURCE_DIR)
    if(EXISTS "${TFLITE_BUILD_DIR}/tensorflow-src")
        set(TENSORFLOW_SOURCE_DIR "${TFLITE_BUILD_DIR}/tensorflow-src")
    else()
        message(FATAL_ERROR "TensorFlow source directory not found. Please build TFLite first or set
ENSORFLOW_SOURCE_DIR")
    endif()
endif()

message(STATUS "Using TensorFlow source directory: ${TENSORFLOW_SOURCE_DIR}")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${TFLITE_SOURCE_DIR}
    ${TENSORFLOW_SOURCE_DIR}
)

# Platform detection
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(LITERT_PLATFORM_MACOS ON)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(LITERT_PLATFORM_LINUX ON)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(LITERT_PLATFORM_WINDOWS ON)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(LITERT_PLATFORM_ANDROID ON)
endif()

# Find required packages
find_package(Threads REQUIRED)

# Add subdirectories
add_subdirectory(c)
add_subdirectory(cc)
add_subdirectory(core)
add_subdirectory(runtime)
add_subdirectory(compiler_plugin)
