# Copyright 2026 Google LLC.
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

project(litert_source_build_cc_api_example LANGUAGES CXX C)

# Some TensorFlow Lite third-party CMake projects still declare compatibility
# below CMake 3.5. CMake 4 rejects those projects unless this floor is set.
if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING
        "Minimum CMake policy compatibility for vendored dependencies")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LITERT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../..")
set(LITERT_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/litert_build")
set(TFLITE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../tflite")
set(LITERT_EXAMPLE_DEFAULT_MODEL
    "${LITERT_SOURCE_DIR}/test/testdata/simple_add_dynamic_shape.tflite")
set(LITERT_EXAMPLE_DEFAULT_GPU_MODEL
    "${LITERT_SOURCE_DIR}/test/testdata/mobilenet_v2_1.0_224.tflite")

add_subdirectory("${LITERT_SOURCE_DIR}" "${LITERT_BUILD_DIR}" EXCLUDE_FROM_ALL)

add_executable(litert_source_build_cc_api_run_model
    source_build_cc_api_run_model.cc
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(litert_source_build_cc_api_run_model
        PRIVATE -Wno-deprecated-declarations)
endif()

target_compile_definitions(litert_source_build_cc_api_run_model
    PRIVATE
        LITERT_EXAMPLE_DEFAULT_MODEL="${LITERT_EXAMPLE_DEFAULT_MODEL}"
        LITERT_EXAMPLE_DEFAULT_GPU_MODEL="${LITERT_EXAMPLE_DEFAULT_GPU_MODEL}"
        LITERT_EXAMPLE_DEFAULT_RUNTIME_LIBRARY_DIR="$<TARGET_FILE_DIR:litert_runtime_c_api_shared_lib>"
)

target_link_libraries(litert_source_build_cc_api_run_model
    PRIVATE
        litert_cc_api_with_dynamic_runtime
)

set_target_properties(litert_source_build_cc_api_run_model PROPERTIES
    BUILD_RPATH "$<TARGET_FILE_DIR:litert_runtime_c_api_shared_lib>"
)

if(LITERT_ENABLE_GPU)
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _litert_example_processor)
    set(_litert_gpu_accelerator_relpath "")

    if(APPLE AND _litert_example_processor MATCHES "^(arm64|aarch64)$")
        set(_litert_gpu_accelerator_relpath
            "macos_arm64/libLiteRtMetalAccelerator.dylib")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND _litert_example_processor MATCHES "^(x86_64|amd64)$")
        set(_litert_gpu_accelerator_relpath
            "linux_x86_64/libLiteRtWebGpuAccelerator.so")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND _litert_example_processor MATCHES "^(aarch64|arm64)$")
        set(_litert_gpu_accelerator_relpath
            "linux_arm64/libLiteRtWebGpuAccelerator.so")
    elseif(WIN32)
        set(_litert_gpu_accelerator_relpath
            "windows_x86_64/libLiteRtWebGpuAccelerator.dll")
    elseif(ANDROID AND _litert_example_processor MATCHES "^(aarch64|arm64)$")
        set(_litert_gpu_accelerator_relpath
            "android_arm64/libLiteRtClGlAccelerator.so")
    endif()

    if(_litert_gpu_accelerator_relpath)
        set(_litert_gpu_accelerator
            "${CMAKE_CURRENT_BINARY_DIR}/litert_prebuilts/${_litert_gpu_accelerator_relpath}")
        set(_litert_gpu_accelerator_url
            "https://storage.googleapis.com/litert/binaries/latest/${_litert_gpu_accelerator_relpath}")
        get_filename_component(_litert_gpu_accelerator_name
            "${_litert_gpu_accelerator}" NAME)
        get_filename_component(_litert_gpu_accelerator_dir
            "${_litert_gpu_accelerator}" DIRECTORY)

        file(MAKE_DIRECTORY "${_litert_gpu_accelerator_dir}")
        message(STATUS
            "Downloading LiteRT GPU accelerator from ${_litert_gpu_accelerator_url}")
        file(DOWNLOAD
            "${_litert_gpu_accelerator_url}"
            "${_litert_gpu_accelerator}"
            STATUS _litert_download_status
            TLS_VERIFY ON
            SHOW_PROGRESS)
        list(GET _litert_download_status 0 _litert_download_code)
        list(GET _litert_download_status 1 _litert_download_message)
        if(NOT _litert_download_code EQUAL 0)
            message(FATAL_ERROR
                "Failed to download LiteRT GPU accelerator: "
                "${_litert_download_message}")
        endif()

        add_custom_command(TARGET litert_source_build_cc_api_run_model POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_litert_gpu_accelerator}"
                "$<TARGET_FILE_DIR:litert_runtime_c_api_shared_lib>/${_litert_gpu_accelerator_name}"
            COMMENT "Copying LiteRT GPU accelerator next to libLiteRt")
    else()
        message(WARNING
            "LITERT_ENABLE_GPU=ON, but this example does not know which "
            "prebuilt GPU accelerator to download for this platform.")
    endif()
endif()
