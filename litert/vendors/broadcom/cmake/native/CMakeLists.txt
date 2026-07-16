cmake_minimum_required(VERSION 3.19)
project(native-tools C CXX)

set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "")

get_filename_component(TFLITE_SOURCE_DIR
  "${CMAKE_CURRENT_LIST_DIR}/../../../LiteRT/tflite" ABSOLUTE)
list(APPEND CMAKE_MODULE_PATH "${TFLITE_SOURCE_DIR}/tools/cmake/modules")
list(APPEND CMAKE_PREFIX_PATH "${TFLITE_SOURCE_DIR}/tools/cmake/modules")

add_subdirectory(${TFLITE_SOURCE_DIR}/tools/cmake/native_tools/flatbuffers
                 ${CMAKE_BINARY_DIR}/flatc)
