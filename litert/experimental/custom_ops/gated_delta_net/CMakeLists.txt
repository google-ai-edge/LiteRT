# Copyright 2026 The LiteRT Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set(GATED_DELTA_NET_SRC
  "gated_delta_update_impl.cc"
  "gated_delta_update_litert_custom_op.cc"
  "gated_delta_update_tflite_op.cc"
)

add_library(litert_gated_delta_net STATIC
  ${GATED_DELTA_NET_SRC}
)
target_link_libraries(litert_gated_delta_net
  PUBLIC
    flatbuffers::flatbuffers
    absl::log
    Eigen3::Eigen
)
target_include_directories(litert_gated_delta_net
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}>
        $<BUILD_INTERFACE:${TFLITE_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/opencl_headers>
        ${Python3_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)