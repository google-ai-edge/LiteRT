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

# Measurement of the memory a process uses, split into host (CPU) and
# accelerator (GPU / NPU) memory. One static library carries the four Bazel
# targets of BUILD (host_memory, accelerator_memory, memory_snapshot,
# memory_monitor); it depends on absl and operating-system APIs only.

cmake_minimum_required(VERSION 3.20)

# ==============================================================================
# Host memory: exactly one host_memory_<os>.cc implements the platform
# contract for the target OS. Keep this in sync with the select() in BUILD.
# ==============================================================================
if(ANDROID OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(LITERT_HOST_MEMORY_SOURCES host_memory_linux.cc host_memory_linux.h)
elseif(WIN32)
  set(LITERT_HOST_MEMORY_SOURCES host_memory_windows.cc)
elseif(APPLE)
  set(LITERT_HOST_MEMORY_SOURCES host_memory_apple.cc)
else()
  set(LITERT_HOST_MEMORY_SOURCES host_memory_unsupported.cc)
endif()

# ==============================================================================
# Accelerator memory: accelerator_memory.cc is platform-neutral; exactly one
# accelerator_memory_<os>.cc (.mm on Apple platforms) implements the platform
# contract for the target OS. Linux and Android share the /proc/self/fdinfo
# reader (fdinfo.cc), the dma-buf reader built on it
# (accelerator_memory_dmabuf.cc), the /proc/self/{maps,status,smaps_rollup}
# readers (proc_mappings.cc) and the Android combination rule
# (accelerator_memory_android_sources.cc). Keep this in sync with the select()
# in BUILD.
# ==============================================================================
set(LITERT_ACCELERATOR_MEMORY_SOURCES accelerator_memory.cc)
if(ANDROID)
  list(APPEND LITERT_ACCELERATOR_MEMORY_SOURCES
    accelerator_memory_android.cc
    accelerator_memory_android.h
    accelerator_memory_android_sources.cc
    accelerator_memory_dmabuf.cc
    accelerator_memory_dmabuf.h
    fdinfo.cc
    fdinfo.h
    proc_mappings.cc
    proc_mappings.h
  )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND LITERT_ACCELERATOR_MEMORY_SOURCES
    accelerator_memory_android.h
    accelerator_memory_android_sources.cc
    accelerator_memory_dmabuf.cc
    accelerator_memory_dmabuf.h
    accelerator_memory_linux.cc
    accelerator_memory_linux.h
    accelerator_memory_nvml.cc
    accelerator_memory_nvml.h
    fdinfo.cc
    fdinfo.h
    proc_mappings.cc
    proc_mappings.h
  )
elseif(WIN32)
  list(APPEND LITERT_ACCELERATOR_MEMORY_SOURCES
    accelerator_memory_windows.cc
    accelerator_memory_windows.h
  )
elseif(APPLE)
  # Objective-C++: it reads MTLDevice. The rest of the build is C / C++.
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD ${CMAKE_CXX_STANDARD})
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
  list(APPEND LITERT_ACCELERATOR_MEMORY_SOURCES
    accelerator_memory_apple.mm
    accelerator_memory_apple.h
  )
else()
  list(APPEND LITERT_ACCELERATOR_MEMORY_SOURCES accelerator_memory_unsupported.cc)
endif()

add_library(litert_memory_profiling STATIC
  ${LITERT_HOST_MEMORY_SOURCES}
  ${LITERT_ACCELERATOR_MEMORY_SOURCES}
  accelerator_memory.h
  host_memory.h
  memory_monitor.cc
  memory_monitor.h
  memory_snapshot.cc
  memory_snapshot.h
)

target_include_directories(litert_memory_profiling
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../..>
)

target_link_libraries(litert_memory_profiling
  PUBLIC
    absl::core_headers
    absl::flat_hash_map
    absl::flat_hash_set
    absl::function_ref
    absl::log
    absl::str_format
    absl::strings
    absl::synchronization
    absl::time
  PRIVATE
    Threads::Threads
)

if(WIN32)
  # dxgi: IDXGIAdapter3::QueryVideoMemoryInfo for per-process video memory
  # usage. gdi32: D3DKMTQueryStatistics for the part of it that is charged to
  # the process's private memory footprint.
  target_link_libraries(litert_memory_profiling PUBLIC dxgi gdi32)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # dlopen() of libnvidia-ml.so at run time.
  target_link_libraries(litert_memory_profiling PUBLIC ${CMAKE_DL_LIBS})
elseif(APPLE)
  # Metal: MTLDevice.currentAllocatedSize. Foundation: NSArray / NSString.
  target_link_libraries(litert_memory_profiling
    PUBLIC "-framework Metal" "-framework Foundation")
endif()
