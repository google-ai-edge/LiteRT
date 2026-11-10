# Copyright 2026 Google LLC
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

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# 1. Check if Yocto/OE environment-setup script was sourced
if(DEFINED ENV{SDKTARGETSYSROOT})
  set(CMAKE_SYSROOT $ENV{SDKTARGETSYSROOT})
elseif(DEFINED ENV{OECORE_TARGET_SYSROOT})
  set(CMAKE_SYSROOT $ENV{OECORE_TARGET_SYSROOT})
endif()

# 2. Check explicit ESDK root environment variable
if(DEFINED ENV{LINUX_AARCH64_ESDK})
  set(ESDK_ROOT $ENV{LINUX_AARCH64_ESDK})
elseif(DEFINED ENV{OECORE_NATIVE_SYSROOT})
  get_filename_component(ESDK_ROOT "$ENV{OECORE_NATIVE_SYSROOT}/../.." ABSOLUTE)
endif()

# 3. Locate C / C++ Compilers
if(DEFINED ENV{CC} AND "$ENV{CC}" MATCHES "aarch64.*(gcc|clang)")
  separate_arguments(_CC_ARGS NATIVE_COMMAND "$ENV{CC}")
  list(GET _CC_ARGS 0 CMAKE_C_COMPILER)
  list(REMOVE_AT _CC_ARGS 0)
  if(_CC_ARGS)
    string(JOIN " " _CC_FLAGS ${_CC_ARGS})
    set(CMAKE_C_FLAGS_INIT "${_CC_FLAGS}")
  endif()
elseif(DEFINED ESDK_ROOT)
  file(GLOB COMPILER_C_CANDIDATES
    "${ESDK_ROOT}/tmp/sysroots/x86_64/usr/bin/aarch64-*/aarch64-*-gcc"
    "${ESDK_ROOT}/sysroots/x86_64*linux/usr/bin/aarch64-*/aarch64-*-gcc"
    "${ESDK_ROOT}/usr/bin/aarch64-*/aarch64-*-gcc"
  )
  list(LENGTH COMPILER_C_CANDIDATES NUM_C_CANDIDATES)
  if(NUM_C_CANDIDATES GREATER 0)
    list(GET COMPILER_C_CANDIDATES 0 CMAKE_C_COMPILER)
  else()
    set(CMAKE_C_COMPILER ${ESDK_ROOT}/tmp/sysroots/x86_64/usr/bin/aarch64-qcom-linux/aarch64-qcom-linux-gcc)
  endif()
else()
  message(FATAL_ERROR "LINUX_AARCH64_ESDK is not set, and no OE SDK environment is active. "
    "Please set LINUX_AARCH64_ESDK to the root of your Qualcomm ESDK installation, e.g.: "
    "export LINUX_AARCH64_ESDK=/path/to/ESDK/install")
endif()

if(DEFINED ENV{CXX} AND "$ENV{CXX}" MATCHES "aarch64.*(g\\+\\+|c\\+\\+|clang\\+\\+)")
  separate_arguments(_CXX_ARGS NATIVE_COMMAND "$ENV{CXX}")
  list(GET _CXX_ARGS 0 CMAKE_CXX_COMPILER)
  list(REMOVE_AT _CXX_ARGS 0)
  if(_CXX_ARGS)
    string(JOIN " " _CXX_FLAGS ${_CXX_ARGS})
    set(CMAKE_CXX_FLAGS_INIT "${_CXX_FLAGS}")
  endif()
elseif(DEFINED ESDK_ROOT)
  file(GLOB COMPILER_CXX_CANDIDATES
    "${ESDK_ROOT}/tmp/sysroots/x86_64/usr/bin/aarch64-*/aarch64-*-g++"
    "${ESDK_ROOT}/sysroots/x86_64*linux/usr/bin/aarch64-*/aarch64-*-g++"
    "${ESDK_ROOT}/usr/bin/aarch64-*/aarch64-*-g++"
  )
  list(LENGTH COMPILER_CXX_CANDIDATES NUM_CXX_CANDIDATES)
  if(NUM_CXX_CANDIDATES GREATER 0)
    list(GET COMPILER_CXX_CANDIDATES 0 CMAKE_CXX_COMPILER)
  else()
    set(CMAKE_CXX_COMPILER ${ESDK_ROOT}/tmp/sysroots/x86_64/usr/bin/aarch64-qcom-linux/aarch64-qcom-linux-g++)
  endif()
elseif(DEFINED CMAKE_C_COMPILER)
  # Derive CXX from C compiler if C was specified via ENV{CC}
  string(REGEX REPLACE "gcc$" "g++" _DERIVED_CXX "${CMAKE_C_COMPILER}")
  string(REGEX REPLACE "clang$" "clang++" _DERIVED_CXX "${_DERIVED_CXX}")
  set(CMAKE_CXX_COMPILER "${_DERIVED_CXX}")
endif()

# 4. Locate Sysroot
if(NOT DEFINED CMAKE_SYSROOT AND DEFINED ESDK_ROOT)
  file(GLOB _RAW_SYSROOT_CANDIDATES
    "${ESDK_ROOT}/tmp/sysroots/qcs*"
    "${ESDK_ROOT}/tmp/sysroots/qrb*"
    "${ESDK_ROOT}/tmp/sysroots/armv8*"
    "${ESDK_ROOT}/tmp/sysroots/aarch64*"
    "${ESDK_ROOT}/sysroots/qcs*"
    "${ESDK_ROOT}/sysroots/qrb*"
    "${ESDK_ROOT}/sysroots/armv8*"
    "${ESDK_ROOT}/sysroots/aarch64*"
  )
  list(FILTER _RAW_SYSROOT_CANDIDATES EXCLUDE REGEX ".*x86_64.*")
  set(SYSROOT_CANDIDATES "")
  foreach(_candidate IN LISTS _RAW_SYSROOT_CANDIDATES)
    if(IS_DIRECTORY "${_candidate}/usr/include" AND IS_DIRECTORY "${_candidate}/usr/lib")
      list(APPEND SYSROOT_CANDIDATES "${_candidate}")
    endif()
  endforeach()
  list(LENGTH SYSROOT_CANDIDATES NUM_SYSROOTS)
  if(NUM_SYSROOTS GREATER 0)
    list(GET SYSROOT_CANDIDATES 0 CMAKE_SYSROOT)
  else()
    set(CMAKE_SYSROOT ${ESDK_ROOT}/tmp/sysroots/qcs8275-iq-8275-evk-pro-sku)
  endif()
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# 5. Fail-fast validation of resolved compiler and sysroot paths
if(NOT IS_ABSOLUTE "${CMAKE_C_COMPILER}")
  find_program(_RESOLVED_C_COMPILER "${CMAKE_C_COMPILER}" NO_CACHE)
  if(_RESOLVED_C_COMPILER)
    set(CMAKE_C_COMPILER "${_RESOLVED_C_COMPILER}")
  endif()
endif()
if(NOT EXISTS "${CMAKE_C_COMPILER}")
  message(FATAL_ERROR "C compiler not found at: ${CMAKE_C_COMPILER}")
endif()

if(NOT IS_ABSOLUTE "${CMAKE_CXX_COMPILER}")
  find_program(_RESOLVED_CXX_COMPILER "${CMAKE_CXX_COMPILER}" NO_CACHE)
  if(_RESOLVED_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${_RESOLVED_CXX_COMPILER}")
  endif()
endif()
if(NOT EXISTS "${CMAKE_CXX_COMPILER}")
  message(FATAL_ERROR "C++ compiler not found at: ${CMAKE_CXX_COMPILER}")
endif()

if(NOT EXISTS "${CMAKE_SYSROOT}/usr/include")
  message(FATAL_ERROR "Target sysroot directory (with usr/include) not found at: ${CMAKE_SYSROOT}")
endif()

