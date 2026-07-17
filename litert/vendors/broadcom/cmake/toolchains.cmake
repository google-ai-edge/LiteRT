##############################################################################
#  Copyright (C) 2022 Broadcom.
#  The term "Broadcom" refers to Broadcom Inc. and/or its subsidiaries.
#
#  This program is the proprietary software of Broadcom and/or its licensors,
#  and may only be used, duplicated, modified or distributed pursuant to
#  the terms and conditions of a separate, written license agreement executed
#  between you and Broadcom (an "Authorized License").  Except as set forth in
#  an Authorized License, Broadcom grants no license (express or implied),
#  right to use, or waiver of any kind with respect to the Software, and
#  Broadcom expressly reserves all rights in and to the Software and all
#  intellectual property rights therein. IF YOU HAVE NO AUTHORIZED LICENSE,
#  THEN YOU HAVE NO RIGHT TO USE THIS SOFTWARE IN ANY WAY, AND SHOULD
#  IMMEDIATELY NOTIFY BROADCOM AND DISCONTINUE ALL USE OF THE SOFTWARE.
#
#  Except as expressly set forth in the Authorized License,
#
#  1.     This program, including its structure, sequence and organization,
#  constitutes the valuable trade secrets of Broadcom, and you shall use all
#  reasonable efforts to protect the confidentiality thereof, and to use this
#  information only in connection with your use of Broadcom integrated circuit
#  products.
#
#  2.     TO THE MAXIMUM EXTENT PERMITTED BY LAW, THE SOFTWARE IS PROVIDED
#  "AS IS" AND WITH ALL FAULTS AND BROADCOM MAKES NO PROMISES, REPRESENTATIONS
#  OR WARRANTIES, EITHER EXPRESS, IMPLIED, STATUTORY, OR OTHERWISE, WITH
#  RESPECT TO THE SOFTWARE.  BROADCOM SPECIFICALLY DISCLAIMS ANY AND ALL
#  IMPLIED WARRANTIES OF TITLE, MERCHANTABILITY, NONINFRINGEMENT, FITNESS FOR
#  A PARTICULAR PURPOSE, LACK OF VIRUSES, ACCURACY OR COMPLETENESS, QUIET
#  ENJOYMENT, QUIET POSSESSION OR CORRESPONDENCE TO DESCRIPTION. YOU ASSUME
#  THE ENTIRE RISK ARISING OUT OF USE OR PERFORMANCE OF THE SOFTWARE.
#
#  3.     TO THE MAXIMUM EXTENT PERMITTED BY LAW, IN NO EVENT SHALL BROADCOM
#  OR ITS LICENSORS BE LIABLE FOR (i) CONSEQUENTIAL, INCIDENTAL, SPECIAL,
#  INDIRECT, OR EXEMPLARY DAMAGES WHATSOEVER ARISING OUT OF OR IN ANY WAY
#  RELATING TO YOUR USE OF OR INABILITY TO USE THE SOFTWARE EVEN IF BROADCOM
#  HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES; OR (ii) ANY AMOUNT IN
#  EXCESS OF THE AMOUNT ACTUALLY PAID FOR THE SOFTWARE ITSELF OR U.S. $1,
#  WHICHEVER IS GREATER. THESE LIMITATIONS SHALL APPLY NOTWITHSTANDING ANY
#  FAILURE OF ESSENTIAL PURPOSE OF ANY LIMITED REMEDY.
##############################################################################

# CMake toolchain file for the stbgcc toolchains
# set `BRCM_ARCH` to aarch64 for arm64 bit

# Optionally set `toolchain` to point to the location of the toolchain
if(STBGCC_CMAKE_INCLUDED OR CMAKE_PROJECT_NAME STREQUAL CMAKE_TRY_COMPILE)
  return()
endif()
set(STBGCC_CMAKE_INCLUDED true)

if (NOT DEFINED toolchain)
  message(FATAL_ERROR "missing -D toolchain=<toolchain location> command line option")
endif()

if (NOT DEFINED BRCM_ARCH)
  set(BRCM_ARCH "aarch64" CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES BRCM_ARCH toolchain)

set(CMAKE_SYSTEM_NAME Linux)
if (BRCM_ARCH STREQUAL "aarch64")
  set(CMAKE_SYSTEM_PROCESSOR aarch64)
  set(CMAKE_SYSROOT ${toolchain}/aarch64-unknown-linux-gnu/sys-root)
  set(CMAKE_C_COMPILER ${toolchain}/bin/aarch64-linux-gcc)
  set(CMAKE_CXX_COMPILER ${toolchain}/bin/aarch64-linux-g++)
elseif(BRCM_ARCH STREQUAL "armv7l")
  set(CMAKE_SYSTEM_PROCESSOR "armv7l")
  set(CMAKE_SYSROOT ${toolchain}/arm-unknown-linux-gnueabihf/sys-root)
  set(CMAKE_C_COMPILER ${toolchain}/bin/arm-linux-gcc)
  set(CMAKE_CXX_COMPILER ${toolchain}/bin/arm-linux-g++)
elseif(BRCM_ARCH STREQUAL "armv7a")
  set(CMAKE_SYSTEM_PROCESSOR "armv7a")
  set(CMAKE_SYSROOT ${toolchain}/arm-unknown-linux-gnueabihf/sys-root)
  set(CMAKE_C_COMPILER ${toolchain}/bin/arm-linux-gcc)
  set(CMAKE_CXX_COMPILER ${toolchain}/bin/arm-linux-g++)
else()
  message(FATAL_ERROR "Specified architecture is not supported by this toolchain file : ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if (NOT EXISTS ${CMAKE_SYSROOT})
  execute_process(COMMAND ${CMAKE_C_COMPILER} -print-sysroot
    OUTPUT_VARIABLE DETECTED_SYSROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
  get_filename_component(CMAKE_SYSROOT ${DETECTED_SYSROOT} ABSOLUTE)
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
