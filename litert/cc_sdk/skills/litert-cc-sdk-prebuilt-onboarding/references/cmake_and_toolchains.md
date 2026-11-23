# CMake, Toolchain, and Deployment Templates for LiteRT C++ SDK Prebuilts

## 1. Application `CMakeLists.txt` (Stages `litert_cc_sdk` + `libLiteRt`)

The CMake script inside `litert_cc_sdk/CMakeLists.txt` declares `litert_c_api`
with `IMPORTED_LOCATION` pointing to `${CMAKE_CURRENT_SOURCE_DIR}/libLiteRt.so`
(Linux/Android), `libLiteRt.dylib` (macOS/iOS), or `libLiteRt.dll` (Windows).
Because the prebuilt archive places `litert_cc_sdk/` and
`<platform>/libLiteRt.*` in sibling directories, copy both into
`${CMAKE_BINARY_DIR}/litert_cc_sdk_staged` before calling
`add_subdirectory(... EXCLUDE_FROM_ALL)`:

```cmake
cmake_minimum_required(VERSION 3.22)
project(litert_prebuilt_app LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LITERT_CC_SDK_DIR "" CACHE PATH "Path to unpacked litert_cc_sdk directory")
set(LITERT_PREBUILT_DIR "" CACHE PATH "Path to platform prebuilt directory containing libLiteRt")

if(WIN32)
  set(_LITERT_C_API_LIB_NAME "libLiteRt.dll")
elseif(APPLE)
  set(_LITERT_C_API_LIB_NAME "libLiteRt.dylib")
else()
  set(_LITERT_C_API_LIB_NAME "libLiteRt.so")
endif()

if(NOT EXISTS "${LITERT_CC_SDK_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "LITERT_CC_SDK_DIR must point to litert_cc_sdk/ (${LITERT_CC_SDK_DIR})")
endif()
if(NOT EXISTS "${LITERT_PREBUILT_DIR}/${_LITERT_C_API_LIB_NAME}")
  message(FATAL_ERROR "LITERT_PREBUILT_DIR must contain ${_LITERT_C_API_LIB_NAME} (${LITERT_PREBUILT_DIR})")
endif()

# Stage litert_cc_sdk and libLiteRt into a single build-tree directory so
# the unmodified litert_cc_sdk/CMakeLists.txt resolves ${CMAKE_CURRENT_SOURCE_DIR}/${_LITERT_C_API_LIB_NAME}.
# On Windows, litert_cc_sdk/CMakeLists.txt generates libLiteRt.lib (IMPORTED_IMPLIB)
# from libLiteRt.dll via dumpbin /exports and lib /def if libLiteRt.lib is not present.
set(_LITERT_SDK_STAGED "${CMAKE_BINARY_DIR}/litert_cc_sdk_staged")
file(MAKE_DIRECTORY "${_LITERT_SDK_STAGED}")
file(COPY "${LITERT_CC_SDK_DIR}/" DESTINATION "${_LITERT_SDK_STAGED}")
file(COPY "${LITERT_PREBUILT_DIR}/${_LITERT_C_API_LIB_NAME}" DESTINATION "${_LITERT_SDK_STAGED}")
if(WIN32 AND EXISTS "${LITERT_PREBUILT_DIR}/libLiteRt.lib")
  file(COPY "${LITERT_PREBUILT_DIR}/libLiteRt.lib" DESTINATION "${_LITERT_SDK_STAGED}")
endif()

# EXCLUDE_FROM_ALL prevents building the SDK's optional sample binaries by default.
add_subdirectory(
  "${_LITERT_SDK_STAGED}"
  "${CMAKE_BINARY_DIR}/litert_cc_sdk_build"
  EXCLUDE_FROM_ALL
)

add_executable(my_litert_app main.cc)

# litert_cc_api declares the SDK header tree, generated includes, Abseil, and
# FlatBuffers as PUBLIC dependencies, so linking litert_cc_api propagates those
# include directories automatically.
target_link_libraries(my_litert_app PRIVATE
  litert_cc_api
)

if(WIN32)
  add_custom_command(TARGET my_litert_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${_LITERT_SDK_STAGED}/libLiteRt.dll"
      "$<TARGET_FILE_DIR:my_litert_app>/libLiteRt.dll"
  )
elseif(ANDROID)
  target_link_libraries(my_litert_app PRIVATE
    "${_LITERT_SDK_STAGED}/${_LITERT_C_API_LIB_NAME}"
    log android dl
  )
elseif(UNIX)
  target_link_libraries(my_litert_app PRIVATE
    "${_LITERT_SDK_STAGED}/${_LITERT_C_API_LIB_NAME}"
    pthread dl
  )
  set_target_properties(my_litert_app PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
  )
endif()
```

---

## 2. Linux `aarch64` Cross-Compile Toolchain (`toolchain_aarch64_ubuntu.cmake`)

When cross-compiling from an `x86_64` Linux host for a Qualcomm Linux / Ubuntu
`aarch64` IoT board that has no on-device compiler, use host `clang`/`clang++`
with `--target=aarch64-linux-gnu`, an `aarch64` Ubuntu sysroot, and the Android
NDK's `ld.lld` linker:

```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(TRIPLE aarch64-linux-gnu)
if(DEFINED ENV{IOT_SYSROOT})
  set(SYSROOT "$ENV{IOT_SYSROOT}")
else()
  message(FATAL_ERROR "Set ENV{IOT_SYSROOT} to the aarch64 Ubuntu sysroot path")
endif()

if(DEFINED ENV{IOT_LLD})
  set(LLD_PATH "$ENV{IOT_LLD}")
else()
  message(FATAL_ERROR "Set ENV{IOT_LLD} to the ld.lld binary path")
endif()

set(CMAKE_SYSROOT "${SYSROOT}")
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER_TARGET ${TRIPLE})
set(CMAKE_CXX_COMPILER_TARGET ${TRIPLE})

# Adjust GCC_VER to match the GCC version in your target sysroot
# (e.g., 13 for Ubuntu 24.04 Noble, 11 for Ubuntu 22.04 Jammy).
set(GCC_VER "13" CACHE STRING "GCC version in the target aarch64 sysroot")

set(_COMMON_FLAGS
  "--target=${TRIPLE} --sysroot=${SYSROOT} "
  "-isystem ${SYSROOT}/usr/include/c++/${GCC_VER} "
  "-isystem ${SYSROOT}/usr/include/${TRIPLE}/c++/${GCC_VER} "
  "-isystem ${SYSROOT}/usr/include/${TRIPLE} "
  "-isystem ${SYSROOT}/usr/include"
)
string(CONCAT _COMMON_FLAGS ${_COMMON_FLAGS})

set(_LINK_FLAGS
  "--target=${TRIPLE} --sysroot=${SYSROOT} "
  "-fuse-ld=${LLD_PATH} "
  "-B${SYSROOT}/usr/lib/gcc-cross/${TRIPLE}/${GCC_VER} "
  "-B${SYSROOT}/usr/lib/${TRIPLE} "
  "-L${SYSROOT}/usr/lib/gcc-cross/${TRIPLE}/${GCC_VER} "
  "-L${SYSROOT}/usr/lib/${TRIPLE} "
  "-Wl,-rpath-link,${SYSROOT}/usr/lib/${TRIPLE}"
)
string(CONCAT _LINK_FLAGS ${_LINK_FLAGS})

set(CMAKE_C_FLAGS_INIT "${_COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${_COMMON_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${_LINK_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${_LINK_FLAGS}")

set(CMAKE_FIND_ROOT_PATH "${SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

### Constructing the `aarch64` Ubuntu 24.04 (Noble) Sysroot on an `x86_64` Host

Download and unpack the `arm64` `.deb` packages from
`http://ports.ubuntu.com/ubuntu-ports`:

- `libc6`, `libc6-dev`, `linux-libc-dev`
- `libgcc-s1`, `libgcc-13-dev`, `libstdc++6`, `libstdc++-13-dev`, `libatomic1`,
  `libcrypt-dev`

After extracting each `.deb` with `dpkg -x <pkg>.deb "${IOT_SYSROOT}"`:

1. Create the top-level symlink `ln -snf usr/lib "${IOT_SYSROOT}/lib"`.
2. Convert any absolute symlinks under
   `${IOT_SYSROOT}/usr/lib/aarch64-linux-gnu` that start with `/lib/...` or
   `/usr/lib/...` into relative symlinks so `ld.lld` resolves them inside
   `${IOT_SYSROOT}` instead of on the `x86_64` host.
