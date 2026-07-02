# Example toolchain file for a Qualcomm aarch64 (oe-linux) target.
# Copy this file, edit the paths below to match your eSDK install, then:
#   export LITERT_TOOLCHAIN_FILE=/path/to/this/file.cmake
#   cmake --preset custom-toolchain

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# <ESDK> is the root of your Qualcomm eSDK install.
set(CMAKE_C_COMPILER   <ESDK>/tmp/sysroots/x86_64/usr/bin/aarch64-qcom-linux/aarch64-qcom-linux-gcc)
set(CMAKE_CXX_COMPILER <ESDK>/tmp/sysroots/x86_64/usr/bin/aarch64-qcom-linux/aarch64-qcom-linux-g++)
set(CMAKE_SYSROOT      <ESDK>/tmp/sysroots/qcs8275-iq-8275-evk-pro-sku)

# Optional: only if the host flatc auto-detect (clang -> cc -> gcc) picks wrong.
# set(LITERT_HOST_C_COMPILER   /usr/bin/clang)
# set(LITERT_HOST_CXX_COMPILER /usr/bin/clang++)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
