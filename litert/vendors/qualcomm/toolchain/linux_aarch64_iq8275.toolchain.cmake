set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

if(NOT DEFINED ENV{LINUX_AARCH64_ESDK})
  message(FATAL_ERROR "LINUX_AARCH64_ESDK is not set. "
    "Please set it to the root of your Qualcomm ESDK installation, e.g.: "
    "export LINUX_AARCH64_ESDK=/path/to/ESDK/install")
endif()

set(CMAKE_C_COMPILER $ENV{LINUX_AARCH64_ESDK}/tmp/sysroots/x86_64/usr/bin/aarch64-qcom-linux/aarch64-qcom-linux-gcc)
set(CMAKE_CXX_COMPILER $ENV{LINUX_AARCH64_ESDK}/tmp/sysroots/x86_64/usr/bin/aarch64-qcom-linux/aarch64-qcom-linux-g++)
set(CMAKE_SYSROOT $ENV{LINUX_AARCH64_ESDK}/tmp/sysroots/qcs8275-iq-8275-evk-pro-sku)
