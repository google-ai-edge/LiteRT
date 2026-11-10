############################################################
# Copyright (C) 2025 Broadcom.
# The term "Broadcom" refers to Broadcom Inc. and/or its subsidiaries.
#
# This program is the proprietary software of Broadcom and/or its licensors,
# and may only be used, duplicated, modified or distributed pursuant to
# the terms and conditions of a separate, written license agreement executed
# between you and Broadcom (an "Authorized License").  Except as set forth in
# an Authorized License, Broadcom grants no license (express or implied),
# right to use, or waiver of any kind with respect to the Software, and
# Broadcom expressly reserves all rights in and to the Software and all
# intellectual property rights therein. IF YOU HAVE NO AUTHORIZED LICENSE,
# THEN YOU HAVE NO RIGHT TO USE THIS SOFTWARE IN ANY WAY, AND SHOULD
# IMMEDIATELY NOTIFY BROADCOM AND DISCONTINUE ALL USE OF THE SOFTWARE.
#
# Except as expressly set forth in the Authorized License,
#
# 1.     This program, including its structure, sequence and organization,
# constitutes the valuable trade secrets of Broadcom, and you shall use all
# reasonable efforts to protect the confidentiality thereof, and to use this
# information only in connection with your use of Broadcom integrated circuit
# products.
#
# 2.     TO THE MAXIMUM EXTENT PERMITTED BY LAW, THE SOFTWARE IS PROVIDED
# "AS IS" AND WITH ALL FAULTS AND BROADCOM MAKES NO PROMISES, REPRESENTATIONS
# OR WARRANTIES, EITHER EXPRESS, IMPLIED, STATUTORY, OR OTHERWISE, WITH
# RESPECT TO THE SOFTWARE.  BROADCOM SPECIFICALLY DISCLAIMS ANY AND ALL
# IMPLIED WARRANTIES OF TITLE, MERCHANTABILITY, NONINFRINGEMENT, FITNESS FOR
# A PARTICULAR PURPOSE, LACK OF VIRUSES, ACCURACY OR COMPLETENESS, QUIET
# ENJOYMENT, QUIET POSSESSION OR CORRESPONDENCE TO DESCRIPTION. YOU ASSUME
# THE ENTIRE RISK ARISING OUT OF USE OR PERFORMANCE OF THE SOFTWARE.
#
# 3.     TO THE MAXIMUM EXTENT PERMITTED BY LAW, IN NO EVENT SHALL BROADCOM
# OR ITS LICENSORS BE LIABLE FOR (i) CONSEQUENTIAL, INCIDENTAL, SPECIAL,
# INDIRECT, OR EXEMPLARY DAMAGES WHATSOEVER ARISING OUT OF OR IN ANY WAY
# RELATING TO YOUR USE OF OR INABILITY TO USE THE SOFTWARE EVEN IF BROADCOM
# HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES; OR (ii) ANY AMOUNT IN
# EXCESS OF THE AMOUNT ACTUALLY PAID FOR THE SOFTWARE ITSELF OR U.S. $1,
# WHICHEVER IS GREATER. THESE LIMITATIONS SHALL APPLY NOTWITHSTANDING ANY
# FAILURE OF ESSENTIAL PURPOSE OF ANY LIMITED REMEDY.
############################################################

ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
LITERT_DIR := $(abspath $(ROOT_DIR)/../../..)
CORE_DIR := $(abspath $(ROOT_DIR)/../../../../bstorm/core)
CMAKE_TOOLS_DIR := $(abspath $(ROOT_DIR)/../../../../bstorm/cmake)

BSTORM_CHIP ?= 74110
BSTORM_CHIP_REV ?= b0
BSTORM_BSTM_REV ?= 2003
BSTORM_PLATFORM ?= 74110_B0

BRCM_TOOLCHAIN_TOP ?= $(realpath /opt/toolchains/stbgcc-12.4-2.0)
CMAKE_OUT ?= $(abspath $(ROOT_DIR)/../../../../build_arm64)
CMAKE_OUT_HOST ?= $(abspath $(ROOT_DIR)/../../../../build_host)
HOST_TOOLS ?= $(abspath $(ROOT_DIR)/../../../../host_tools_build)
FLATC_BINARY = ${HOST_TOOLS}/flatbuffers-flatc/bin/flatc

DEVICE_USER ?= user
DEVICE_IP ?= device
DEVICE_DIR ?= /tmp/bstorm_run

.PHONY: host_tools build_host compile_model build_target deploy_target run_target clean dist_clean

# 1. Build host FlatBuffers dependency
host_tools:
	mkdir -p $(HOST_TOOLS)
	cmake -DCMAKE_BUILD_TYPE=Release \
		-DTFLITE_SOURCE_DIR=$(LITERT_DIR)/tflite \
		-S "$(LITERT_DIR)/tflite/tools/cmake/native_tools/flatbuffers" \
		-B $(HOST_TOOLS)
	cmake --build $(HOST_TOOLS) --target flatbuffers-flatc

# 2. Build host compilation tool and compiler plugin
build_host: host_tools
	mkdir -p $(CMAKE_OUT_HOST)
	cmake -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CXX_STANDARD=17 \
		-DTFLITE_HOST_TOOLS_DIR=$(dir ${FLATC_BINARY}) \
		-DFLATC_EXECUTABLE=${FLATC_BINARY} \
		-DFLATBUFFERS_BUILD_FLATC=OFF \
		-DBSTORM_CHIP=$(BSTORM_CHIP) \
		-DBSTORM_CHIP_REV=$(BSTORM_CHIP_REV) \
		-DBSTORM_BSTM_REV=$(BSTORM_BSTM_REV) \
		-S $(LITERT_DIR)/litert \
		-B $(CMAKE_OUT_HOST)
	cmake --build $(CMAKE_OUT_HOST) -j$$(nproc) --target bstorm_compiler_plugin apply_plugin_main

# 3. Pre-compile .tflite model to .npu.tflite on host
compile_model:
	@test -n "$(MODEL)" || (echo "ERROR: Set MODEL=<path_to_tflite>" && exit 1)
	@test -n "$(MODEL_NPU)" || (echo "ERROR: Set MODEL_NPU=<path_to_npu_tflite>" && exit 1)
	$(CMAKE_OUT_HOST)/tools/apply_plugin_main \
		--cmd apply --libs $(CMAKE_OUT_HOST)/vendors/broadcom/compiler \
		--soc_manufacturer Broadcom --soc_model $(BSTORM_PLATFORM) \
		--model "$(MODEL)" --o "$(MODEL_NPU)"

# 4. Cross-compile target artifacts
build_target: host_tools
	mkdir -p $(CMAKE_OUT)
	cmake -DCMAKE_TOOLCHAIN_FILE=$(CMAKE_TOOLS_DIR)/toolchains.cmake \
		-Dtoolchain=$(BRCM_TOOLCHAIN_TOP) \
		-DBRCM_ARCH=aarch64 \
		-DCMAKE_BUILD_TYPE=Release \
		-DTFLITE_HOST_TOOLS_DIR=$(dir ${FLATC_BINARY}) \
		-DFLATC_EXECUTABLE=${FLATC_BINARY} \
		-DFLATBUFFERS_BUILD_FLATC=OFF \
		-DBSTORM_CHIP=$(BSTORM_CHIP) \
		-DBSTORM_CHIP_REV=$(BSTORM_CHIP_REV) \
		-DBSTORM_BSTM_REV=$(BSTORM_BSTM_REV) \
		"-DLITERT_TFLITE_EXTRA_CMAKE_ARGS=-DCMAKE_TOOLCHAIN_FILE=$(CMAKE_TOOLS_DIR)/toolchains.cmake;-Dtoolchain=$(BRCM_TOOLCHAIN_TOP);-DFLATBUFFERS_BUILD_FLATC=OFF;-DTFLITE_HOST_TOOLS_DIR=$(dir ${FLATC_BINARY})" \
		-S $(LITERT_DIR)/litert \
		-B $(CMAKE_OUT)
	cmake --build $(CMAKE_OUT) -j$$(nproc) --target \
		bstorm_dispatch_plugin bstorm_example_next bstorm_simple_server

# 5. Transfer binaries and compiled model to target hardware
deploy_target:
	@test -n "$(MODEL_NPU)" || (echo "ERROR: Set MODEL_NPU=<path_to_compiled_npu_tflite>" && exit 1)
	ssh $(DEVICE_USER)@$(DEVICE_IP) "mkdir -p $(DEVICE_DIR)"
	scp $(CMAKE_OUT)/vendors/broadcom/libbstorm_core.so \
		$(CMAKE_OUT)/vendors/broadcom/dispatch/libLiteRtDispatch_bstorm.so \
		$(CMAKE_OUT)/c/libLiteRt.so \
		$(CMAKE_OUT)/vendors/broadcom/example/bstorm_example_next \
		$(CMAKE_OUT)/vendors/broadcom/core_build/src/app/bstorm_simple_server \
		$(CORE_DIR)/scripts/target/bstm_simple_run.sh \
		$(MODEL_NPU) \
		$(if $(INPUT),$(INPUT)) \
		$(if $(EXPECTED),$(EXPECTED)) \
		$(DEVICE_USER)@$(DEVICE_IP):$(DEVICE_DIR)/

# 6. Execute model inference on target hardware
run_target:
	@MODEL_FILE="$(if $(MODEL_NPU),$(notdir $(MODEL_NPU)),$(MODEL_NAME))"; \
	test -n "$$MODEL_FILE" || (echo "ERROR: Set MODEL_NPU=<path_to_compiled_npu_tflite> or MODEL_NAME=<filename>" && exit 1); \
	ssh $(DEVICE_USER)@$(DEVICE_IP) "cd $(DEVICE_DIR) && \
		LD_LIBRARY_PATH=. \
		BSTORM_DISPATCH_OPTIONS=\"$(CMAKE_DISPATCH_OPT)\" \
		./bstm_simple_run.sh bstorm_simple_server \
		./bstorm_example_next \
		--dispatch_library_dir . \
		$(if $(ACCURACY),--accuracy $(ACCURACY)) \
		$(if $(INPUT),--input $(notdir $(INPUT))) \
		$(if $(EXPECTED),--expected $(notdir $(EXPECTED))) \
		$$MODEL_FILE"

# Clean build object files via CMake
clean:
	cmake --build $(CMAKE_OUT_HOST) --target clean
	cmake --build $(CMAKE_OUT) --target clean

# Full reset (wipes build outputs entirely)
dist_clean:
	rm -rf $(CMAKE_OUT) $(CMAKE_OUT_HOST) $(HOST_TOOLS)
