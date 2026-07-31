# Copyright (c) 2026 Broadcom. All Rights Reserved.
# The term "Broadcom" refers to Broadcom Inc. and/or its subsidiaries.
# This software is owned by Broadcom and licensed under Apache-2.0.
#
# SPDX-License-Identifier: Apache-2.0
#
# Convenience wrapper around build_binaries.sh / run_inference.sh so a
# customer building this repo can:
#  - cross-compile the Broadcom LiteRT dispatch plugin + runtime for target
#  - host-build the Broadcom compiler plugin and AOT-compile a .tflite model
#  - run a single model on target
# with familiar `make` targets.
#
# Typical flow (see the Broadcom LiteRT Plugin Release Guide for full setup,
# including bstorm_headers/bstorm_binaries/models tarballs under vendors/broadcom/):
#   export BSTORM_SDK_CORE_OUT=<bstorm sdk>/core/out/core.<chip>.<rev>.target
#   make build PLATFORM=74110_B0
#   make build-host
#   make compile-model MODEL=<model_name> PLATFORM=74110_B0
#   make run TARGET=<ip> PLATFORM=74110_B0 MODEL=<model_name>
#
# See build_binaries.sh and run_inference.sh for full option/env docs.

PLATFORM ?=
TARGET ?=
MODEL ?=
PORT ?=
USER_ ?=
MODELS_DIR ?= models
SKIP_NODES ?=
GRAPH_TOOL ?=
GRAPH_TRANSFORM ?=
GRAPH_PROCESS ?=
comma := ,

.PHONY: all build build-host compile-model run list-models clean help

all: build

build:
	@test -n "$(PLATFORM)" || { echo "ERROR: PLATFORM=<platform> is required, e.g. PLATFORM=74110_B0"; exit 1; }
	./build_binaries.sh --platform "$(PLATFORM)"

build-host:
	./build_binaries.sh --host

# AOT-compiles models/<MODEL>/<MODEL>.tflite into
# models/<MODEL>/<MODEL>.npu.tflite using the host-built compiler plugin
# (see `make build-host`). The Broadcom subgraph is compiled for PLATFORM
# and embedded in the output; no target board is required for this step.
#
# Optional graph-transform knobs, forwarded to the compiler plugin via
# BSTORM_COMPILER_PLUGIN_OPTIONS (see compiler/bstorm_compiler_plugin.cc):
#   SKIP_NODES=<n|n-m>        skip-nodes    - exclude node(s)/range from the partition
#   GRAPH_TOOL=<path>         graph-tool    - external graph transform/process tool binary
#   GRAPH_TRANSFORM=<type>    graph-transform - run `<tool> transform --type <type>` first
#   GRAPH_PROCESS=<p1,p2,..>  graph-process - run `<tool> process --process <p1+p2+...>` first
# GRAPH_TRANSFORM and GRAPH_PROCESS are mutually exclusive; GRAPH_PROCESS wins if both are set.
compile-model:
	@test -n "$(MODEL)" || { echo "ERROR: MODEL=<name> is required, e.g. MODEL=torchvision_mobilenet_v2"; exit 1; }
	@test -n "$(PLATFORM)" || { echo "ERROR: PLATFORM=<platform> is required, e.g. PLATFORM=74110_B0"; exit 1; }
	@test -f "lib/host/apply_plugin_main" || { echo "ERROR: lib/host/apply_plugin_main not found. Run: make build-host"; exit 1; }
	BSTORM_COMPILER_PLUGIN_OPTIONS="graph-name=$(MODEL),graph-path=$(MODELS_DIR)/$(MODEL)$(if $(SKIP_NODES),$(comma)skip-nodes=$(SKIP_NODES))$(if $(GRAPH_PROCESS),$(comma)graph-process=$(subst $(comma),+,$(GRAPH_PROCESS)),$(if $(GRAPH_TRANSFORM),$(comma)graph-transform=$(GRAPH_TRANSFORM)))$(if $(GRAPH_TOOL),$(comma)graph-tool=$(GRAPH_TOOL))" \
	./lib/host/apply_plugin_main --cmd apply \
		--model "$(MODELS_DIR)/$(MODEL)/$(MODEL).tflite" \
		--soc_manufacturer Broadcom --soc_model "$(PLATFORM)" \
		--libs lib/host \
		--o "$(MODELS_DIR)/$(MODEL)/$(MODEL).npu.tflite"

run:
	@test -n "$(TARGET)" || { echo "ERROR: TARGET=<ip> is required (or set BSTM_HW_TARGET)"; exit 1; }
	@test -n "$(PLATFORM)" || { echo "ERROR: PLATFORM=<platform> is required, e.g. PLATFORM=74110_B0"; exit 1; }
	./run_inference.sh --target "$(TARGET)" --platform "$(PLATFORM)" \
		$(if $(PORT),--port "$(PORT)") \
		$(if $(USER_),--user "$(USER_)") \
		$(if $(MODEL),--model "$(MODEL)")

list-models:
	./run_inference.sh --list-models

clean:
	./build_binaries.sh --clean
	./run_inference.sh --clean

help:
	@echo "Targets:"
	@echo "  make build PLATFORM=<p>                        - cross-compile dispatch plugin + runtime into lib/"
	@echo "  make build-host                                - host-build compiler plugin + apply_plugin_main into lib/host/"
	@echo "  make compile-model MODEL=<m> PLATFORM=<p>       - AOT-compile models/<m>/<m>.tflite -> <m>.npu.tflite"
	@echo "                     [SKIP_NODES=<n|n-m>] [GRAPH_TOOL=<path>]"
	@echo "                     [GRAPH_TRANSFORM=<type> | GRAPH_PROCESS=<p1,p2,...>]"
	@echo "  make run TARGET=<ip> PLATFORM=<p> [MODEL=<m>]   - deploy to target board and run inference"
	@echo "  make list-models                                - list models available under models/"
	@echo "  make clean                                      - remove build outputs and staged libs"
