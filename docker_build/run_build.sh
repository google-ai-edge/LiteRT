#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
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

# Default build script for the Docker hermetic build environment.
# This is the CMD target in hermetic_build.Dockerfile.  Individual
# build_*.sh scripts override the CMD to run their own targets while
# sourcing setup_bazel_env.sh for proxy/SVE helpers.
#
# To build different targets, override from your host script:
#   docker run ... litert_build_env bash -c 'source /setup_bazel_env.sh && bazel ${EXTRA_STARTUP} build //your:target'

set -euo pipefail

source /setup_bazel_env.sh

bazel ${EXTRA_STARTUP} build //litert/runtime:compiled_model
