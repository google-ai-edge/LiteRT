# Copyright 2025 Google LLC.
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

"""Workspace definition for LiteRT-GPU library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def litert_gpu():
    http_archive(
        name = "litert_gpu",
        build_file = "@//third_party/litert_gpu:litert_gpu.BUILD",
        type = "jar",
        url = "https://dl.google.com/android/maven2/com/google/ai/edge/litert/litert/2.0.1-alpha/litert-2.0.1-alpha.aar",
    )
