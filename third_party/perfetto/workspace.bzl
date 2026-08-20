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

"""Workspace definition for the Perfetto library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "perfetto",
        # The patch is necessary to:
        # - avoid an error that happens when the hermetic python toolchain is
        #   too recent;
        # - remove the @rules_android dependency in Perfetto's bazel/rules.bzl
        #   and bazel/run_ait_with_adb.bzl which breaks WORKSPACE-based Android
        #   builds (see note in Perfetto's bazel/deps.bzl perfetto_deps()).
        patch_args = ["-p1"],
        patches = ["//:PATCH.perfetto"],
        sha256 = "b25023f3281165a1a7d7cde9f3ed2dfcfce022ffd727e77f6589951e0ba6af9a",
        strip_prefix = "perfetto-53.0",
        urls = ["https://github.com/google/perfetto/archive/refs/tags/v53.0.tar.gz"],
    )

    http_archive(
        name = "perfetto_cfg",
        build_file_content = "exports_files([\"perfetto_cfg.bzl\"])",
        sha256 = "b25023f3281165a1a7d7cde9f3ed2dfcfce022ffd727e77f6589951e0ba6af9a",
        strip_prefix = "perfetto-53.0/bazel/standalone",
        urls = ["https://github.com/google/perfetto/archive/refs/tags/v53.0.tar.gz"],
    )
