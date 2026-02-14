# Copyright 2017 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""# Bazel rules for creating macOS applications and bundles."""

load(
    "//apple/internal:macos_rules.bzl",
    _macos_application = "macos_application",
    _macos_bundle = "macos_bundle",
    _macos_command_line_application = "macos_command_line_application",
    _macos_dylib = "macos_dylib",
    _macos_dynamic_framework = "macos_dynamic_framework",
    _macos_extension = "macos_extension",
    _macos_framework = "macos_framework",
    _macos_kernel_extension = "macos_kernel_extension",
    _macos_quick_look_plugin = "macos_quick_look_plugin",
    _macos_spotlight_importer = "macos_spotlight_importer",
    _macos_static_framework = "macos_static_framework",
    _macos_xpc_service = "macos_xpc_service",
)

# Re-export original rules rather than their wrapper macros
# so that stardoc documents the rule attributes, not an opaque
# **kwargs argument.
load(
    "//apple/internal/testing:macos_rules.bzl",
    _macos_ui_test = "macos_ui_test",
    _macos_unit_test = "macos_unit_test",
)
load(":macos.bzl", _macos_build_test = "macos_build_test")

macos_application = _macos_application
macos_bundle = _macos_bundle
macos_command_line_application = _macos_command_line_application
macos_dylib = _macos_dylib
macos_extension = _macos_extension
macos_kernel_extension = _macos_kernel_extension
macos_quick_look_plugin = _macos_quick_look_plugin
macos_spotlight_importer = _macos_spotlight_importer
macos_unit_test = _macos_unit_test
macos_ui_test = _macos_ui_test
macos_xpc_service = _macos_xpc_service
macos_build_test = _macos_build_test

macos_framework = _macos_framework
macos_dynamic_framework = _macos_dynamic_framework
macos_static_framework = _macos_static_framework
