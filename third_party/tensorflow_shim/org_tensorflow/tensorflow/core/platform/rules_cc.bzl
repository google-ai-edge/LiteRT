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

"""Minimal stand-in for @org_tensorflow//tensorflow/core/platform:rules_cc.bzl."""

load("@rules_cc//cc:cc_binary.bzl", _cc_binary = "cc_binary")
load("@rules_cc//cc:cc_import.bzl", _cc_import = "cc_import")
load("@rules_cc//cc:cc_library.bzl", _cc_library = "cc_library")
load("@rules_cc//cc:cc_shared_library.bzl", _cc_shared_library = "cc_shared_library")
load("@rules_cc//cc:cc_test.bzl", _cc_test = "cc_test")

cc_binary = _cc_binary
cc_import = _cc_import
cc_library = _cc_library
cc_shared_library = _cc_shared_library
cc_test = _cc_test
