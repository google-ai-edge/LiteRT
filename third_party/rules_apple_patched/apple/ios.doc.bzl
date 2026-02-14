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

"""# Build rules for iOS"""

# Re-export original rules rather than their wrapper macros
# so that stardoc documents the rule attributes, not an opaque
# **kwargs argument.
load(
    "//apple/internal:ios_rules.bzl",
    _ios_app_clip = "ios_app_clip",
    _ios_application = "ios_application",
    _ios_dynamic_framework = "ios_dynamic_framework",
    _ios_extension = "ios_extension",
    _ios_framework = "ios_framework",
    _ios_imessage_application = "ios_imessage_application",
    _ios_imessage_extension = "ios_imessage_extension",
    _ios_static_framework = "ios_static_framework",
    _ios_sticker_pack_extension = "ios_sticker_pack_extension",
)
load(
    "//apple/internal/testing:ios_rules.bzl",
    _ios_ui_test = "ios_ui_test",
    _ios_unit_test = "ios_unit_test",
)
load(
    "//apple/testing/default_runner:ios_test_runner.bzl",
    _ios_test_runner = "ios_test_runner",
)
load(
    "//apple/testing/default_runner:ios_xctestrun_runner.bzl",
    _ios_xctestrun_runner = "ios_xctestrun_runner",
)
load(
    ":ios.bzl",
    _ios_build_test = "ios_build_test",
    _ios_ui_test_suite = "ios_ui_test_suite",
    _ios_unit_test_suite = "ios_unit_test_suite",
)

ios_app_clip = _ios_app_clip
ios_application = _ios_application
ios_build_test = _ios_build_test
ios_dynamic_framework = _ios_dynamic_framework
ios_extension = _ios_extension
ios_framework = _ios_framework
ios_imessage_application = _ios_imessage_application
ios_imessage_extension = _ios_imessage_extension
ios_static_framework = _ios_static_framework
ios_sticker_pack_extension = _ios_sticker_pack_extension
ios_test_runner = _ios_test_runner
ios_ui_test = _ios_ui_test
ios_ui_test_suite = _ios_ui_test_suite
ios_unit_test = _ios_unit_test
ios_unit_test_suite = _ios_unit_test_suite
ios_xctestrun_runner = _ios_xctestrun_runner
