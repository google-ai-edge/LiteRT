# Copyright 2018 The Bazel Authors. All rights reserved.
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

"""Proxy file for referencing resource processing actions."""

load(
    "//apple/internal/resource_actions:actool.bzl",
    _compile_asset_catalog = "compile_asset_catalog",
)
load(
    "//apple/internal/resource_actions:datamodel.bzl",
    _compile_datamodels = "compile_datamodels",
    _compile_mappingmodel = "compile_mappingmodel",
    _generate_datamodels = "generate_datamodels",
)
load(
    "//apple/internal/resource_actions:ibtool.bzl",
    _compile_storyboard = "compile_storyboard",
    _compile_xib = "compile_xib",
    _link_storyboards = "link_storyboards",
)
load(
    "//apple/internal/resource_actions:intent.bzl",
    _generate_intent_classes_sources = "generate_intent_classes_sources",
)
load(
    "//apple/internal/resource_actions:metals.bzl",
    _compile_metals = "compile_metals",
)
load(
    "//apple/internal/resource_actions:mlmodel.bzl",
    _compile_mlmodel = "compile_mlmodel",
    _generate_mlmodel_sources = "generate_mlmodel_sources",
)
load(
    "//apple/internal/resource_actions:plist.bzl",
    _compile_plist = "compile_plist",
    _merge_resource_infoplists = "merge_resource_infoplists",
    _merge_root_infoplists = "merge_root_infoplists",
    _plisttool_action = "plisttool_action",
)
load(
    "//apple/internal/resource_actions:png.bzl",
    _copy_png = "copy_png",
)
load(
    "//apple/internal/resource_actions:texture_atlas.bzl",
    _compile_texture_atlas = "compile_texture_atlas",
)
load(
    "//apple/internal/resource_actions:xcstrings.bzl",
    _compile_xcstrings = "compile_xcstrings",
)

resource_actions = struct(
    compile_asset_catalog = _compile_asset_catalog,
    compile_datamodels = _compile_datamodels,
    compile_mappingmodel = _compile_mappingmodel,
    compile_metals = _compile_metals,
    compile_mlmodel = _compile_mlmodel,
    compile_plist = _compile_plist,
    compile_storyboard = _compile_storyboard,
    compile_texture_atlas = _compile_texture_atlas,
    compile_xcstrings = _compile_xcstrings,
    compile_xib = _compile_xib,
    copy_png = _copy_png,
    generate_datamodels = _generate_datamodels,
    generate_intent_classes_sources = _generate_intent_classes_sources,
    generate_mlmodel_sources = _generate_mlmodel_sources,
    link_storyboards = _link_storyboards,
    merge_resource_infoplists = _merge_resource_infoplists,
    merge_root_infoplists = _merge_root_infoplists,
    plisttool_action = _plisttool_action,
)
