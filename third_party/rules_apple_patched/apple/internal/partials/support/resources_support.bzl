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

"""Support code for resource processing.

All methods in this file follow this convention:
  - The argument signature is a combination of; actions, apple_mac_toolchain_info, bundle_id, files,
    output_discriminator, parent_dir, platform_prerequisites, product_type, and/or rule_label. Only
    the arguments required for each resource action should be referenced directly by keyword in the
    argument signature and implementation. Arguments should not be referenced through kwargs. The
    presence of kwargs is only necessary to ignore unused keywords.
  - They all return a struct with the following optional fields:
      - files: A list of tuples with the following structure:
          - Processor location: The location type in the archive where these files should be placed.
          - Parent directory: The structured path on where the files should be placed, within the
            processor location.
          - Files: Depset of files to be placed in the location described, under the name described
            by their basenames.
      - infoplists: A list of files representing plist files that will be merged to compose the main
        bundle's Info.plist.
"""

load(
    "@bazel_skylib//lib:paths.bzl",
    "paths",
)
load(
    "//apple:utils.bzl",
    "group_files_by_directory",
)
load(
    "//apple/internal:experimental.bzl",
    "is_experimental_tree_artifact_enabled",
)
load(
    "//apple/internal:intermediates.bzl",
    "intermediates",
)
load(
    "//apple/internal:processor.bzl",
    "processor",
)
load(
    "//apple/internal:resource_actions.bzl",
    "resource_actions",
)

def _compile_datamodels(
        *,
        actions,
        datamodel_groups,
        label_name,
        output_discriminator,
        parent_dir,
        platform_prerequisites,
        swift_module,
        xctoolrunner):
    "Compiles datamodels into mom files."
    output_files = []
    processed_origins = {}
    module_name = swift_module or label_name
    for datamodel_path, files in datamodel_groups.items():
        input_files = files.to_list()

        datamodel_name = paths.replace_extension(paths.basename(datamodel_path), "")
        datamodel_parent = parent_dir
        if datamodel_path.endswith(".xcdatamodeld"):
            basename = datamodel_name + ".momd"
            output_file = intermediates.directory(
                actions = actions,
                target_name = label_name,
                output_discriminator = output_discriminator,
                dir_name = basename,
            )
            datamodel_parent = paths.join(datamodel_parent or "", basename)
        else:
            output_file = intermediates.file(
                actions = actions,
                target_name = label_name,
                output_discriminator = output_discriminator,
                file_name = datamodel_name + ".mom",
            )

        processed_origins[output_file.short_path] = [f.short_path for f in input_files]

        resource_actions.compile_datamodels(
            actions = actions,
            datamodel_path = datamodel_path,
            input_files = input_files,
            module_name = module_name,
            output_file = output_file,
            platform_prerequisites = platform_prerequisites,
            xctoolrunner = xctoolrunner,
        )
        output_files.append(
            (processor.location.resource, datamodel_parent, depset(direct = [output_file])),
        )

    return (output_files, processed_origins)

def _compile_mappingmodels(
        *,
        actions,
        label_name,
        mappingmodel_groups,
        output_discriminator,
        parent_dir,
        platform_prerequisites,
        xctoolrunner):
    """Compiles mapping models into cdm files."""
    output_files = []
    processed_origins = {}
    for mappingmodel_path, files in mappingmodel_groups.items():
        input_files = files.to_list()

        compiled_model_name = paths.replace_extension(paths.basename(mappingmodel_path), ".cdm")
        output_file = intermediates.file(
            actions = actions,
            target_name = label_name,
            output_discriminator = output_discriminator,
            file_name = paths.join(parent_dir or "", compiled_model_name),
        )
        processed_origins[output_file.short_path] = [f.short_path for f in input_files]

        resource_actions.compile_mappingmodel(
            actions = actions,
            input_files = input_files,
            mappingmodel_path = mappingmodel_path,
            output_file = output_file,
            platform_prerequisites = platform_prerequisites,
            xctoolrunner = xctoolrunner,
        )

        output_files.append(
            (processor.location.resource, parent_dir, depset(direct = [output_file])),
        )

    return (output_files, processed_origins)

def _asset_catalogs(
        *,
        actions,
        apple_mac_toolchain_info,
        bundle_id,
        files,
        output_discriminator,
        parent_dir,
        platform_prerequisites,
        primary_icon_name,
        product_type,
        rule_label,
        **_kwargs):
    """Processes asset catalog files."""
    processed_origins = {}

    all_asset_files = files.to_list()

    # A list of all known asset catalog types besides Icon Composer icons can be found at this link
    # within the Apple legacy documentation archive:
    # https://developer.apple.com/library/archive/documentation/Xcode/Reference/xcode_ref-Asset_Catalog_Format/AssetTypes.html#//apple_ref/doc/uid/TP40015170-CH30-SW1
    #
    # Check for empty asset catalogs; these will waste time and resources executing actool.
    contains_assets_to_compile = False
    for file in all_asset_files:
        # Skip directories and Contents.json files outside of .colorset folders.
        #
        # Contents.json files *inside* of .colorset folders are transformed into compiled asset
        # catalogs with color sets, and do not require any other files to generate these color sets.
        if (file.is_directory or (
            file.basename == "Contents.json" and not file.dirname.endswith(".colorset")
        )):
            continue
        contains_assets_to_compile = True
        break
    if not contains_assets_to_compile:
        # There is no other way to issue a warning, so print is the only way to message.
        # buildifier: disable=print
        print("""
WARNING: No assets to compile for {rule_label} even though an asset catalog (.xcassets directory) \
was declared. Skipping asset catalog compilation.
""".format(rule_label = str(rule_label)))
        return struct(files = [], infoplists = [])

    # Only merge the resulting plist for the top level bundle. For resource
    # bundles, skip generating the plist.
    assets_plist = None
    infoplists = []
    if not parent_dir:
        assets_plist_path = paths.join(parent_dir or "", "xcassets-info.plist")
        assets_plist = intermediates.file(
            actions = actions,
            target_name = rule_label.name,
            output_discriminator = output_discriminator,
            file_name = assets_plist_path,
        )
        infoplists.append(assets_plist)

    assets_dir = intermediates.directory(
        actions = actions,
        target_name = rule_label.name,
        output_discriminator = output_discriminator,
        dir_name = paths.join(parent_dir or "", "xcassets"),
    )

    # Separate alternate icons from regular assets
    alternate_icons = []
    asset_files = []
    for f in all_asset_files:
        if ".alticon/" in f.path:
            if f.extension != "png":
                fail("Alternate icons must be .png files packaged in a .alticon folder. Not: {}".format(f.path))

            # Process icon PNGs like any other PNG
            alticon_id = paths.basename(f.dirname)
            png_file = intermediates.file(
                actions = actions,
                file_name = paths.join("alticons", alticon_id, f.basename),
                output_discriminator = None,
                target_name = rule_label.name,
            )
            processed_origins[png_file.short_path] = f.short_path
            resource_actions.copy_png(
                actions = actions,
                input_file = f,
                output_file = png_file,
                platform_prerequisites = platform_prerequisites,
            )
            alternate_icons.append(png_file)
        else:
            asset_files.append(f)

    resource_actions.compile_asset_catalog(
        actions = actions,
        alternate_icons = alternate_icons,
        alticonstool = apple_mac_toolchain_info.alticonstool,
        asset_files = asset_files,
        bundle_id = bundle_id,
        output_dir = assets_dir,
        output_plist = assets_plist,
        platform_prerequisites = platform_prerequisites,
        primary_icon_name = primary_icon_name,
        product_type = product_type,
        rule_label = rule_label,
        xctoolrunner = apple_mac_toolchain_info.xctoolrunner,
    )

    asset_origin_short_paths = [f.short_path for f in asset_files]
    processed_origins[assets_dir.short_path] = asset_origin_short_paths
    if assets_plist:
        processed_origins[assets_plist.short_path] = asset_origin_short_paths

    return struct(
        files = [
            (
                processor.location.resource,
                parent_dir,
                depset(direct = [assets_dir] + alternate_icons),
            ),
        ],
        infoplists = infoplists,
        processed_origins = processed_origins,
    )

def _datamodels(
        *,
        actions,
        apple_mac_toolchain_info,
        files,
        output_discriminator,
        parent_dir,
        platform_prerequisites,
        rule_label,
        swift_module,
        **_kwargs):
    "Processes datamodel related files."
    datamodel_files = files.to_list()

    standalone_datamodels = []
    grouped_datamodels = []
    mappingmodels = []

    # Split the datamodels into whether they are inside an xcdatamodeld bundle or not.
    for datamodel in datamodel_files:
        datamodel_short_path = datamodel.short_path
        if ".xcmappingmodel/" in datamodel_short_path:
            mappingmodels.append(datamodel)
        elif ".xcdatamodeld/" in datamodel_short_path:
            grouped_datamodels.append(datamodel)
        else:
            standalone_datamodels.append(datamodel)

    # Create a map of highest-level datamodel bundle to the files it contains. Datamodels can be
    # present within standalone .xcdatamodel/ folders or in a versioned bundle, in which many
    # .xcdatamodel/ are contained inside an .xcdatamodeld/ bundle. .xcdatamodeld/ bundles are
    # processed altogether, while .xcdatamodel/ bundles are processed by themselves.
    datamodel_groups = group_files_by_directory(
        grouped_datamodels,
        ["xcdatamodeld"],
        attr = "datamodels",
    )
    datamodel_groups.update(group_files_by_directory(
        standalone_datamodels,
        ["xcdatamodel"],
        attr = "datamodels",
    ))
    mappingmodel_groups = group_files_by_directory(
        mappingmodels,
        ["xcmappingmodel"],
        attr = "resources",
    )

    output_files = []
    (
        compiled_datamodels,
        datamodels_processed_origins,
    ) = _compile_datamodels(
        actions = actions,
        datamodel_groups = datamodel_groups,
        label_name = rule_label.name,
        output_discriminator = output_discriminator,
        parent_dir = parent_dir,
        platform_prerequisites = platform_prerequisites,
        swift_module = swift_module,
        xctoolrunner = apple_mac_toolchain_info.xctoolrunner,
    )
    output_files.extend(compiled_datamodels)
    (
        compiled_mappingmodels,
        mappingmodels_processed_origins,
    ) = _compile_mappingmodels(
        actions = actions,
        label_name = rule_label.name,
        output_discriminator = output_discriminator,
        parent_dir = parent_dir,
        mappingmodel_groups = mappingmodel_groups,
        platform_prerequisites = platform_prerequisites,
        xctoolrunner = apple_mac_toolchain_info.xctoolrunner,
    )
    output_files.extend(compiled_mappingmodels)

    return struct(
        files = output_files,
        processed_origins = (
            datamodels_processed_origins | mappingmodels_processed_origins
        ),
    )

def _infoplists(
        *,
        actions,
        apple_mac_toolchain_info,
        bundle_id,
        files,
        output_discriminator,
        parent_dir,
        platform_prerequisites,
        rule_label,
        **_kwargs):
    """Processes infoplists.

    If parent_dir is not empty, the files will be treated as resource bundle infoplists and are
    merged into one. If parent_dir is empty (or None), the files are be treated as root level
    infoplist and returned to be processed along with other root plists (e.g. asset catalog
    processing returns a plist that needs to be merged into the root).

    Args:
        actions: The actions provider from `ctx.actions`.
        apple_mac_toolchain_info: `struct` of tools from the shared Apple toolchain.
        bundle_id: The bundle ID to use when templating plist files.
        files: The infoplist files to process.
        output_discriminator: A string to differentiate between different target intermediate files
            or `None`.
        parent_dir: The path under which the merged Info.plist should be placed for resource bundles.
        platform_prerequisites: Struct containing information on the platform being targeted.
        rule_label: The label of the target being analyzed.
        **_kwargs: Extra parameters forwarded to this support macro.

    Returns:
        A struct containing a `files` field with tuples as described in processor.bzl, and an
        `infoplists` field with the plists that need to be merged for the root Info.plist
    """
    processed_origins = {}
    if parent_dir:
        input_files = files.to_list()
        out_plist = intermediates.file(
            actions = actions,
            target_name = rule_label.name,
            output_discriminator = output_discriminator,
            file_name = paths.join(parent_dir, "Info.plist"),
        )
        processed_origins[out_plist.short_path] = [f.short_path for f in input_files]
        resource_actions.merge_resource_infoplists(
            actions = actions,
            bundle_id = bundle_id,
            bundle_name_with_extension = paths.basename(parent_dir),
            input_files = input_files,
            output_discriminator = output_discriminator,
            output_plist = out_plist,
            platform_prerequisites = platform_prerequisites,
            plisttool = apple_mac_toolchain_info.plisttool,
            rule_label = rule_label,
        )
        return struct(
            files = [
                (processor.location.resource, parent_dir, depset(direct = [out_plist])),
            ],
            processed_origins = processed_origins,
        )
    else:
        return struct(
            files = [],
            infoplists = files.to_list(),
            processed_origins = processed_origins,
        )

def _metals(
        *,
        actions,
        rule_label,
        parent_dir,
        platform_prerequisites,
        files,
        output_filename = "default.metallib",
        **_kwargs):
    """Processes metal files.

    The metal files will be compiled into a Metal library named `default.metallib`.

    Args:
        actions: The actions provider from `ctx.actions`.
        rule_label: The label of the target being analyzed.
        parent_dir: The path under which the library should be placed.
        platform_prerequisites: Struct containing information on the platform being targeted.
        files: The metal files to process.
        output_filename: The output .metallib filename.
        **_kwargs: Ignored

    Returns:
        A struct containing a `files` field with tuples as described in processor.bzl.
    """
    input_files = files.to_list()
    processed_origins = {}
    metallib_path = paths.join(parent_dir or "", output_filename)
    metallib_file = intermediates.file(
        actions = actions,
        file_name = metallib_path,
        output_discriminator = None,
        target_name = rule_label.name,
    )
    processed_origins[metallib_file.short_path] = [f.short_path for f in input_files]
    resource_actions.compile_metals(
        actions = actions,
        input_files = input_files,
        output_file = metallib_file,
        platform_prerequisites = platform_prerequisites,
    )

    return struct(
        files = [(
            processor.location.resource,
            parent_dir,
            depset(direct = [metallib_file]),
        )],
        processed_origins = processed_origins,
    )

def _mlmodels(
        *,
        actions,
        apple_mac_toolchain_info,
        files,
        output_discriminator,
        parent_dir,
        platform_prerequisites,
        rule_label,
        **_kwargs):
    """Processes mlmodel files."""

    mlmodel_bundles = []
    infoplists = []
    processed_origins = {}
    for file in files.to_list():
        basename = file.basename

        output_bundle = intermediates.directory(
            actions = actions,
            target_name = rule_label.name,
            output_discriminator = output_discriminator,
            dir_name = paths.join(parent_dir or "", paths.replace_extension(basename, ".mlmodelc")),
        )
        processed_origins[output_bundle.short_path] = [file.short_path]

        output_plist = intermediates.file(
            actions = actions,
            target_name = rule_label.name,
            output_discriminator = output_discriminator,
            file_name = paths.join(parent_dir or "", paths.replace_extension(basename, ".plist")),
        )
        processed_origins[output_plist.short_path] = [file.short_path]

        resource_actions.compile_mlmodel(
            actions = actions,
            input_file = file,
            output_bundle = output_bundle,
            output_plist = output_plist,
            platform_prerequisites = platform_prerequisites,
            xctoolrunner = apple_mac_toolchain_info.xctoolrunner,
        )

        mlmodel_bundles.append(
            (
                processor.location.resource,
                paths.join(parent_dir or "", output_bundle.basename),
                depset(direct = [output_bundle]),
            ),
        )
        infoplists.append(output_plist)

    return struct(
        files = mlmodel_bundles,
        infoplists = infoplists,
        processed_origins = processed_origins,
    )

def _plists_and_strings(
        *,
        actions,
        files,
        force_binary = False,
        output_discriminator,
        parent_dir,
        platform_prerequisites,
        rule_label,
        **_kwargs):
    """Processes plists and string files.

    If compilation mode is `opt`, or if force_binary is True, the plist files will be compiled into
    binary to make them smaller. Otherwise, they will be copied verbatim to avoid the extra
    processing time.

    Args:
        actions: The actions provider from `ctx.actions`.
        files: The plist or string files to process.
        force_binary: If true, files will be converted to binary independently of the compilation
            mode.
        output_discriminator: A string to differentiate between different target intermediate files
            or `None`.
        parent_dir: The path under which the files should be placed.
        platform_prerequisites: Struct containing information on the platform being targeted.
        rule_label: The label of the target being analyzed.
        **_kwargs: Extra parameters forwarded to this support macro.

    Returns:
        A struct containing a `files` field with tuples as described in processor.bzl.
    """

    # If this is not an optimized build, and force_compile is False, then just copy the files
    if not force_binary and platform_prerequisites.config_vars["COMPILATION_MODE"] != "opt":
        return _noop(
            parent_dir = parent_dir,
            files = files,
        )

    plist_files = []
    processed_origins = {}
    for file in files.to_list():
        plist_file = intermediates.file(
            actions = actions,
            target_name = rule_label.name,
            output_discriminator = output_discriminator,
            file_name = paths.join(parent_dir or "", file.basename),
        )
        processed_origins[plist_file.short_path] = [file.short_path]
        resource_actions.compile_plist(
            actions = actions,
            input_file = file,
            output_file = plist_file,
            platform_prerequisites = platform_prerequisites,
        )
        plist_files.append(plist_file)

    return struct(
        files = [
            (processor.location.resource, parent_dir, depset(direct = plist_files)),
        ],
        processed_origins = processed_origins,
    )

def _pngs(
        *,
        actions,
        files,
        output_discriminator,
        parent_dir,
        platform_prerequisites,
        rule_label,
        **_kwargs):
    """Register PNG processing actions.

    The PNG files will be copied using `pngcopy` to make them smaller.

    Args:
        actions: The actions provider from `ctx.actions`.
        files: The PNG files to process.
        output_discriminator: A string to differentiate between different target intermediate files
            or `None`.
        parent_dir: The path under which the images should be placed.
        platform_prerequisites: Struct containing information on the platform being targeted.
        rule_label: The label of the target being analyzed.
        **_kwargs: Extra parameters forwarded to this support macro.

    Returns:
        A struct containing a `files` field with tuples as described in processor.bzl.
    """
    png_files = []
    processed_origins = {}
    for file in files.to_list():
        png_path = paths.join(parent_dir or "", file.basename)
        png_file = intermediates.file(
            actions = actions,
            target_name = rule_label.name,
            output_discriminator = output_discriminator,
            file_name = png_path,
        )
        processed_origins[png_file.short_path] = [file.short_path]
        resource_actions.copy_png(
            actions = actions,
            input_file = file,
            output_file = png_file,
            platform_prerequisites = platform_prerequisites,
        )
        png_files.append(png_file)

    return struct(
        files = [
            (processor.location.resource, parent_dir, depset(direct = png_files)),
        ],
        processed_origins = processed_origins,
    )

def _storyboards(
        *,
        actions,
        apple_mac_toolchain_info,
        files,
        output_discriminator,
        parent_dir,
        platform_prerequisites,
        rule_label,
        swift_module,
        **_kwargs):
    """Processes storyboard files."""
    swift_module = swift_module or rule_label.name

    # First, compile all the storyboard files and collect the output folders.
    compiled_storyboardcs = []
    origin_short_paths = []
    for storyboard in files.to_list():
        storyboardc_path = paths.join(
            # We append something at the end of the name to avoid having X.lproj names in the path.
            # It seems like ibtool will output with different paths depending on whether it is part
            # of a localization bundle. By appending something at the end, we avoid having X.lproj
            # directory names in ibtool's arguments. This is not the case from the storyboard
            # linking action, so we do not change the path there.
            (parent_dir or "") + "_storyboardc",
            paths.replace_extension(storyboard.basename, ".storyboardc"),
        )
        storyboardc_dir = intermediates.directory(
            actions = actions,
            target_name = rule_label.name,
            output_discriminator = output_discriminator,
            dir_name = storyboardc_path,
        )
        origin_short_paths.append(storyboard.short_path)
        resource_actions.compile_storyboard(
            actions = actions,
            input_file = storyboard,
            output_dir = storyboardc_dir,
            platform_prerequisites = platform_prerequisites,
            swift_module = swift_module,
            xctoolrunner = apple_mac_toolchain_info.xctoolrunner,
        )
        compiled_storyboardcs.append(storyboardc_dir)

    # Then link all the output folders into one folder, which will then be the
    # folder to be bundled.
    linked_storyboard_dir = intermediates.directory(
        actions = actions,
        target_name = rule_label.name,
        output_discriminator = output_discriminator,
        dir_name = paths.join("storyboards", parent_dir or "", swift_module or ""),
    )
    resource_actions.link_storyboards(
        actions = actions,
        output_dir = linked_storyboard_dir,
        platform_prerequisites = platform_prerequisites,
        storyboardc_dirs = compiled_storyboardcs,
        xctoolrunner = apple_mac_toolchain_info.xctoolrunner,
    )
    return struct(
        files = [
            (processor.location.resource, parent_dir, depset(direct = [linked_storyboard_dir])),
        ],
        processed_origins = {
            linked_storyboard_dir.short_path: origin_short_paths,
        },
    )

def _texture_atlases(
        *,
        actions,
        files,
        output_discriminator,
        parent_dir,
        platform_prerequisites,
        rule_label,
        **_kwargs):
    """Processes texture atlas files."""
    atlases_groups = group_files_by_directory(
        files.to_list(),
        ["atlas"],
        attr = "texture_atlas",
    )

    atlasc_files = []
    processed_origins = {}
    for atlas_path, files in atlases_groups.items():
        input_files = files.to_list()
        atlasc_path = paths.join(
            parent_dir or "",
            paths.replace_extension(paths.basename(atlas_path), ".atlasc"),
        )
        atlasc_dir = intermediates.directory(
            actions = actions,
            target_name = rule_label.name,
            output_discriminator = output_discriminator,
            dir_name = atlasc_path,
        )
        processed_origins[atlasc_dir.short_path] = [f.short_path for f in input_files]
        resource_actions.compile_texture_atlas(
            actions = actions,
            input_files = input_files,
            input_path = atlas_path,
            output_dir = atlasc_dir,
            platform_prerequisites = platform_prerequisites,
        )
        atlasc_files.append(atlasc_dir)

    return struct(
        files = [
            (processor.location.resource, parent_dir, depset(direct = atlasc_files)),
        ],
        processed_origins = processed_origins,
    )

def _xibs(
        *,
        actions,
        apple_mac_toolchain_info,
        files,
        output_discriminator,
        parent_dir,
        platform_prerequisites,
        rule_label,
        swift_module,
        **_kwargs):
    """Processes Xib files."""
    swift_module = swift_module or rule_label.name
    nib_files = []
    processed_origins = {}
    for file in files.to_list():
        basename = paths.replace_extension(file.basename, "")
        out_path = paths.join("nibs", parent_dir or "", basename)
        out_dir = intermediates.directory(
            actions = actions,
            target_name = rule_label.name,
            output_discriminator = output_discriminator,
            dir_name = out_path,
        )
        processed_origins[out_dir.short_path] = [file.short_path]
        resource_actions.compile_xib(
            actions = actions,
            input_file = file,
            output_dir = out_dir,
            platform_prerequisites = platform_prerequisites,
            swift_module = swift_module,
            xctoolrunner = apple_mac_toolchain_info.xctoolrunner,
        )
        nib_files.append(out_dir)

    return struct(
        files = [(processor.location.resource, parent_dir, depset(direct = nib_files))],
        processed_origins = processed_origins,
    )

def _xcstrings(
        *,
        actions,
        apple_mac_toolchain_info,
        files,
        output_discriminator,
        parent_dir,
        platform_prerequisites,
        rule_label,
        **_kwargs):
    """Process xcstrings files."""
    lproj_files = []
    processed_origins = {}
    for file in files.to_list():
        basename = paths.replace_extension(file.basename, "")
        out_path = paths.join("xcstrings", parent_dir or "", basename)
        out_dir = intermediates.directory(
            actions = actions,
            target_name = rule_label.name,
            output_discriminator = output_discriminator,
            dir_name = out_path,
        )
        processed_origins[out_dir.short_path] = [file.short_path]
        resource_actions.compile_xcstrings(
            actions = actions,
            input_file = file,
            output_dir = out_dir,
            platform_prerequisites = platform_prerequisites,
            xctoolrunner = apple_mac_toolchain_info.xctoolrunner,
        )
        lproj_files.append(out_dir)

    return struct(
        files = [(processor.location.resource, parent_dir, depset(lproj_files))],
        processed_origins = processed_origins,
    )

def _noop(
        *,
        parent_dir,
        files,
        **_kwargs):
    """Registers files to be bundled as is."""
    processed_origins = {}
    for file in files.to_list():
        processed_origins[file.short_path] = [file.short_path]
    return struct(
        files = [(processor.location.resource, parent_dir, files)],
        processed_origins = processed_origins,
    )

def _apple_bundle(bundle_type):
    """Returns a function to register bundling of Apple bundles at their appropriate location.

    Args:
        bundle_type: The Apple bundle type to bundle for.

    Returns:
        A function to register bundling of an Apple bundle.
    """
    if not hasattr(processor.location, bundle_type):
        fail("Bundle type location not supported: ", bundle_type)

    def _bundle_at_location(*, files, platform_prerequisites, **_kwargs):
        location = getattr(processor.location, bundle_type)

        # If tree artifacts are enabled, iterate each bundle and set the bundle name
        # as the parent directory. Otherwise, let bundletool unzip the bundle as is.
        if is_experimental_tree_artifact_enabled(config_vars = platform_prerequisites.config_vars):
            bundle_files = []
            for bundle in files.to_list():
                # TODO(b/271899726): Prepend parent_dir if embeddeding frameworks inside a resource bundle is allowed.
                basename = paths.basename(bundle.short_path)
                bundle_files.append((location, basename, depset([bundle])))
            return struct(files = bundle_files)
        else:
            return struct(archives = [(location, None, files)])

    return _bundle_at_location

# Map of resource provider fields to a tuple that contains the method to use
# to process those resources and a boolean indicating whether the Swift
# module is required for that processing.
PROVIDER_TO_FIELD_ACTION = {
    "asset_catalogs": (_asset_catalogs, False),
    "datamodels": (_datamodels, True),
    "framework": (_apple_bundle(processor.location.framework), False),
    "infoplists": (_infoplists, False),
    "metals": (_metals, False),
    "mlmodels": (_mlmodels, False),
    "plists": (_plists_and_strings, False),
    "pngs": (_pngs, False),
    "processed": (_noop, False),
    "storyboards": (_storyboards, True),
    "strings": (_plists_and_strings, False),
    "texture_atlases": (_texture_atlases, False),
    "unprocessed": (_noop, False),
    "xcstrings": (_xcstrings, False),
    "xibs": (_xibs, True),
}
