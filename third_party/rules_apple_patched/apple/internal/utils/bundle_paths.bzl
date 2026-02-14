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

"""Support functions bundle related paths operations."""

load(
    "@bazel_skylib//lib:paths.bzl",
    "paths",
)

def _farthest_parent(path, extension):
    """Returns the part of a path with the given extension closest to the root.

    For example, if `path` is `"foo/bar.bundle/baz.bundle"`, passing `".bundle"`
    as the extension will return `"foo/bar.bundle"`.

    Args:
      path: The path.
      extension: The extension of the directory to find.

    Returns:
      The portion of the path that ends in the given extension that is closest
      to the root of the path.
    """
    prefix, ext, _ = path.partition("." + extension)
    if ext:
        return prefix + ext

    fail("Expected path %r to contain %r, but it did not" % (
        path,
        "." + extension,
    ))

def _owner_relative_path(f):
    """Returns the portion of `f`'s path relative to its owner.

    Args:
      f: A file.

    Returns:
      The owner-relative path to the file.
    """
    if f.is_source:
        # Even though the docs says a File's `short_path` doesn't include the
        # root, Bazel special cases anything that is external and includes a
        # relative path (../) to the file. On the File's `owner` we can get the
        # `workspace_root` to try and line things up, but it is in the form of
        # "external/[name]". However the File's `path` does include the root and
        # leaves it in the "externa/" form.
        return paths.relativize(
            f.path,
            paths.join(f.owner.workspace_root, f.owner.package),
        )
    elif f.owner.workspace_root:
        # Just like the above comment but for generated files, the same mangling
        # happen in `short_path`, but since it is generated, the `path` includes
        # the extra output directories bazel makes. So pick off what bazel will do
        # to the `short_path` ("../"), and turn it into an "external/" so a
        # relative path from the owner can be calculated.
        workspace_root = f.owner.workspace_root
        short_path = f.short_path
        if (not workspace_root.startswith("external/") or
            not short_path.startswith("../")):
            fail(("Generated file in a different workspace with unexpected " +
                  "short_path (%s) and owner.workspace_root (%r).") % (
                short_path,
                workspace_root,
            ))
        return paths.relativize(
            "external" + short_path[2:],
            paths.join(f.owner.workspace_root, f.owner.package),
        )
    else:
        return paths.relativize(f.short_path, f.owner.package)

def _locale_for_path(resource_path):
    """Returns the detected locale for the given resource path."""
    if not resource_path:
        return None

    loc = resource_path.find(".lproj")
    if loc == -1:
        return None

    # If there was more after '.lproj', then it has to be a directory, otherwise
    # it was part of some other extension.
    if (loc + 6) > len(resource_path) and resource_path[loc + 6] != "/":
        return None

    locale_start = resource_path.rfind("/", 0, loc)
    if locale_start < 0:
        return resource_path[0:loc]

    return resource_path[locale_start + 1:loc]

# Define the loadable module that lists the exported symbols in this file.
bundle_paths = struct(
    farthest_parent = _farthest_parent,
    owner_relative_path = _owner_relative_path,
    locale_for_path = _locale_for_path,
)
