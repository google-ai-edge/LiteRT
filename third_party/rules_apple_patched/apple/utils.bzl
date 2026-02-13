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

"""Utility functions for working with strings, lists, and files in Starlark."""

def full_label(lbl):
    """Converts a label to full format, e.g. //a/b/c -> //a/b/c:c.

    If the label is already in full format, it returns it as it is, otherwise
    appends the folder name as the target name.

    Args:
      lbl: The label to convert to full format.

    Returns:
      The label in full format, or the original input if it was already in full
      format.
    """
    if lbl.find(":") != -1:
        return lbl
    target_name = lbl.rpartition("/")[-1]
    return lbl + ":" + target_name

def group_files_by_directory(files, extensions, attr):
    """Groups files based on their containing directories.

    This function examines each file in |files| and looks for a containing
    directory with the given extension. It then returns a dictionary that maps
    the directory names to the files they contain.

    For example, if you had the following files:
      - some/path/foo.images/bar.png
      - some/path/foo.images/baz.png
      - some/path/quux.images/blorp.png

    Then passing the extension "images" to this function would return:
      {
          "some/path/foo.images": depset([
              "some/path/foo.images/bar.png",
              "some/path/foo.images/baz.png"
          ]),
          "some/path/quux.images": depset([
              "some/path/quux.images/blorp.png"
          ])
      }

    If an input file does not have a containing directory with the given
    extension, the build will fail.

    Args:
      files: An iterable of File objects.
      extensions: The list of extensions of the containing directories to return.
          The extensions should NOT include the leading dot.
      attr: The attribute to associate with the build failure if the list of
          files has an element that is not in a directory with the given
          extension.
    Returns:
      A dictionary whose keys are directories with the given extension and their
      values are the sets of files within them.
    """
    grouped_files = {}
    paths_not_matched = {}

    ext_info = [(".%s" % e, len(e) + 1) for e in extensions]

    for f in files:
        path = f.path

        not_matched = True
        for search_string, search_string_len in ext_info:
            # Make sure the matched string either has a '/' after it, or occurs at
            # the end of the string (this lets us match directories without requiring
            # a trailing slash but prevents matching something like '.xcdatamodeld'
            # when passing 'xcdatamodel'). The ordering of these checks is also
            # important, to ensure that we can handle cases that occur when working
            # with common Apple file structures, like passing 'xcdatamodel' and
            # correctly parsing paths matching 'foo.xcdatamodeld/bar.xcdatamodel/...'.
            after_index = -1
            index_with_slash = path.find(search_string + "/")
            if index_with_slash != -1:
                after_index = index_with_slash + search_string_len
            else:
                index_without_slash = path.find(search_string)
                after_index = index_without_slash + search_string_len

                # If the search string wasn't at the end of the string, it must have a
                # non-slash character after it (because we already checked the slash case
                # above), so eliminate it.
                if after_index != len(path):
                    after_index = -1

            if after_index != -1:
                not_matched = False
                container = path[:after_index]
                contained_files = grouped_files.setdefault(container, [])
                contained_files.append(f)

                # No need to check other extensions
                break

        if not_matched:
            paths_not_matched[path] = True

    if len(paths_not_matched):
        formatted_files = "[\n  %s\n]" % ",\n  ".join(paths_not_matched.keys())
        fail("Expected only files inside directories named with the extensions " +
             "%r, but found: %s" % (extensions, formatted_files), attr)

    return {k: depset(v) for k, v in grouped_files.items()}
