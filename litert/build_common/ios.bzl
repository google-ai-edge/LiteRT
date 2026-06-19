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

"""LiteRT Build Configurations for iOS"""

def strip_common_include_path_prefix(name, hdr_labels, prefix = ""):
    """Create modified header files with the inclusion path prefixes removed.

    Args:
      name: The name to be used as a prefix to the generated genrules.
      hdr_labels: List of header labels to strip out the include path. Each
          label must end with a colon followed by the header file name.
      prefix: Optional prefix path to prepend to the final inclusion path.
    """

    outs = []
    for hdr_label in hdr_labels:
        hdr_filename = hdr_label.split(":")[-1]
        hdr_basename = hdr_filename.split(".")[0]

        native.genrule(
            name = "{}_{}".format(name, hdr_basename),
            srcs = [hdr_label],
            outs = [hdr_filename],
            cmd = """
            sed -E 's|#include ".*/([^/]+\\.h)"|#include "{}\\1"|g'\
            "$(location {})"\
            > "$@"
            """.format(prefix, hdr_label),
        )
        outs.append(hdr_filename)

    native.filegroup(
        name = name,
        srcs = outs,
    )
