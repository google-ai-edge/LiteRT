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

"""
Helpers to build artifacts for releasing vendor sdks.
"""

def sdk_release_tar(
        name,
        srcs,
        strip_prefix = "",
        description = ""):
    """Creates a tar.gz file from the given sources.

    Args:
      name: The name of the rule.
      srcs: The list of files to include in the tar.gz file.
      strip_prefix: The prefix to remove from the file paths in the tar.gz file.
      description: The description of the release.
    """
    if strip_prefix and not strip_prefix.endswith("/"):
        strip_prefix = strip_prefix + "/"
    stamp = r"$$(date +%Y_%m_%d_%H_%M_%S)"
    cmd = """
    TARGETS=""
    for SRC in $(SRCS); do
      TARGETS+="$${SRC#%s} "
    done
    mkdir $@.tmp
    echo %s >> $@.tmp/README.txt
    echo %s >> $@.tmp/README.txt
    tar -czhf $@ $@.tmp/README.txt -C %s $$TARGETS --transform "s|$@.tmp||g"
    rm -rf $@.tmp
    """ % (strip_prefix, stamp, description, strip_prefix if strip_prefix else "./")
    native.genrule(
        name = name,
        srcs = srcs,
        outs = [name + ".tar.gz"],
        cmd = cmd,
    )
