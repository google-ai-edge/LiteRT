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
"""A tool to find all Clang runtime libraries linked into a Mach-O binary.

Some Clang features, such as Address Sanitizer, require a runtime library to be
packaged into the app's bundle. It's not always possible to know which libraries
will be needed at analysis time. This script scans the linked binary for
references to such libraries, and copies them into a ZIP archive that can then
be merged into an IPA product.
"""

import os
import re
import subprocess
import sys
import zipfile

def normalize_clang_lip_path(path, local_developer_dir):
  return re.sub(".*\.app/Contents/Developer",
                local_developer_dir,
                path,
                count=1)


class ClangRuntimeToolError(RuntimeError):
  """Raised for all errors.

  Custom RuntimeError used to allow catching (and logging) just the
  ClangRuntimeTool errors.
  """

  def __init__(self, msg):
    """Initializes an error with the given message.

    Args:
      msg: The message for the error.
    """
    RuntimeError.__init__(self, msg)


class ClangRuntimeTool(object):
  """Implements the clang runtime tool."""

  def __init__(self, binary_path, output_zip_path):
    """Initializes ClangRuntimeTool.

    Args:
      binary_path: The path to the binary to scan.
      output_zip_path: The path to the output zip file.
    """

    self._binary_path = binary_path
    self._output_zip_path = output_zip_path

  def _get_xcode_clang_path_and_clang_libs(self, objdump_output):
    """Returns the path to the clang directory inside of Xcode.

    Each version of Xcode comes with clang packaged under a versioned directory
    that is unknown in advanced. In order to copy the runtime libraries, we need
    to figure out the path inside of the Xcode.app bundle that contains clang
    runtime. The path is written into the Mach-O header by the clang linker as
    a RPATH commmand:

      Load command 24
            cmd LC_RPATH
        cmdsize 136
          path /Applications/Xcode.app/.../lib/clang/X.Y.Z/lib/darwin (offset
          12)

    This method uses a simple heuristic to find the Xcode path amongst all RPATH
    entries on the given header. While not ideal, this approach lets us support
    any version of Xcode, installed into any location on the system.

    Args:
      header: A Mach-O header object to parse.

    Returns:
      A tuple with the first element as a string representing the path to the
      clang's lib directory, or `None` if one cannot be found and the second
      element as a set of library names.
    """
    found_rpath = None
    libs = set()
    for index, line in enumerate(objdump_output):
      if line.endswith(" LC_RPATH"):
        rpath_line = objdump_output[index + 2].strip()
        rpath_segments = rpath_line.split(" ")
        if len(rpath_segments) != 4:
          raise ClangRuntimeToolError("Unexpected objdump format.")
        rpath = rpath_segments[1]
        if not rpath.startswith("@") and "lib/clang" in rpath:
          found_rpath = rpath
      elif line.endswith(" LC_LOAD_DYLIB"):
        # TODO(b/249129510): Make sure we have test coverage for this case when
        # LC_RPATH is absent from the objdump output.
        library_line = objdump_output[index + 2].strip()
        library_segments = library_line.split(" ")
        if len(library_segments) != 4:
          raise ClangRuntimeToolError("Unexpected objdump format.")
        library = library_segments[1]
        if library.startswith("@rpath/libclang_rt"):
          libs.add(library[len("@rpath/"):])

    return found_rpath, libs

  def run(self):
    objdump_args = [
        "xcrun", "llvm-objdump", "--macho", "--private-headers",
        "--non-verbose", binary_path
    ]
    objdump_output = subprocess.check_output(objdump_args,
                                             encoding="utf8",
                                             errors="replace")
    objdump_output = [x.strip() for x in objdump_output.splitlines()]
    clang_lib_path, clang_libraries = self._get_xcode_clang_path_and_clang_libs(
        objdump_output)
    if not clang_lib_path:
      raise ClangRuntimeToolError("Could not find clang library path.")

    if not clang_libraries:
      raise ClangRuntimeToolError(
          "Could not find any clang runtime libraries to package."
          "This is likely a configuration error")
    if "DEVELOPER_DIR" in os.environ:
      clang_lib_path = normalize_clang_lip_path(clang_lib_path, os.environ["DEVELOPER_DIR"])

    with zipfile.ZipFile(out_path, "w") as out_zip:
      for lib in clang_libraries:
        full_path = os.path.join(clang_lib_path, lib)
        if os.path.exists(full_path):
          out_zip.write(full_path, arcname=lib)
        else:
          raise ClangRuntimeToolError("Could not read library at %s." %
                                      full_path)


if __name__ == "__main__":
  binary_path = sys.argv[1]
  out_path = sys.argv[2]

  tool = ClangRuntimeTool(binary_path, out_path)
  try:
    tool.run()
  except ClangRuntimeToolError as e:
    # Log tools errors cleanly for build output.
    sys.stderr.write("ERROR: %s\n" % e)
    sys.exit(1)
