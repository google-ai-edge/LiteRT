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

"""File-system bundling logic for Apple bundles.

THIS IS AN EXPERIMENTAL REWRITE OF THE BUNDLER that combines the bundling,
post-processing, and signing steps into a single action. It also omits the
archiving step.

The bundler takes a set of files and merges them into a bundle on the file
system, then signs the bundle.

This script takes a single argument that points to a file containing the JSON
representation of a "control" structure, which makes it easier to pass in
complex structured data. This control structure is a dictionary with the
following keys:

  bundle_merge_files: A list of dictionaries representing files to be merged
      into the bundle. Each dictionary contains the following fields: "src", the
      path of the file to be added to the bundle; "dest", the path inside the
      bundle where the file should live, including its filename (which lets the
      name be changed, if desired); and "executable", a Boolean value indicating
      whether or not the executable bit should be set on the file. If
      `executable` is omitted, False is used.
  bundle_merge_zips: A list of dictionaries representing ZIP archives whose
      contents should be merged into the bundle. Each dictionary contains two
      fields: "src", the path of the archive whose contents should be merged
      into the bundle; and "dest", the path inside the bundle where the ZIPs
      contents should be placed.
  code_signing_commands: An optional list of shell commands that should be
      executed to sign the bundle.
  output: The path to the directory (which will be created/cleared) that will
      represent the complete bundle.
  post_processor: The optional path to an executable that will be run after the
      bundle is complete but before it is signed.
"""

import errno
import filecmp
import json
import os
import shlex
import shutil
import sys
import zipfile
from ctypes import CDLL, c_char_p, c_int, get_errno

_CLONEFILE = None
_USE_CLONEFILE = sys.platform == "darwin"
def _load_clonefile():
  global _CLONEFILE
  if _CLONEFILE:
    return _CLONEFILE

  system = CDLL('/usr/lib/libSystem.dylib', use_errno=True)
  _CLONEFILE = system.clonefile
  _CLONEFILE.argtypes = [c_char_p, c_char_p, c_int] # src, dest, flags
  _CLONEFILE.restype = c_int  # 0 on success
  return _CLONEFILE

BUNDLE_CONFLICT_MSG_TEMPLATE = (
    'Cannot place two files at the same location %r in the bundle')

CODE_SIGN_ERROR_MSG_TEMPLATE = 'Code signing failed with exit code %d'

POST_PROCESSOR_ERROR_MSG_TEMPLATE = 'Post processor failed with exit code %d'

class BundleConflictError(ValueError):
  """Raised when two different files would be bundled in the same location."""

  def __init__(self, dest):
    """Initializes an error with the given key and values.

    Args:
      dest: The destination path inside the bundle.
    """
    self.dest = dest
    ValueError.__init__(self, BUNDLE_CONFLICT_MSG_TEMPLATE % dest)


class CodeSignError(EnvironmentError):
  """Raised if the code signing commands fail."""

  def __init__(self, exit_code):
    self.exit_code = exit_code
    EnvironmentError.__init__(self, CODE_SIGN_ERROR_MSG_TEMPLATE % exit_code)


class PostProcessorError(EnvironmentError):
  """Raised if the post processing tool fails."""

  def __init__(self, exit_code):
    self.exit_code = exit_code
    EnvironmentError.__init__(self,
                              POST_PROCESSOR_ERROR_MSG_TEMPLATE % exit_code)


class Bundler(object):
  """Implements the core functionality of the bundler."""

  def __init__(self, control):
    """Initializes Bundler with the given control options.

    Args:
      control: The dictionary of options used to control the tool. Please see
          the moduledoc for a description of the format of this dictionary.
    """
    self._control = control

  def run(self):
    """Performs the operations requested by the control struct."""
    output_path = self._control.get('output')
    if not output_path:
      raise ValueError('No output file specified.')

    bundle_merge_files = self._control.get('bundle_merge_files', [])
    bundle_merge_zips = self._control.get('bundle_merge_zips', [])

    # Clear the output directory if it already exists.
    if os.path.exists(output_path):
      shutil.rmtree(output_path)
    self._makedirs_safely(output_path)

    for z in bundle_merge_zips:
      self._add_zip_contents(z['src'], z['dest'], output_path)

    for f in bundle_merge_files:
      self._add_files(f['src'], f['dest'], f.get('executable', False),
                      output_path)

    os.chmod(output_path, 0o755)

    post_processor = self._control.get('post_processor')
    if post_processor:
      self._post_process_bundle(output_path, post_processor)

    code_signing_commands = self._control.get('code_signing_commands')
    if code_signing_commands:
      self._sign_bundle(output_path, code_signing_commands)

  def _add_files(self, src, dest, executable, bundle_root):
    """Adds a file or a directory of files to the bundle.

    Args:
      src: The path to the file or directory that should be added.
      dest: The path relative to the bundle root where the files should be
          stored. If `src` is a single file, then `dest` should include the
          filename that the file should have within the bundle. If `src` is a
          directory, it represents the directory into which the files underneath
          `src` will be recursively added.
      executable: A Boolean value indicating whether or not the file(s) should
          be made executable.
      bundle_root: The bundle root directory into which the files should be
          added.
    """
    if os.path.isdir(src):
      for root, _, files in os.walk(src):
        relpath = os.path.relpath(root, src)
        for filename in files:
          fsrc = os.path.join(root, filename)
          fdest = os.path.normpath(os.path.join(dest, relpath, filename))
          self._copy_file(fsrc, fdest, executable, bundle_root)
    elif os.path.isfile(src):
      self._copy_file(src, dest, executable, bundle_root)

  def _add_zip_contents(self, src, dest, bundle_root):
    """Adds the contents of another ZIP file/App to the bundle.

    Args:
      src: The path to the file or directory that should be added.
      dest: The path relative to the bundle root where the contents of `src`
          should be expanded. The directory structure of `src` is preserved
          underneath this path.
      bundle_root: The bundle root directory into which the files should be
          added.
    """
    # Some bundle_zip entries may be bundled apps which are nested
    # into the current bundle.
    _, ext = os.path.splitext(src)
    if ext == ".app":
        # If so, we can just copy those into the expected location
        # without any additional processing
        app_name = os.path.basename(src)
        shutil.copytree(src, os.path.join(bundle_root, dest, app_name))
    else:
        with zipfile.ZipFile(src, "r") as src_zip:
            for src_zipinfo in src_zip.infolist():
                # Normalize the destination path to remove any extraneous internal
                # slashes or "." segments, but retain the final slash for directory
                # entries.
                file_dest = os.path.normpath(
                    os.path.join(dest, src_zipinfo.filename)
                )
                if src_zipinfo.filename.endswith("/"):
                    continue

                # Check for Unix --x--x--x permissions.
                executable = src_zipinfo.external_attr >> 16 & 0o111 != 0
                data = src_zip.read(src_zipinfo)
                self._write_entry(file_dest, data, executable, bundle_root)

  def _copy_file(self, src, dest, executable, bundle_root):
    """Copies a file into the bundle.

    Args:
      src: The path to the file or directory that should be added.
      dest: The path relative to the bundle root where the file should be
          stored.
      executable: A Boolean value indicating whether or not the file(s) should
          be made executable.
      bundle_root: The bundle root directory into which the files should be
          added.
    """
    full_dest = os.path.join(bundle_root, dest)
    if (os.path.isfile(full_dest) and
        not filecmp.cmp(full_dest, src, shallow=False)):
      raise BundleConflictError(dest)

    self._makedirs_safely(os.path.dirname(full_dest))
    global _USE_CLONEFILE
    if _USE_CLONEFILE:
      clonefile = _load_clonefile()
      result = clonefile(src.encode(), full_dest.encode(), 0)
      if result != 0:
        if get_errno() in (errno.EXDEV, errno.ENOTSUP):
          _USE_CLONEFILE = False
          shutil.copy(src, full_dest)
        else:
          raise Exception(f"failed to clonefile {src} to {full_dest}")
    else:
      shutil.copy(src, full_dest)
    os.chmod(full_dest, 0o755 if executable else 0o644)

  def _write_entry(self, dest, data, executable, bundle_root):
    """Writes the given data as a file in the output ZIP archive.

    Args:
      data: The data to be written in a file in the bundle.
      dest: The path relative to the bundle root where the data should be
          written.
      executable: A Boolean value indicating whether or not the file should be
          made executable.
      bundle_root: The bundle root directory into which the files should be
          added.
    Raises:
      BundleConflictError: If two files with different content would be placed
          at the same location in the ZIP file.
    """
    full_dest = os.path.join(bundle_root, dest)
    if os.path.isfile(full_dest):
      with open(full_dest, "rb") as f:
        if f.read() != data:
          raise BundleConflictError(dest)

    self._makedirs_safely(os.path.dirname(full_dest))
    with open(full_dest, 'wb') as f:
      f.write(data)
    os.chmod(full_dest, 0o755 if executable else 0o644)

  def _makedirs_safely(self, path):
    """Creates a new directory, silently succeeding if it already exists.

    Args:
      path: The path to the directory. Any parent directories that do not exist
          will also be created.
    """
    if not os.path.isdir(path):
      os.makedirs(path)

  def _post_process_bundle(self, bundle_root, post_processor):
    """Executes the post processing tool for the bundle.

    Args:
      bundle_root: The path to the bundle.
      post_processor: The path to the tool or script that should be executed on
          the bundle before it is signed.
    """
    work_dir = os.path.dirname(bundle_root)
    # Configure the TREE_ARTIFACT_OUTPUT environment variable to the path of the
    # bundle, but keep the work_dir for compatibility with the bundletool post
    # processing.
    exit_code = os.system('TREE_ARTIFACT_OUTPUT="%s" %s "%s"' %
                          (bundle_root, post_processor, work_dir))
    if exit_code:
      raise PostProcessorError(exit_code)

  def _sign_bundle(self, bundle_root, command_lines):
    """Executes the signing command lines on the bundle.

    Args:
      bundle_root: The path to the bundle.
      command_lines: A newline-separated list of command lines that should be
        executed in the bundle to sign it.
    """
    exit_code = os.system(
        'WORK_DIR=%s\n%s' % (shlex.quote(bundle_root), command_lines)
    )
    if exit_code:
      raise CodeSignError(exit_code)


def _main(control_path):
  with open(control_path) as control_file:
    control = json.load(control_file)

  bundler = Bundler(control)
  bundler.run()


if __name__ == '__main__':
  if len(sys.argv) != 2:
    sys.stderr.write('ERROR: Expected path to control file and nothing else.\n')
    exit(1)

  _main(sys.argv[1])
