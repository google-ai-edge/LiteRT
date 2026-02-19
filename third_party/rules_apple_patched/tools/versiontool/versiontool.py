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

"""Version number extraction from Bazel build labels.

Arbitrary labels can be associated with a build by passing the `--embed_label`
flag. Release management systems can use this to tag the build with information
that can be used to decide the build number/version number of a bundle without
requiring that that transient information be checked into source control.

This script takes two arguments. The first points to a file containing the JSON
representation of a "control" structure. This control structure is a dictionary
with the following keys:

  build_info_path: The path to the build info file (`ctx.info_file.path` from
      Starlark) that contains the embedded label information.
  build_label_pattern: The regular expression that should be matched against the
      build label, with possible placeholders corresponding to `capture_groups`.
  build_version_pattern: The string (possibly containing placeholders) that
      should be used as the value of `CFBundleVersion`.
  fallback_build_label: A build label to use when the no `--embed_label` was
      provided on the build.
  capture_groups: A dictionary whose keys correspond to placeholders found in
      `build_label_pattern` and whose values are regular expressions that should
      be used to match and capture those segments.
  short_version_string_pattern: The string (possibly containing placeholders)
      that should be used as the value of `CFBundleShortVersionString`. If
      omitted, `build_version_pattern` will be used.

The second argument is the path to the output file. The output is written as a
JSON dictionary containing at most two values:

  build_version: The string to use for `CFBundleVersion`.
  short_version_string: The string to use for `CFBundleShortVersionString`.

This dictionary may be empty if there was no build label found in the build info
file. (This allows the script to complete gracefully in local development when
the --embed_label flag is often not being passed.)
"""

import contextlib
import json
import re
import string
import sys


class VersionToolError(ValueError):
  """Raised for all errors.

  Custom ValueError used to allow catching (and logging) just the VersionTool
  errors.
  """

  def __init__(self, msg):
    """Initializes an error with the given message.

    Args:
      msg: The message for the error.
    """
    ValueError.__init__(self, msg)


class DefaultFormatDict(dict):
  """A dictionary that ignores non-present args when passed to `vformat`.

  If a key is requested that is not in the dictionary, then `{key}` is returned,
  which effectively ignores formatting placeholders in the `vformat` string that
  are not present in the dictonary.
  """

  def __missing__(self, key):
    return '{%s}' % key


@contextlib.contextmanager
def _testable_open(fp, mode='r'):
  """Opens a file or uses an existing file-like object.

  This allows the logic to be written in such a way that it does not care
  whether the "paths" its given in the control structure are paths to files or
  file-like objects (such as StringIO) that support testing.

  Args:
    fp: Either a string representing the path to a file that should be opened,
        or an existing file-like object that should be used directly.
    mode: The mode with which to open the file, if `fp` is a string.
  Yields:
    The file-like object to be used in the body of the nested statements.
  """
  if hasattr(fp, 'read') and hasattr(fp, 'write'):
    yield fp
  else:
    yield open(fp, mode)


class VersionTool(object):
  """Implements the core functionality of the versioning tool."""

  def __init__(self, control):
    """Initializes VersionTool with the given control options.

    Args:
      control: The dictionary of options used to control the tool. Please see
          the moduledoc for a description of the format of this dictionary.
    """
    self._build_info_path = control.get('build_info_path')
    self._build_label_pattern = control.get('build_label_pattern')
    self._build_version_pattern = control.get('build_version_pattern')
    self._capture_groups = control.get('capture_groups')
    # `or None` on the next to normalize empty string to None also.
    self._fallback_build_label = control.get('fallback_build_label') or None

    # Use the build_version pattern if short_version_string is not specified so
    # that they both end up the same.
    self._short_version_string_pattern = control.get(
        'short_version_string_pattern') or self._build_version_pattern

  def run(self):
    """Performs the operations requested by the control struct."""
    substitutions = {}
    build_label = None

    if self._build_label_pattern:
      build_label = self._extract_build_label() or self._fallback_build_label

      # It's ok if the build label is not present; this is common during local
      # development.
      if build_label:
        # Substitute the placeholders with named capture groups to extract
        # the components from the label and add them to the substitutions
        # dictionary.
        resolved_pattern = self._build_label_pattern
        for name, pattern in self._capture_groups.items():
          resolved_pattern = resolved_pattern.replace(
              "{%s}" % name, "(?P<%s>%s)" % (name, pattern))
        match = re.match(resolved_pattern, build_label)
        if match:
          substitutions = match.groupdict()
        else:
          raise VersionToolError(
              'The build label ("%s") did not match the pattern ("%s").' %
              (build_label, resolved_pattern))

    # Build the result dictionary by substituting the extracted values for
    # the placeholders. Also, verify that all groups have been substituted; it's
    # an error if they weren't (unless no --embed_label was provided at all, in
    # which case we silently allow it to support local development easily).
    result = {}

    build_version = self._substitute_and_verify(
        self._build_version_pattern, substitutions, 'build_version',
        build_label)
    if build_version:
      result['build_version'] = build_version

    short_version_string = self._substitute_and_verify(
        self._short_version_string_pattern, substitutions,
        'short_version_string', build_label)
    if short_version_string:
      result['short_version_string'] = short_version_string

    return result

  def _extract_build_label(self):
    """Extracts and returns the build label from the build info file.

    Returns:
      The value of the `BUILD_EMBED_LABEL` line in the build info file, or None
      if the file did not exist.
    """
    if not self._build_info_path:
      return None

    with _testable_open(self._build_info_path) as build_info_file:
      content = build_info_file.read()
      match = re.search(r"^BUILD_EMBED_LABEL\s(.*)$", content, re.MULTILINE)
      if match:
        return match.group(1)

    return None

  @staticmethod
  def _substitute_and_verify(pattern, substitutions, key, build_label):
    """Substitutes placeholders with captured values and verifies completeness.

    If no build label was passed via --embed_label, then the version number will
    be used only if it does not contain any placeholders. If it does, then it
    is an error.

    Args:
      pattern: The build version or short version string pattern, potentially
          containing placeholders, to substitute into.
      substitutions: The dictionary of substitutions to make.
      key: The name of the result dictionary key being processed, for error
          reporting purposes.
      build_label: The build label from which values were extracted, for error
          reporting purposes.
    Returns:
      The substituted version string, or None if it still contained
      placeholders but no --embed_label was set.
    Raises:
      VersionToolError if --embed_label was provided but the version string
      still contained placeholders after substitution.
    """
    version = string.Formatter().vformat(
        pattern, (), DefaultFormatDict(**substitutions))
    if re.search(r"\{[^}]*\}", version):
      if build_label:
        raise VersionToolError(
            '--embed_label had a non-empty label ("%s") but the version string '
            '"%s" ("%s") still contained placeholders after substitution' % (
                build_label, key, version))
      else:
        return None

    return version


def _main(control_path, output_path):
  """Called when running the tool from a shell.

  Args:
    control_path: The path to the control file.
    output_path: The path to the file where the output will be written.
  """
  with open(control_path) as control_file:
    control = json.load(control_file)

  tool = VersionTool(control)
  try:
    version_data = tool.run()
  except VersionToolError as e:
    # Log tools errors cleanly for build output.
    sys.stderr.write('ERROR: %s\n' % e)
    sys.exit(1)

  with open(output_path, 'w') as output_file:
    # Sort the keys to get deterministic ordering of the output JSON.
    json.dump(version_data, output_file, sort_keys=True)


if __name__ == '__main__':
  if len(sys.argv) < 3:
    sys.stderr.write(
        'ERROR: Path to control file and/or output file not specified.\n')
    exit(1)

  _main(sys.argv[1], sys.argv[2])
