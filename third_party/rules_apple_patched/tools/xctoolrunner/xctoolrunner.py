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

"""Wrapper for "xcrun" tools.

This script only runs on Darwin and you must have Xcode installed.

Usage:

  xctoolrunner [SUBCOMMAND] [<args>...]

Subcommands:
  actool [<args>...]

  coremlc [<args>...]

  ibtool [<args>...]

  intentbuilderc [<args>....]

  mapc [<args>...]

  momc [<args>...]

  xcstringstool [<args>...]
"""

import argparse
import os
import re
import shutil
import sys

from tools.wrapper_common import execute

# This prefix is set for rules_apple rules in:
# apple/internal/utils/xctoolrunner.bzl
_PATH_PREFIX = "[ABSOLUTE]"
_PATH_PREFIX_LEN = len(_PATH_PREFIX)
_HEADER_SUFFIX = ".h"


def _apply_realpath(argv):
  """Run "realpath" on any path-related arguments.

  Paths passed into the tool will be prefixed with the contents of _PATH_PREFIX.
  If we find an argument with this prefix, we strip out the prefix and run
  "realpath".

  Args:
    argv: A list of command line arguments.
  """
  for i, arg in enumerate(argv):
    if arg.startswith(_PATH_PREFIX):
      arg = arg[_PATH_PREFIX_LEN:]
      argv[i] = os.path.realpath(arg)

def _execute_and_filter_with_retry(xcrunargs, filtering):
  # Note: `actool`/`ibtool` is problematic on all Xcode 12 builds including to 12.1. 25%
  # of the time, it fails with the error:
  # "failed to open # liblaunch_sim.dylib"
  #
  # This workaround adds a retry it works due to logic in `actool`:
  # The first time `actool` runs, it spawns a dependent service as the current
  # user. After a failure, `actool` spawns it in a way that subsequent
  # invocations will not have the error. It only needs 1 retry.
  return_code, stdout, stderr = execute.execute_and_filter_output(
      xcrunargs,
      trim_paths=True,
      filtering=filtering,
      print_output=False)

  # If there's a retry, don't print the first failing output.
  if return_code == 0:
    if stdout:
      sys.stdout.write("%s" % stdout)
    if stderr:
      sys.stderr.write("%s" % stderr)
    return return_code

  return_code, _, _ = execute.execute_and_filter_output(
      xcrunargs,
      trim_paths=True,
      filtering=filtering,
      print_output=True)
  return return_code

def _ensure_clean_path(path):
  """Ensure a directory exists and is empty."""
  if os.path.exists(path):
    shutil.rmtree(path)
  os.makedirs(path)

def _listdir_full(path):
  """List a directory but output the full path to the files instead of only the
  file names."""
  for f in os.listdir(path):
    yield os.path.join(path, f)

def ibtool_filtering(tool_exit_status, raw_stdout, raw_stderr):
  """Filter messages from ibtool.

  Args:
    tool_exit_status: The exit status of "xcrun ibtool".
    raw_stdout: This is the unmodified stdout captured from "xcrun ibtool".
    raw_stderr: This is the unmodified stderr captured from "xcrun ibtool".

  Returns:
    A tuple of the filtered exit_status, stdout and strerr.
  """

  spurious_patterns = [
      re.compile(x)
      for x in [r"WARNING: Unhandled destination metrics: \(null\)"]
  ]

  def is_spurious_message(line):
    for pattern in spurious_patterns:
      match = pattern.search(line)
      if match is not None:
        return True
    return False

  stdout = []
  for line in raw_stdout.splitlines():
    if not is_spurious_message(line):
      stdout.append(line + "\n")

  # Some of the time, in a successful run, ibtool reports on stderr some
  # internal assertions and ask "Please file a bug report with Apple", but
  # it isn't clear that there is really a problem. Since everything else
  # (warnings about assets, etc.) is reported on stdout, just drop stderr
  # on successful runs.
  if tool_exit_status == 0:
    raw_stderr = None

  return (tool_exit_status, "".join(stdout), raw_stderr)


def ibtool(_, toolargs):
  """Assemble the call to "xcrun ibtool"."""
  xcrunargs = ["xcrun", "ibtool"]

  _apply_realpath(toolargs)

  xcrunargs += toolargs

  # If we are running into problems figuring out "ibtool" issues, there are a
  # couple of environment variables that may help. Both of the following must be
  # set to work.
  #   IBToolDebugLogFile=<OUTPUT FILE PATH>
  #   IBToolDebugLogLevel=4
  # You may also see if
  #   IBToolNeverDeque=1
  # helps.
  return _execute_and_filter_with_retry(xcrunargs=xcrunargs, filtering=ibtool_filtering)


def actool_filtering(tool_exit_status, raw_stdout, raw_stderr):
  """Filter the stdout messages from "actool".

  Args:
    tool_exit_status: The exit status of "xcrun actool".
    raw_stdout: This is the unmodified stdout captured from "xcrun actool".
    raw_stderr: This is the unmodified stderr captured from "xcrun actool".

  Returns:
    A tuple of the filtered exit_status, stdout and strerr.
  """
  section_header = re.compile("^/\\* ([^ ]+) \\*/$")

  excluded_sections = ["com.apple.actool.compilation-results"]

  spurious_patterns = [
      re.compile(x) for x in [
          r"\[\]\[ipad\]\[76x76\]\[\]\[\]\[1x\]\[\]\[\]\[\]: notice: \(null\)",
          r"\[\]\[ipad\]\[76x76\]\[\]\[\]\[1x\]\[\]\[\]\[\]: notice: 76x76@1x "
          r"app icons only apply to iPad apps targeting releases of iOS prior "
          r"to 10\.0\.",
      ]
  ]

  def is_spurious_message(line):
    for pattern in spurious_patterns:
      match = pattern.search(line)
      if match is not None:
        return True
    return False

  def is_warning_or_notice_an_error(line):
    """Returns True if the warning/notice should be treated as an error."""

    # List of warnings that should continue to be treated as warnings.
    warnings = [
        # Requiring a 1024x1024 PNG for App Store distribution. (b/246165573)
        "1024x1024",
        # "foo" is used by multiple imagesets. (b/139094648)
        "is used by multiple",
        # Launch image deprecations.
        "Launch images are deprecated in iOS 13.0",
        "Launch images are deprecated in tvOS 13.0",
        # Xcode 26 watchOS "failed to generate flattened icon stack" errors
        # when building Icon Composer icon bundles without legacy xcassets App
        # Icons that define a "universal" 1024x1024 PNG icon. (b/430862638)
        "Failed to generate flattened icon stack for icon named ",
    ]
    for warning in warnings:
      if warning in line:
        return False
    return True

  def is_warning_muted_by_xcode(line):
    """Returns True if the warning is nonsense and it is ignored by Xcode."""
    warning_substrings_to_ignore = [
        # Xcode 26's actool attempts to parse PNG files as XML and fails; these
        # are predictably consistent and should be ignored. All of these are
        # prepended by whitespace.
        "Failure Reason: The data is not in the correct format.",
        "Underlying Errors:",
        "Debug Description: Garbage at end around line ",
        "Description: The data couldn’t be read because it isn’t in the " +
        "correct format.",
        "Failed to parse icontool JSON output.",
    ]
    for muted_warning in warning_substrings_to_ignore:
      if muted_warning in line:
        return True
    return False

  def is_error_muted_by_xcode(line):
    """Returns True if the error is nonsense and it is ignored by Xcode."""
    error_substrings_to_ignore = [
        # Sometimes we get CoreImage errors that are harmless, referencing SDK
        # artifacts that are not provided by any Xcode.
        "CIPortraitEffectSpillCorrection",
        "RuntimeRoot/System/Library/CoreImage/PortraitFilters.cifilter",
    ]
    exact_errors_to_ignore = [
        # Xcode 26's actool attempts to parse PNG files as XML and fails; these
        # are predictably consistent and should be ignored.
        "Entity: line 1: parser error : Start tag expected, '<' not found",
        "�PNG",
        "^",
    ]
    for muted_error in error_substrings_to_ignore:
      if muted_error in line:
        return True
    if line in exact_errors_to_ignore:
      return True
    return False

  output = set()
  errors_collected = []
  current_section = None

  for line in raw_stderr.splitlines():
    if is_error_muted_by_xcode(line):
      continue
    errors_collected.append(line + "\n")

  if not errors_collected:
    # If the errors reported were all muted, then the tool_exit_status should
    # be 0 at this point, allowing for the possibility that the output might
    # still have warnings we want to upgrade as errors.
    tool_exit_status = 0

  for line in raw_stdout.splitlines():
    header_match = section_header.search(line)

    if header_match:
      current_section = header_match.group(1)
      continue

    if not current_section:
      output.add(line + "\n")
    elif current_section not in excluded_sections:
      if is_spurious_message(line) or is_warning_muted_by_xcode(line):
        continue

      if is_warning_or_notice_an_error(line):
        line = line.replace(": warning: ", ": error: ")
        line = line.replace(": notice: ", ": error: ")
        tool_exit_status = 1

      output.add(line + "\n")

  if tool_exit_status == 0:
    # Some of the time, in a successful run, actool reports on stderr some
    # internal assertions and ask "Please file a bug report with Apple", but
    # it isn't clear that there is really a problem. Since everything else
    # (warnings about assets, etc.) is reported on stdout, just drop stderr
    # on successful runs.
    errors_collected = []

  return (tool_exit_status, "".join(output), "".join(errors_collected))


def actool(_, toolargs):
  """Assemble the call to "xcrun actool"."""
  xcrunargs = ["xcrun", "actool"]

  _apply_realpath(toolargs)

  xcrunargs += toolargs

  # The argument coming after "--compile" is the output directory. "actool"
  # expects an directory to exist at that path. Create an empty directory there
  # if one doesn't exist yet.
  for idx, arg in enumerate(toolargs):
    if arg == "--compile":
      output_dir = toolargs[idx + 1]
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)
      break

  # If we are running into problems figuring out "actool" issues, there are a
  # couple of environment variables that may help. Both of the following must be
  # set to work.
  #   IBToolDebugLogFile=<OUTPUT FILE PATH>
  #   IBToolDebugLogLevel=4
  # You may also see if
  #   IBToolNeverDeque=1
  # helps.
  # Yes, IBTOOL appears to be correct here due to "actool" and "ibtool" being
  # based on the same codebase.
  return _execute_and_filter_with_retry(xcrunargs=xcrunargs, filtering=actool_filtering)


def coremlc(_, toolargs):
  """Assemble the call to "xcrun coremlc"."""
  xcrunargs = ["xcrun", "coremlc"]
  _apply_realpath(toolargs)
  xcrunargs += toolargs

  return_code, _, _ = execute.execute_and_filter_output(
      xcrunargs, print_output=True)
  return return_code

def intentbuilderc(args, toolargs):
  """Assemble the call to "xcrun intentbuilderc"."""
  xcrunargs = ["xcrun", "intentbuilderc"]
  _apply_realpath(toolargs)
  is_swift = args.language == "Swift"

  output_path = None
  objc_output_srcs = None
  objc_output_hdrs = None

  # If the language is Swift, create a temporary directory for codegen output.
  # If the language is Objective-C, ensure the module name directory and headers
  # are created and empty (clean).
  if is_swift:
    output_path = "{}.out.tmp".format(args.swift_output_src)
  else:
    output_path = args.objc_output_srcs
    _ensure_clean_path(args.objc_output_hdrs)

  _ensure_clean_path(output_path)
  output_path = os.path.realpath(output_path)

  toolargs += [
    "-language",
    args.language,
    "-output",
    output_path,
  ]

  xcrunargs += toolargs

  return_code, _, _ = execute.execute_and_filter_output(
      xcrunargs,
      print_output=True)

  if return_code != 0:
    return return_code

  # If the language is Swift, concatenate all the output files into one.
  # If the language is Objective-C, put the headers into the pre-declared
  # headers directory. Because the .m files reference headers via quotes, copy
  # them instead of moving them and doing some -iquote fu.
  if is_swift:
    with open(args.swift_output_src, "w") as output_src:
      for src in _listdir_full(output_path):
        with open(src) as intput_src:
          shutil.copyfileobj(intput_src, output_src)
  else:
    with open(args.objc_public_header, "w") as public_header_f:
      for source_file in _listdir_full(output_path):
        if source_file.endswith(_HEADER_SUFFIX):
          out_hdr = os.path.join(args.objc_output_hdrs, os.path.basename(source_file))
          shutil.copy(source_file, out_hdr)
          public_header_f.write("#import \"{}\"\n".format(os.path.relpath(out_hdr)))

  return return_code

def momc_filtering(tool_exit_status, raw_stdout, raw_stderr):
  """Filter messages from momc.

  Args:
    tool_exit_status: The exit status of "xcrun momc".
    raw_stdout: This is the unmodified stdout captured from "xcrun momc".
    raw_stderr: This is the unmodified stderr captured from "xcrun momc".

  Returns:
    A tuple of the filtered exit_status, stdout and strerr.
  """

  spurious_patterns = [
      re.compile(x) for x in [
          # Xcode 15 prints an internal version checksum for each compiled file:
          #     {name}.xcdatamodel: note: Model {name} version checksum: {base64}
          r": note: Model .* version checksum:",
      ]
  ]

  def is_spurious_message(line):
    for pattern in spurious_patterns:
      match = pattern.search(line)
      if match is not None:
        return True
    return False

  output = set()

  for line in raw_stderr.splitlines():
    if is_spurious_message(line):
      continue

    output.add(line + "\n")

  return (tool_exit_status, raw_stdout, "".join(output))


def momc(args, toolargs):
  """Assemble the call to "xcrun momc"."""
  xcrunargs = ["xcrun", "momc"]
  _apply_realpath(toolargs)
  xcrunargs += toolargs

  return_code, stdout, stderr = execute.execute_and_filter_output(
      xcrunargs, filtering=momc_filtering, print_output=False)

  destination_dir = args.xctoolrunner_assert_nonempty_dir
  if args.xctoolrunner_assert_nonempty_dir and not os.listdir(destination_dir):
    raise FileNotFoundError(
        f"xcrun momc did not generate artifacts at: {destination_dir}\n"
        "Core Data model was not configured to have code generation.")

  if stdout:
    sys.stdout.write("%s" % stdout)
  if stderr:
    sys.stderr.write("%s" % stderr)

  return return_code


def mapc(_, toolargs):
  """Assemble the call to "xcrun mapc"."""
  xcrunargs = ["xcrun", "mapc"]
  _apply_realpath(toolargs)
  xcrunargs += toolargs

  return_code, _, _ = execute.execute_and_filter_output(
      xcrunargs, print_output=True)
  return return_code


def xcstringstool(_, toolargs):
  xcrunargs = ["xcrun", "xcstringstool"]
  _apply_realpath(toolargs)
  xcrunargs += toolargs

  return_code, _, _ = execute.execute_and_filter_output(
      xcrunargs, print_output=True)
  return return_code


def main(argv):
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers()

  # IBTOOL Argument Parser
  ibtool_parser = subparsers.add_parser("ibtool")
  ibtool_parser.set_defaults(func=ibtool)

  # ACTOOL Argument Parser
  actool_parser = subparsers.add_parser("actool")
  actool_parser.set_defaults(func=actool)

  # COREMLC Argument Parser
  coremlc_parser = subparsers.add_parser("coremlc")
  coremlc_parser.set_defaults(func=coremlc)

  # INTENTBUILDERC Argument Parser
  intentbuilderc_parser = subparsers.add_parser("intentbuilderc")
  intentbuilderc_parser.set_defaults(func=intentbuilderc)
  intentbuilderc_parser.add_argument("-language")
  intentbuilderc_parser.add_argument("-objc_output_srcs")
  intentbuilderc_parser.add_argument("-objc_output_hdrs")
  intentbuilderc_parser.add_argument("-objc_public_header")
  intentbuilderc_parser.add_argument("-swift_output_src")

  # MOMC Argument Parser
  momc_parser = subparsers.add_parser("momc")
  momc_parser.add_argument(
      "--xctoolrunner_assert_nonempty_dir",
      help="Enables non-empty destination dir assertion after execution.")
  momc_parser.set_defaults(func=momc)

  # MAPC Argument Parser
  mapc_parser = subparsers.add_parser("mapc")
  mapc_parser.set_defaults(func=mapc)

  xcstringstool_parser = subparsers.add_parser("xcstringstool")
  xcstringstool_parser.set_defaults(func=xcstringstool)

  # Parse the command line and execute subcommand
  args, toolargs = parser.parse_known_args(argv)
  sys.exit(args.func(args, toolargs))


if __name__ == "__main__":
  main(sys.argv[1:])
