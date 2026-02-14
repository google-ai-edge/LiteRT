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

"""Plist manipulation for Apple packaging rules.

The "defaults" tool provided with OS X is somewhat satisfactory for reading and
writing single values in a plist, but merging whole plists with conflict
detection is not as easy.

This script takes a single argument that points to a file containing the JSON
representation of a "control" structure (similar to the PlMerge tool, which
takes a binary protocol buffer). This control structure is a dictionary with
the following keys:

  plists: A list of plists that will be merged. The items in this list may be
      strings (which are interpreted as paths), readable file-like objects
      containing XML-formatted plist data (for testing), or dictionaries that
      are treated as inlined plists. Key-value pairs within the plists in this
      list must not conflict (i.e., the same key must not have different values
      in different plists) or the tool will raise an error.
  forced_plists: A list of plists that will be merged after those in "plists".
      Unlike those, collisions between key-value pairs in these plists do not
      raise an error; they replace any values from the originals instead. If
      multiple plists have the same key, the last one in this list is the one
      that will be kept.
  output: A string indicating the path to where the merged plist will be
      written, or a writable file-like object (for testing).
  binary: If true, the output plist file will be written in binary format;
      otherwise, it will be written in XML format. This property is ignored if
      |output| is not a path.
  entitlements_options: A dictionary containing options specific to
      entitlements plist files. Omit this key if you are merging or converting
      other plists (such as Info.plists or other files). See below for more
      details.
  info_plist_options: A dictionary containing options specific to Info.plist
      files. Omit this key if you are merging or converting general plists
      (such as entitlements or other files). See below for more details.
  raw_substitutions: A dictionary of string pairs to use for substitutions.
      Unlike variable_substitutions, there is now "wrapper" added to the keys
      so this can match any *raw* substring in any value in the plist. This
      should be used with extreme care.
  variable_substitutions: A dictionary of string pairs to use for ${VAR}/$(VAR)
      substitutions when processing the plists. All keys/values will get
      support for the rfc1034identifier qualifier.
  target: The target name, used for warning/error messages.

The info_plist_options dictionary can contain the following keys:

  pkginfo: If present, a string that denotes the path to a PkgInfo file that
      should be created from the CFBundlePackageType and CFBundleSignature keys
      in the final merged plist. (For testing purposes, this may also be a
      writable file-like object.)
  version_file: If present, a string that denotes the path to the version file
      propagated by an `AppleBundleVersionInfo` provider, which contains values
      that will be used for the version keys in the Info.plist.
  version_keys_required: If True, the tool will error if the merged Info.plist
      does not contain both CFBundleShortVersionString and CFBundleVersion.
  child_plists: If present, a dictionary containing plists that will be
      compared against the final compiled plist for consistency. The keys of
      the dictionary are the labels of the targets to which the associated
      plists belong. See below for the details of how these are validated.
  child_plist_required_values: If present, a dictionary constaining the
      entries for key/value pairs a child is required to have. This
      dictionary is keyed by the label of the child targets (just like the
      `child_plists`), and the valures are a list of key/value pairs. The
      key/value pairs are encoded as a list of exactly two items, the key is
      actually an array of keys, so it can walk into the child plist.

If info_plist_options is present, validation will be performed on the output
file after merging is complete. If any of the following conditions are not
satisfied, an error will be raised:

  * The CFBundleIdentifier and CFBundleShortVersionString values of the
    output file will be compared to the child plists for consistency. Child
    plists are expected to have the same bundle version string as the parent
    and should have bundle IDs that are prefixed by the bundle ID of the
    parent.

The entitlements_options dictionary can contain the following keys:

  bundle_id: String with the bundle id for the app the entitlements are for.
  profile_metadata_file: A string that denotes the path to a provisioning
      profiles metadata plist. This is the the subset of data as created by
      provisioning_profile_tool.
  validation_mode: A string of "error", "warn", or "skip" to control how
      entitlements are checked against the provisioning profile's entitlements.
      If no value is given, "error" is assumed.

If entitlements_options is present, validation will be performed on the output
file after merging is complete. If any of the following conditions are not
satisfied, an error will be raised:

  * The bundle_id provided in the options will be checked against the
    `application-identifier` in the entitlements to ensure they are
    a match.
  * The requested entitlements from the merged file are checked against
    those provided by the provisioning profile's supported entitlements.
    Some mismatches are only warnings, others are errors. Warnings are
    used when the build target may still work (iOS Simulator), but
    errors are used where the build result is known not to work (iOS
    devices).

"""

# NOTE: Ideally the final plist will always be byte for byte the same, not
# just contain the same data. Xcode, which is built on Foundation's
# non-order-preserving NSDictionary, is harmful to caching because it merges
# keys in arbitrary order. Python dictionaries make no guarantee about an
# iteration order, but appear to have repeatable behavior for the same inputs,
# for the same version of python.
#
# tl;dr; - Rely on Python's dictionary iteration behavior until it becomes
# a problem.
#
# What was considered...
#
# Inputs come in via plist files and json files. Since the inputs can come
# from anything a developer what (i.e. - they could have a genrule and
# failed to have worried about stable outs), the best approach is to ensure
# stabilization during output. One could use python dictionaries until the
# the very end; then iterator over its keys in sorted order, recursively
# copying into an OrderedDict. It has to be recursive to ensure sub
# dictionaries are also stable.
#
# But, plistlib.writePlist doesn't make any promises about how it works, so
# passing it an OrderedDict might or might not work, and could be subject to
# versions of the module.  But this output approach is likely the best to
# getting stable outputs.
#
# However... when a binary file is the desired result, plutil is invoked to
# convert the file to binary, and that again makes no promises. So even if
# feed a stable input, the output might not be deterministic when run on
# different machines and/or different macOS versions.

import copy
import datetime
import json
import plistlib
import re
import subprocess
import sys


# Format strings for errors that are raised, exposed here to the tests
# can validate against them.

CHILD_BUNDLE_ID_MISMATCH_MSG = (
    'While processing target "%s"; the CFBundleIdentifier of the child target '
    '"%s" should have "%s" as its prefix, but found "%s".'
)

CHILD_BUNDLE_VERSION_MISMATCH_MSG = (
    'While processing target "%s"; the %s of the child target "%s" should be '
    'the same as its parent\'s version string "%s", but found "%s".'
)

REQUIRED_CHILD_MISSING_MSG = (
    'While processing target "%s"; "child_plist_required_values" wanted to '
    'check "%s", but it wasn\'t in the the "child_plists".'
)

REQUIRED_CHILD_NOT_PAIR = (
    'While processing target "%s"; "child_plist_required_values" for "%s", '
    'got something other than a key/value pair: %r'
)

REQUIRED_CHILD_KEYPATH_NOT_FOUND = (
    'While processing target "%s"; the Info.plist for child target "%s" '
    'should have and entry for "%s" or %r, but does not.'
)

REQUIRED_CHILD_KEYPATH_NOT_MATCHING = (
    'While processing target "%s"; the Info.plist for child target "%s" '
    'has the wrong value for "%s"; expected %r, but found %r.'
)

MISSING_KEY_MSG = (
    'Target "%s" is missing %s.'
)

UNEXPECTED_KEY_MSG = (
    'Unexpectedly found key "%s" in target "%s".'
)

INVALID_VERSION_KEY_VALUE_MSG = (
    'Target "%s" has a %s that doesn\'t meet Apple\'s guidelines: "%s". See '
    'https://developer.apple.com/library/content/technotes/tn2420/_index.html'
    ' and '
    'https://developer.apple.com/library/content/documentation/General/Reference/InfoPlistKeyReference/Articles/CoreFoundationKeys.html'
)

PLUTIL_CONVERSION_TO_XML_FAILED_MSG = (
    'While processing target "%s", plutil failed (%d) to convert "%s" to xml.'
)

CONFLICTING_KEYS_MSG = (
    'While processing target "%s"; found key "%s" in two plists with different '
    'values: "%s" != "%s"'
)

UNKNOWN_CONTROL_KEYS_MSG = (
    'Target "%s" used a control structure has unknown key(s): %s'
)

UNKNOWN_TASK_OPTIONS_KEYS_MSG = (
    'Target "%s" used %s that included unknown key(s): %s'
)

INVALID_SUBSTITUTATION_REFERENCE_MSG = (
    'In target "%s"; invalid variable reference "%s" while merging '
    'plists (key: "%s", value: "%s").'
)

UNKNOWN_SUBSTITUTATION_REFERENCE_MSG = (
    'In target "%s"; unknown variable reference "%s" while merging '
    'plists (key: "%s", value: "%s").'
)

UNKNOWN_SUBSTITUTION_ADDITION_APPIDENTIFIERPREFIX_MSG = (
    'This can mean the rule failed to set the "provisioning_profile" ' +
    'attribute so the prefix could be extracted.'
)

UNSUPPORTED_SUBSTITUTATION_REFERENCE_IN_KEY_MSG = (
    'In target "%s"; variable reference "%s" found in key "%s" merging '
    'plists.'
)

INVALID_SUBSTITUTION_VARIABLE_NAME = (
    'On target "%s"; invalid variable name for substitutions: "%s".'
)

SUBSTITUTION_VARIABLE_CANT_HAVE_QUALIFIER = (
    'On target "%s"; variable name for substitutions can not have a '
    'qualifier: "%s".'
)

OVERLAP_IN_SUBSTITUTION_KEYS = (
    'Target "%s" has overlap in the from a raw substitution, overlapping '
    'keys: "%s" and "%s".'
)

RAW_SUBSTITUTION_KEY_IN_VALUE = (
    'Target "%s" has raw substitution key "%s" that appears in the another '
    'substitution: "%s" for key "%s".'
)

ENTITLEMENTS_BUNDLE_ID_MISMATCH = (
    'In target "%s"; the bundle_id ("%s") did not match the id in the '
    'entitlements ("%s").'
)

ENTITLEMENTS_PROFILE_HAS_EXPIRED = (
    'On target "%s", provisioning profile ExpirationDate ("%s") is in the '
    'past.'
)

ENTITLEMENTS_TEAM_ID_PROFILE_MISMATCH = (
    'In target "%s"; the entitlements "com.apple.developer.team-identifier" '
    '("%s") did not match the provisioning profile\'s "%s" ("%s").'
)

ENTITLEMENTS_APP_ID_PROFILE_MISMATCH = (
    'In target "%s"; the entitlements "application-identifier" ("%s") did not '
    'match the value in the provisioning profile ("%s").'
)

ENTITLEMENTS_HAS_GROUP_PROFILE_DOES_NOT = (
    'Target "%s" uses entitlements with a "%s" key, but the profile does not '
    'support use of this key.'
)

ENTITLEMENTS_MISSING = (
    'Target "%s" uses entitlements with the '
    '"%s" key, but the profile does not have this key'
)

ENTITLEMENTS_VALUE_MISMATCH = (
    'In target "%s"; the entitlement value for "%s" ("%s") '
    'did not match the value in the provisioning profile ("%s").'
)

ENTITLEMENTS_VALUE_NOT_IN_LIST = (
    'In target "%s"; the entitlement value for "%s" ("%s") '
    'is not in the provisioning profiles potential values ("%s").'
)

_ENTITLEMENTS_TO_VALIDATE_WITH_PROFILE = (
    'aps-environment',
    'com.apple.developer.applesignin',
    'com.apple.developer.carplay-audio',
    'com.apple.developer.carplay-charging',
    'com.apple.developer.carplay-maps',
    'com.apple.developer.carplay-messaging',
    'com.apple.developer.carplay-parking',
    'com.apple.developer.carplay-quick-ordering',
    'com.apple.developer.playable-content',
    'com.apple.developer.networking.wifi-info',
    'com.apple.developer.passkit.pass-presentation-suppression',
    'com.apple.developer.payment-pass-provisioning',
    'com.apple.developer.proximity-reader.payment.acceptance',
    'com.apple.developer.siri',
    'com.apple.developer.usernotifications.critical-alerts',
    'com.apple.developer.usernotifications.time-sensitive',
    # Keys which have a list of potential values in the profile, but only one in
    # the entitlements that must be in the profile's list of values
    'com.apple.developer.devicecheck.appattest-environment',
)

ENTITLEMENTS_BETA_REPORTS_ACTIVE_MISMATCH = (
    'In target "%s"; the entitlements "beta-reports-active" ("%s") did not '
    'match the value in the provisioning profile ("%s").'
)

ENTITLEMENTS_BETA_REPORTS_ACTIVE_MISSING_PROFILE = (
    'In target "%s"; the entitlements file has "beta-reports-active" ("%s") '
    'but it does not exist in the profile.'
)

ENTITLEMENTS_HAS_GROUP_ENTRY_PROFILE_DOES_NOT = (
    'Target "%s" uses entitlements "%s" value of "%s", but the profile does '
    'not support it (["%s"]).'
)

ENTITLEMENTS_VALUE_HAS_WILDCARD = (
    'Target "%s" uses entitlements "%s" value of "%s", but wildcards are not '
    'expected.'
)

# All valid keys in the a control structure.
_CONTROL_KEYS = frozenset([
    'binary', 'forced_plists', 'entitlements_options', 'info_plist_options',
    'output', 'plists', 'raw_substitutions', 'target',
    'variable_substitutions',
])

# All valid keys in the info_plist_options control structure.
_INFO_PLIST_OPTIONS_KEYS = frozenset([
    'child_plists', 'child_plist_required_values', 'pkginfo', 'version_file',
    'version_keys_required', 'extensionkit_keys_required'
])

# All valid keys in the entitlements_options control structure.
_ENTITLEMENTS_OPTIONS_KEYS = frozenset([
    'bundle_id', 'profile_metadata_file', 'validation_mode',
])

# Two regexes for variable matching/validation.
# VARIABLE_REFERENCE_RE: Matches things that look mostly a
#   variable reference.
# VARIABLE_NAME_RE: Is used to match the name from the first regex to
#     confirm it is a valid name.
VARIABLE_REFERENCE_RE = re.compile(r'\$(\(|\{)([^\)\}]*)((\)|\})?|$)')
VARIABLE_NAME_RE = re.compile('^([a-zA-Z0-9_]+)(:rfc1034identifier)?$')

# Regex for RFC1034 normalization, see _convert_to_rfc1034()
_RFC1034_RE = re.compile(r'[^0-9A-Za-z.]')

# Info.plist "versioning" keys: CFBundleVersion & CFBundleShortVersionString
#
# Apple's docs are spares and not very specific. The best info seems to be:
#  "Core Foundation Keys" -
#    https://developer.apple.com/library/content/documentation/General/Reference/InfoPlistKeyReference/Articles/CoreFoundationKeys.html
#  TN2420 - "Version Numbers and Build Numbers" -
#    https://developer.apple.com/library/content/technotes/tn2420/_index.html
#
# - For mobile the details seem to be more complete, and the AppStore process
#   helps fill in some gaps by failing uploads (or installs) for wrong/missing
#   information.
# - As of Xcode 9.0, new projects get a CFBundleVersion of "1" and a
#   CFBundleShortVersionString "1.0".
# - So while it isn't ever fully spelled out any place, the Apple Bazel rules
#   take the stance that when one key is needed, both should be provided, and
#   the formats should match what is outlines in the above two documents and
#   the Xcode templates.

# Regexes for the two version keys, some enforcement is handled via the helper
# methods below.
#
# CFBundleVersion:
# - 1-3 segments of numbers and then the "development, alpha, beta, and final
#   candidate" bits that are allowed.
#   - "Core Foundation Keys" lists a limited on the number of characters in the
#     segments, but that doesn't seem to be enforced.
#   - Leading zero on numbers are ok.
#   - NOTE: "Core Foundation Keys" also says the first segment must have a
#     value that is >= 1. That is *not* going to be enforced since using 0.x.y
#     very early in a project is common and at this level there is no way to
#     really tell if this is a Release/AppStore build or not.
#   - NOTE: While the docs all say 3 segments, enterprise builds (and
#     TestFlight?) are perfectly happy with 4 segment, so 4 is allowed.
#   - The syntax for the optional suffix for development builds is documented
#     incompletely (only by example) and also inconsistently (being
#     described as 'The character in the suffix', implying
#     that only a single character is allowed, but also including 'fc' as an
#     example suffix, which clearly contains more than one character).
#     In practice, xcode allows longer suffixes too, e.g. '1.2.3.foo4',
#     so we allow them here too.  The development suffix in any case needs to
#     be removed before a mobile app can be uploaded to the app store.

# - TechNote also lists an 18 characters max
CF_BUNDLE_VERSION_RE = re.compile(
    r'^[0-9]+(\.[0-9]+){0,3}([a-z]+(?P<track_num>[0-9]{1,3}))?$'
)
BUNDLE_VERSION_VALUE_MAX_LENGTH = 18
# CFBundleShortVersionString:
# - 1-3 segments of numbers.
#   - The "Core Foundation Keys" does not list any limited on the number of
#     characters in the segments.
#   - Doesn't say anything about leading zeros, assume still ok.
#   - NOTE: While the docs all say 3 segments, enterprise builds (and
#     TestFlight?) are perfectly happy with 4 segment, so 4 is allowed.
# - TechNote also lists an 18 characters max
CF_BUNDLE_SHORT_VERSION_RE = re.compile(
    r'^[0-9]+(\.[0-9]+){0,3}$'
)


def plist_from_bytes(byte_content):
  try:
    return plistlib.loads(byte_content)
  except AttributeError:
    return plistlib.readPlistFromString(byte_content)


def extract_variable_from_match(re_match_obj):
  """Takes a match from VARIABLE_REFERENCE_RE and extracts the variable.

  This funciton is exposed to testing.

  Args:
    re_match_obj: a re.MatchObject
  Returns:
    The variable name (with qualifier attached) or None if the match wasn't
    completely valid.
  """
  expected_close = '}' if re_match_obj.group(1) == '{' else ')'
  if re_match_obj.group(3) == expected_close:
    m = VARIABLE_NAME_RE.match(re_match_obj.group(2))
    if m:
      return m.group(0)
  return None


def is_valid_version_string(s):
  """Checks if the given string is a valid CFBundleVersion.

  Args:
    s: The string to check.
  Returns:
    True/False based on if the string meets Apple's rules.
  """
  if len(s) > BUNDLE_VERSION_VALUE_MAX_LENGTH:
    return False
  m = CF_BUNDLE_VERSION_RE.match(s)
  if not m:
    # Didn't match, must be invalid.
    return False
  # The RE doesn't validate the "development, alpha, beta, and final candidate"
  # bits, so that is done manually.
  track_num = m.group('track_num')
  if track_num:
    # Can't start with a zero.
    if track_num.startswith('0'):
      return False
    # Must be <= 255.
    if int(track_num) > 255:
      return False
  return True


def is_valid_short_version_string(s):
  """Checks if the given string is a valid CFBundleShortVersionString.

  Args:
    s: The string to check.
  Returns:
    True/False based on if the string meets Apple's rules.
  """
  if len(s) > BUNDLE_VERSION_VALUE_MAX_LENGTH:
    return False
  m = CF_BUNDLE_SHORT_VERSION_RE.match(s)
  return m is not None


def get_with_key_path(a_dict, key_path):
  """Helper to walk a keypath into a dict and return the value.

  Remember when walking into lists, they are zero indexed.

  Args:
    a_dict: The dictionary to walk into.
    key_path: A list of keys to walk into the dictionary.
  Returns:
    The object or None if the keypath can't be walked.
  """
  value = a_dict
  try:
    for key in key_path:
      if isinstance(value, (str, int, float)):
        # There are leaf types, can't keep pathing down
        return None
      value = value[key]
  except (IndexError, KeyError, TypeError):
    # List index out of range, unknown dict key, passing a string key to a list
    return None
  return value


def _convert_to_rfc1034(string):
  """Forces the given value into RFC 1034 compliance.

  This function replaces any bad characters with '-' as Xcode would in its
  plist substitution.

  Args:
    string: The string to convert.
  Returns:
    The converted string.
  """
  return _RFC1034_RE.sub('-', string)


def _load_json(string_or_file):
  """Helper to load json from a path for file like object.

  Args:
    string_or_file: If a string, load the JSON from the path. Otherwise assume
        it is a file like object and load from it.
  Returns:
    The object graph loaded.
  """
  if isinstance(string_or_file, str):
    with open(string_or_file) as f:
      return json.load(f)
  return json.load(string_or_file)


class PlistToolError(ValueError):
  # pylint: disable=g-bad-exception-name
  """Raised for all errors.

  Custom ValueError used to allow catching (and logging) just the plisttool
  errors.
  """

  def __init__(self, msg):
    """Initializes an error with the given message.

    Args:
      msg: The message for the error.
    """
    ValueError.__init__(self, msg)


class SubstitutionEngine(object):
  """Helper that can apply substitutions while copying values."""

  def __init__(self,
               target,
               variable_substitutions=None,
               raw_substitutions=None):
    """Initialize a SubstitutionEngine.

    Args:
      target: Name of the target being built, used in messages/errors.
      variable_substitutions: A dictionary of variable names to the values
          to use for substitutions.
      raw_substitutions: A dictionary of raw names to the values to use for
          substitutions.

    Raises:
      PlistToolError: if there are any errors with variable/raw subtitutions.
    """
    self._substitutions = {}
    self._substitutions_re = None

    subs = variable_substitutions or {}
    for key, value in subs.items():
      m = VARIABLE_NAME_RE.match(key)
      if not m:
        raise PlistToolError(INVALID_SUBSTITUTION_VARIABLE_NAME % (
            target, key))
      if m.group(2):
        raise PlistToolError(SUBSTITUTION_VARIABLE_CANT_HAVE_QUALIFIER % (
            target, key))
      value_rfc = _convert_to_rfc1034(value)
      for fmt in ('${%s}', '$(%s)'):
        self._substitutions[fmt % key] = value
        self._substitutions[fmt % (key + ':rfc1034identifier')] = value_rfc

    raw_subs = raw_substitutions or {}
    for key, value in raw_subs.items():
      # Raw keys can't overlap any other key (var or raw).
      for existing_key in sorted(self._substitutions):
        if (key in existing_key) or (existing_key in key):
          ordered = sorted([key, existing_key])
          raise PlistToolError(
              OVERLAP_IN_SUBSTITUTION_KEYS % (target, ordered[0], ordered[1]))
      self._substitutions[key] = value

    # A raw key can't overlap any value.
    raw_keys = sorted(raw_subs.keys())
    for k, v in sorted(self._substitutions.items()):
      for raw_key in raw_keys:
        if raw_key in v:
          raise PlistToolError(
              RAW_SUBSTITUTION_KEY_IN_VALUE % (target, raw_key, v, k))

    # Make _substitutions_re.
    if self._substitutions:
      escaped_keys = [re.escape(x) for x in self._substitutions.keys()]
      self._substitutions_re = re.compile('(%s)' % '|'.join(escaped_keys))

  def apply_substitutions(self, value):
    """Applies variable substitutions to the given value.

    If the value is a string, the text will have the substitutions
    applied. If it is an array or dictionary, then the substitutions will
    be recursively applied to its members. Otherwise (for booleans or
    numbers), the value will remain untouched.

    Args:
      value: The value with possible variable references to substitute.
    Returns:
      The value with any variable references substituted with their new
      values.
    """
    if not self._substitutions_re:
      return value
    return self._internal_apply_subs(value)

  def _internal_apply_subs(self, value):
    """Recursive substitutions for string, dictionaries and lists."""
    if isinstance(value, str):

      def sub_helper(match_obj):
        return self._substitutions[match_obj.group(0)]
      return self._substitutions_re.sub(sub_helper, value)

    if isinstance(value, dict):
      return {k: self._internal_apply_subs(v) for k, v in value.items()}

    if isinstance(value, list):
      return [self._internal_apply_subs(v) for v in value]

    return value

  @classmethod
  def validate_no_variable_references(cls,
                                      target,
                                      key_name,
                                      value,
                                      msg_additions=None):
    """Ensures there are no variable references left in value (recursively).

    Args:
      target: The name of the target for which the plist is being built.
      key_name: The name of the key this value is part of.
      value: The value to check.
      msg_additions: Dictionary of variable names to custom strings to add to
        the error messages.
    Raises:
      PlistToolError: If there is a variable substitution that wasn't resolved.
    """
    additions = {}
    if msg_additions:
      for k, v in msg_additions.items():
        additions[k] = v
        additions[k + ':rfc1034identifier'] = v

    def _helper(key_name, value):
      if isinstance(value, str):
        m = VARIABLE_REFERENCE_RE.search(value)
        if m:
          variable_name = extract_variable_from_match(m)
          if not variable_name:
            # Reference wasn't property formed, raise that issue.
            raise PlistToolError(INVALID_SUBSTITUTATION_REFERENCE_MSG % (
                target, m.group(0), key_name, value))
          err_msg = UNKNOWN_SUBSTITUTATION_REFERENCE_MSG % (
              target, m.group(0), key_name, value)
          msg_addition = additions.get(variable_name)
          if msg_addition:
            err_msg = err_msg + ' ' + msg_addition
          raise PlistToolError(err_msg)
        return

      if isinstance(value, dict):
        key_prefix = key_name + ':' if key_name else ''
        for k, v in value.items():
          _helper(key_prefix + k, v)
          m = VARIABLE_REFERENCE_RE.search(k)
          if m:
            raise PlistToolError(
                UNSUPPORTED_SUBSTITUTATION_REFERENCE_IN_KEY_MSG % (
                    target, m.group(0), key_prefix + k))
        return

      if isinstance(value, list):
        for i, v in enumerate(value):
          reporting_key = '%s[%d]' % (key_name, i)
          _helper(reporting_key, v)
        return

    # Off we go...
    _helper(key_name, value)


class PlistIO(object):
  """Helpers for read/writing plists.

  These helpers make it easy to use files, streams, or literals without the
  callers having to know.
  """

  @classmethod
  def get_dict(cls, p, target):
    """Returns a plist dictionary based on the given object.

    This function handles the various input formats for plists in the control
    struct that are supported by this tool. Dictionary objects are returned
    verbatim; strings are treated as paths to plist files, and anything else
    is assumed to be a readable file-like object whose contents are plist data.

    Args:
      p: The object to interpret as a plist.
      target: The name of the target for which the plist is being built.
    Returns:
      A dictionary containing the values from the plist.
    """
    if isinstance(p, dict):
      return p

    if isinstance(p, str):
      with open(p, 'rb') as plist_file:
        return cls._read_plist(plist_file, p, target)

    return cls._read_plist(p, '<input>', target)

  @classmethod
  def _read_plist(cls, plist_file, name, target):
    """Reads a plist file and returns its contents as a dictionary.

    This method wraps the readPlist method in plistlib by checking the format
    of the plist before reading and using plutil to convert it into XML format
    first, to support plain text and binary formats as well.

    Args:
      plist_file: The file-like object containing the plist data.
      name: Name to report the file-like object as if it fails xml conversion.
      target: The name of the target for which the plist is being built.
    Returns:
      The contents of the plist file as a dictionary.

    Raises:
      PlistToolError: if plutil return code is non-zero.
    """
    plist_contents = plist_file.read()

    # Binary plists are easy to identify because they start with 'bplist'. For
    # plain text plists, it may be possible to have leading whitespace, but
    # well-formed XML should *not* have any whitespace before the XML
    # declaration, so we can check that the plist is not XML and let plutil
    # handle them the same way.
    if not plist_contents.startswith(b'<?xml'):
      plutil_process = subprocess.Popen(
          ['plutil', '-convert', 'xml1', '-o', '-', '--', '-'],
          stdout=subprocess.PIPE,
          stdin=subprocess.PIPE
      )
      plist_contents, _ = plutil_process.communicate(plist_contents)
      if plutil_process.returncode:
        raise PlistToolError(PLUTIL_CONVERSION_TO_XML_FAILED_MSG % (
            target, plutil_process.returncode, name))

    return plist_from_bytes(plist_contents)

  @classmethod
  def write(cls, plist, path_or_file, binary=False):
    """Writes the given plist to the output file.

    This method also converts it to binary format if "binary" is True in the
    control struct.

    Args:
      plist: The plist to write to the output path in the control struct.
      path_or_file: The name of file to write or or a file like object to
          write into.
      binary: If True and path_or_file was a file name, reformat the file
          in binary form.
    """
    if isinstance(path_or_file, str):
      with open(path_or_file, 'wb') as fp:
        plistlib.dump(plist, fp)
    else:
      plistlib.dump(plist, path_or_file)

    if binary and isinstance(path_or_file, str):
      subprocess.check_call(['plutil', '-convert', 'binary1', path_or_file])


class PlistToolTask(object):
  """Base for adding subtasks to the plist tool."""

  def __init__(self, target, options):
    """Initialize the task.

    Args:
      target: The name of the target being processed.
      options: The dictionary from the control to configure this option.
    """
    self.target = target
    self.options = options

  @classmethod
  def control_structure_options_name(cls):
    """The name of the dictionary of options for this task.

    The options will be a dictionary in the control structre to plisttool.
    """
    raise NotImplementedError('Subclass must provide this.')

  @classmethod
  def options_keys(cls):
    """Returns the set of valid keys in the options structure."""
    raise NotImplementedError('Subclass must provide this.')

  def extra_variable_substitutions(self):
    """Variable substitutions specific to this task to apply to plist merging."""
    return {}  # Default to nothing for subclasses.

  def extra_raw_substitutions(self):
    """Raw substitutions specific to this task to apply to plist merging."""
    return {}  # Default to nothing for subclasses.

  def unknown_variable_message_additions(self):
    """Things to add to unknown variable messages.

    The resulting dictionary should be keyed by the variable name.

    Returns:
      Empty dictionary
    """
    return {}  # Default to nothing for subclasses.

  def update_plist(self, out_plist, subs_engine):
    """Update anything needed in in the plist.

    Args:
      out_plist: The dictionary representing the merged plist so far. This
          dictionary will may be modified as the task desires.
      subs_engine: A SubstitutionEngine instance to use if needed.
    """
    pass  # Default to nothing for subclasses

  def validate_plist(self, plist):
    """Do any final checks on the resulting plist.

    Args:
      plist: The dictionary representing final plist, no changes may be
        made to the plist. If there are any issues a PlistToolError should
        be raised with the problems.
    """
    pass  # Default to nothing for subclasses


class InfoPlistTask(PlistToolTask):
  """Info.plist specific task when processing."""

  @classmethod
  def control_structure_options_name(cls):
    return 'info_plist_options'

  @classmethod
  def options_keys(cls):
    return _INFO_PLIST_OPTIONS_KEYS

  def update_plist(self, out_plist, subs_engine):
    # Pull in the version info propagated by AppleBundleVersionInfo.
    version_file = self.options.get('version_file')
    if version_file:
      version_info = _load_json(version_file)
      bundle_version = version_info.get('build_version')
      short_version_string = version_info.get('short_version_string')

      if bundle_version:
        out_plist['CFBundleVersion'] = bundle_version
      if short_version_string:
        out_plist['CFBundleShortVersionString'] = short_version_string

  def validate_plist(self, plist):
    if self.options.get('version_keys_required'):
      for k in ('CFBundleVersion', 'CFBundleShortVersionString'):
        # This also errors if they are there but the empty string or zero.
        if not plist.get(k, None):
          raise PlistToolError(MISSING_KEY_MSG % (self.target, k))

    if self.options.get('extensionkit_keys_required'):
      # Check if the old NSExtension plist key has been accidentally set.
      if 'NSExtension' in plist:
        raise PlistToolError(UNEXPECTED_KEY_MSG % ('NSExtension', self.target))

      # Check that the required ExtensionKit extension point ID has been set.
      if 'EXAppExtensionAttributes' not in plist:
        raise PlistToolError(
            MISSING_KEY_MSG % (self.target, 'EXAppExtensionAttributes')
        )
      if not plist['EXAppExtensionAttributes'].get(
          'EXExtensionPointIdentifier', None
      ):
        raise PlistToolError(
            MISSING_KEY_MSG % (self.target, 'EXExtensionPointIdentifier')
        )

    # If the version keys are set, they must be valid (even if they were
    # not required).
    for k, validator in (
        ('CFBundleVersion', is_valid_version_string),
        ('CFBundleShortVersionString', is_valid_short_version_string),
    ):
      v = plist.get(k)
      if v and not validator(v):
        raise PlistToolError(
            INVALID_VERSION_KEY_VALUE_MSG % (self.target, k, v)
        )

    child_plists = self.options.get('child_plists')
    child_plist_required_values = self.options.get(
        'child_plist_required_values'
    )
    if child_plists:
      self._validate_children(
          plist, child_plists, child_plist_required_values, self.target
      )

    pkginfo_file = self.options.get('pkginfo')
    if pkginfo_file:
      if isinstance(pkginfo_file, str):
        with open(pkginfo_file, 'wb') as p:
          self._write_pkginfo(p, plist)
      else:
        self._write_pkginfo(pkginfo_file, plist)

  @staticmethod
  def _validate_children(plist, child_plists, child_required_values, target):
    """Validates a target's plist is consistent with its children.

    This function checks each of the given child plists (which are typically
    extensions or sub-apps embedded in another application) and fails the build
    if there are any issues.

    Args:
      plist: The final plist of the target being built.
      child_plists: The plists of child targets that the target being built
          depends on.
      child_required_values: Mapping of any key/value pairs to validate in
          the children.
      target: The name of the target being processed.
    Raises:
      PlistToolError: if there was an inconsistency between a child target's
          plist and the current target's plist, with a message describing what
          was incorrect.
    """
    if child_required_values is None:
      child_required_values = dict()

    prefix = plist['CFBundleIdentifier'] + '.'
    version = plist.get('CFBundleVersion')
    short_version = plist.get('CFBundleShortVersionString')

    for label, p in child_plists.items():
      child_plist = PlistIO.get_dict(p, target)

      child_id = child_plist['CFBundleIdentifier']
      if not child_id.startswith(prefix):
        raise PlistToolError(CHILD_BUNDLE_ID_MISMATCH_MSG % (
            target, label, prefix, child_id))

      # - TN2420 calls out CFBundleVersion and CFBundleShortVersionString
      #   has having to match for watchOS targets.
      #   https://developer.apple.com/library/content/technotes/tn2420/_index.html
      # - The Application Loader (and Xcode) have also given errors for
      #   iOS Extensions that don't share the same values for the two
      #   version keys as they parent App. So we enforce this for all
      #   platforms just to be safe even though it isn't otherwise
      #   documented.
      #   https://stackoverflow.com/questions/30441750/use-same-cfbundleversion-and-cfbundleshortversionstring-in-all-targets

      child_version = child_plist.get('CFBundleVersion')
      if version != child_version:
        raise PlistToolError(CHILD_BUNDLE_VERSION_MISMATCH_MSG % (
            target, 'CFBundleVersion', label, version, child_version))

      child_version = child_plist.get('CFBundleShortVersionString')
      if short_version != child_version:
        raise PlistToolError(CHILD_BUNDLE_VERSION_MISMATCH_MSG % (
            target, 'CFBundleShortVersionString', label, short_version,
            child_version))

      required_info = child_required_values.get(label, [])
      for pair in required_info:
        if not isinstance(pair, list) or len(pair) != 2:
          raise PlistToolError(REQUIRED_CHILD_NOT_PAIR % (target, label, pair))

        [key_path, expected] = pair
        value = get_with_key_path(child_plist, key_path)
        if value is None:
          key_path_str = ':'.join([str(x) for x in key_path])
          raise PlistToolError(REQUIRED_CHILD_KEYPATH_NOT_FOUND % (
              target, label, key_path_str, expected))

        if value != expected:
          key_path_str = ':'.join([str(x) for x in key_path])
          raise PlistToolError(REQUIRED_CHILD_KEYPATH_NOT_MATCHING % (
              target, label, key_path_str, expected, value))

    # Make sure there wasn't anything listed in required that wasn't listed
    # as a child.
    for label in child_required_values.keys():
      if label not in child_plists:
        raise PlistToolError(REQUIRED_CHILD_MISSING_MSG % (target, label))

  @classmethod
  def _write_pkginfo(cls, pkginfo, plist):
    """Writes a PkgInfo file with contents from the given plist.

    Args:
      pkginfo: A writable file-like object into which the PkgInfo data will be
          written.
      plist: The plist containing the bundle package type and signature that
          will be written into the PkgInfo.
    """
    package_type = cls._four_byte_pkginfo_string(
        plist.get('CFBundlePackageType'))
    signature = cls._four_byte_pkginfo_string(
        plist.get('CFBundleSignature'))

    pkginfo.write(package_type)
    pkginfo.write(signature)

  @staticmethod
  def _four_byte_pkginfo_string(value):
    """Encodes a plist value into four bytes suitable for a PkgInfo file.

    Args:
      value: The value that is a candidate for the PkgInfo file.
    Returns:
      If the value is a string that is exactly four bytes long, it is returned;
      otherwise, '????' is returned instead.
    """
    try:
      if not isinstance(value, str):
        return b'????'

      if isinstance(value, bytes):
        value = value.decode('utf-8')

      # Based on some experimentation, Xcode appears to use MacRoman encoding
      # for the contents of PkgInfo files, so we do the same.
      value = value.encode('mac-roman')

      return value if len(value) == 4 else b'????'
    except (UnicodeDecodeError, UnicodeEncodeError):
      # Return the default string if any character set encoding/decoding errors
      # occurred.
      return b'????'


class EntitlementsTask(PlistToolTask):
  """Entitlements specific task when processing."""

  def __init__(self, target, options):
    super(EntitlementsTask, self).__init__(target, options)
    self._extra_raw_subs = {}
    self._extra_var_subs = {}
    self._unknown_var_msg_addtions = {}
    self._profile_metadata = {}
    self._validation_mode = self.options.get('validation_mode', 'error')

    assert self._validation_mode in ('error', 'warn', 'skip')

    # Load the metadata so the content can be used for substitutions and
    # validations.
    profile_metadata_file = self.options.get('profile_metadata_file')
    if profile_metadata_file:
      self._profile_metadata = PlistIO.get_dict(profile_metadata_file, target)
      ver = self._profile_metadata.get('Version')
      if ver != 1:
        # Just log the message incase something else goes wrong.
        print(('WARNING: On target "%s", got a provisioning profile with a ' +
               '"Version" other than "1" (%s).') % (self.target, ver))

    if self._profile_metadata:
      # Even though the provisioning profile had a TeamIdentifier, the previous
      # entitlements code used ApplicationIdentifierPrefix:0, so use that to
      # maintain behavior in case it was important.
      team_prefix_list = self._profile_metadata.get(
          'ApplicationIdentifierPrefix')
      team_prefix = team_prefix_list[0] if team_prefix_list else None

      if team_prefix:
        # Note: These subs must be set up by plisttool (and not passed in)
        # via the *_substitutions keys in the control because it takes an
        # action running to extract them from the provisioning profile, so
        # the starlark for the rule doesn't have access to the values.
        #
        # Set up the subs using the info extracted from the provisioning
        # profile:
        # - "PREFIX.*" -> "PREFIX.BUNDLE_ID"
        bundle_id = self.options.get('bundle_id')
        if bundle_id:
          self._extra_raw_subs['%s.*' % team_prefix] = '%s.%s' % (
              team_prefix, bundle_id)
        # - "$(AppIdentifierPrefix)" -> "PREFIX."
        self._extra_var_subs['AppIdentifierPrefix'] = '%s.' % team_prefix

    else:
      self._unknown_var_msg_addtions.update({
          'AppIdentifierPrefix':
              UNKNOWN_SUBSTITUTION_ADDITION_APPIDENTIFIERPREFIX_MSG,
      })

  @classmethod
  def control_structure_options_name(cls):
    return 'entitlements_options'

  @classmethod
  def options_keys(cls):
    return _ENTITLEMENTS_OPTIONS_KEYS

  def extra_variable_substitutions(self):
    return self._extra_var_subs

  def extra_raw_substitutions(self):
    return self._extra_raw_subs

  def unknown_variable_message_additions(self):
    return self._unknown_var_msg_addtions

  def update_plist(self, out_plist, subs_engine):
    # Retrieves forced entitlement keys from provisioning profile metadata to
    # add them to the final entitlements used by codesign and clang.
    profile_entitlements = self._profile_metadata.get('Entitlements')

    if not profile_entitlements:
      return

    forced_profile_entitlements = [
        'application-identifier',
        'com.apple.security.get-task-allow',
        'get-task-allow',
    ]

    for forced_entitlement in forced_profile_entitlements:

      if forced_entitlement in out_plist:
        # Validation is skipped since validate_plist takes care of this.
        continue

      if forced_entitlement in profile_entitlements:
        out_plist[forced_entitlement] = profile_entitlements[forced_entitlement]

  def validate_plist(self, plist):
    bundle_id = self.options.get('bundle_id')
    if bundle_id:
      self._validate_bundle_id_covered(bundle_id, plist)

    if self._profile_metadata:
      self._sanity_check_profile()

      if self._validation_mode != 'skip':
        self._validate_entitlements_against_profile(plist)

  def _validate_bundle_id_covered(self, bundle_id, entitlements):
    """Checks that the bundle id is covered by the entitlements.

    Args:
      bundle_id: The bundle id to check.
      entitlements: The entitlements.
    Raises:
      PlistToolError: If the bundle_id isn't covered by the entitlements.
    """
    # The entitlements passed to codesign can completely lack an
    # 'application-identifier' entry. This appears to cause codesign to add
    # one with the "right" info during the signing process. So, no entry means
    # skip this check since it appears to always "just work".
    app_id = entitlements.get('application-identifier')
    if app_id is None:
      return

    # The app id has the team prefix as the first component, so drop that.
    provisioned_id = app_id.split('.', 1)[1]

    if not self._does_id_match(bundle_id, provisioned_id,
                               allowed_supports_wildcards=True):
      raise PlistToolError(ENTITLEMENTS_BUNDLE_ID_MISMATCH % (
          self.target, bundle_id, provisioned_id))

  def _sanity_check_profile(self):
    """Some basic checks of the profile info to ensure signing will work.

    Raises:
      PlistToolError: If there are any issues.
    """
    # The "Version" is checked during __init__ and a printout is generated
    # if it isn't 1.f
    # There also are CreationDate "xxx" and "TimeToLive" keys (date and
    # integer), but just checking "ExpirationDate" seems better.
    expire = self._profile_metadata.get('ExpirationDate')
    if expire and expire < datetime.datetime.now():
      raise PlistToolError(ENTITLEMENTS_PROFILE_HAS_EXPIRED % (
          self.target, expire.isoformat()))

    # There is a "Platform" key (contains an array), but looking at at some
    # profiles used by working watchOS targets, they still say "iOS", so it
    # does not appear to be valid to double check the platform.

    # "ApplicationIdentifierPrefix" and "TeamIdentifier" (arrays) that likely
    # should always match (we use "ApplicationIdentifierPrefix") in __init__
    # for setting up substitutions. At the moment no validation between them
    # is being done.

  def _validate_entitlements_against_profile(self, entitlements):
    """Checks that the given entitlements are valid for the current profile.

    Args:
      entitlements: The entitlements.
    Raises:
      PlistToolError: For any issues found.
    """
    # com.apple.developer.team-identifier vs profile's TeamIdentifier
    # Not verifying against profile's ApplicationIdentifierPrefix here, because
    # it isn't always equal to the Team ID.
    # https://developer.apple.com/library/archive/technotes/tn2415/_index.html#//apple_ref/doc/uid/DTS40016427-CH1-ENTITLEMENTSLIST
    src_team_id = entitlements.get('com.apple.developer.team-identifier')
    if src_team_id:
      key = 'TeamIdentifier'
      from_profile = self._profile_metadata.get(key, [])
      if src_team_id not in from_profile:
        self._report(
            ENTITLEMENTS_TEAM_ID_PROFILE_MISMATCH % (
                  self.target, src_team_id, key, from_profile))

    profile_entitlements = self._profile_metadata.get('Entitlements')

    # application-identifier
    src_app_id = entitlements.get('application-identifier')
    if src_app_id and profile_entitlements:
      profile_app_id = profile_entitlements.get('application-identifier')
      if profile_app_id and not self._does_id_match(
          src_app_id, profile_app_id, allowed_supports_wildcards=True,
          id_supports_wildcards=True):
        self._report(
            ENTITLEMENTS_APP_ID_PROFILE_MISMATCH % (
                self.target, src_app_id, profile_app_id))

    for entitlement in _ENTITLEMENTS_TO_VALIDATE_WITH_PROFILE:
      self._check_entitlement_matches_profile_value(
          entitlement=entitlement,
          entitlements=entitlements,
          profile_entitlements=profile_entitlements)

    # If beta-reports-active is in either the profile or the entitlements file
    # it must be in both or the upload will get rejected by Apple
    beta_reports_active = entitlements.get('beta-reports-active')
    profile_key = (profile_entitlements or {}).get('beta-reports-active')
    if beta_reports_active is not None and profile_key != beta_reports_active:
      error_msg = ENTITLEMENTS_BETA_REPORTS_ACTIVE_MISMATCH % (
          self.target, beta_reports_active, profile_key)
      if profile_key is None:
        error_msg = ENTITLEMENTS_BETA_REPORTS_ACTIVE_MISSING_PROFILE % (
            self.target, beta_reports_active)
      self._report(error_msg)

    # keychain-access-groups
    self._check_entitlements_array(
        entitlements, profile_entitlements,
        'keychain-access-groups', self.target,
        supports_wildcards=True)

    # com.apple.security.application-groups
    # (This check does not apply to macOS-only provisioning profiles.)
    if self._profile_metadata.get('Platform', []) != ['OSX']:
      self._check_entitlements_array(
          entitlements, profile_entitlements,
          'com.apple.security.application-groups', self.target)

    # com.apple.developer.associated-domains
    self._check_entitlements_array(
        entitlements, profile_entitlements,
        'com.apple.developer.associated-domains', self.target,
        supports_wildcards=True,
        allow_wildcards_in_entitlements=True)

    # com.apple.developer.nfc.readersession.formats
    self._check_entitlements_array(
        entitlements,
        profile_entitlements,
        'com.apple.developer.nfc.readersession.formats',
        self.target)

  def _check_entitlement_matches_profile_value(
      self,
      entitlement,
      entitlements,
      profile_entitlements):
    """Checks if an entitlement value matches against profile entitlement.

    If provisioning profile entitlement is defined as a list, this will
    check if entitlement is part of that list.

    Args:
      entitlement: Entitlement key identifier.
      entitlements: Entitlements dictionary.
      profile_entitlements: Provisioning Profile entitlements dictionary.
    """
    entitlements_value = entitlements.get(entitlement)
    if entitlements_value is None:
      return

    profile_value = (profile_entitlements or {}).get(entitlement)
    if profile_value is None:
      # provisioning profile does not have entitlement.
      self._report(ENTITLEMENTS_MISSING % (self.target, entitlement))

    elif isinstance(profile_value, list):
      if (isinstance(entitlements_value, str) and
          entitlements_value not in profile_value):
        # provisioning profile does not have entitlement in list.
        self._report(
            ENTITLEMENTS_VALUE_NOT_IN_LIST %
            (self.target, entitlement, entitlements_value, profile_value))

      if isinstance(entitlements_value, list):
        self._check_entitlements_array(
            entitlements,
            profile_entitlements,
            entitlement,
            self.target)

    elif isinstance(profile_value, (str, bool)):
      if entitlements_value != profile_value:
        # provisioning profile entitlement does not match value.
        self._report(
            ENTITLEMENTS_VALUE_MISMATCH % (
                self.target, entitlement, entitlements_value, profile_value))

  def _does_id_match(self,
                     entitlement_id,
                     allowed,
                     allowed_supports_wildcards=False,
                     id_supports_wildcards=False):
    """Check is an id matches the given allowed id (include wildcards).

    Args:
      entitlement_id: The identifier to check.
      allowed: The allowed identifier which can end in a wildcard.
      allowed_supports_wildcards: True/False for if wildcards should
          be supported in the `allowed` value.
      id_supports_wildcards: True/False for if a wildcard should be
          allowed/supported in the input id. This is very rare.
    Returns:
      True/False if the identifier is covered.
    """
    if allowed_supports_wildcards and allowed.endswith('*'):
      if entitlement_id.startswith(allowed[:-1]):
        return True
    else:
      if entitlement_id == allowed:
        return True

    # Since entitlements files can use wildcards, the file a developer
    # makes could have a wildcard and the value within the profile could
    # also have a wildcard. The substitutions done normally remove the
    # wildcard in the processed entitlements file, but just in case it
    # doesn't, validate that the two agree.
    if id_supports_wildcards and entitlement_id.endswith('*'):
      if allowed.endswith('*'):
        if entitlement_id[:-1].startswith(allowed[:-1]):
          return True
      else:
        if allowed.startswith(entitlement_id[:-1]):
          return True

    return False

  def _does_id_match_list(self,
                          entitlement_id,
                          allowed_list,
                          allowed_supports_wildcards=False):
    """Check is an id matches the given allowed id list (include wildcards).

    Args:
      entitlement_id: The identifier to check.
      allowed_list: The allowed identifiers which can end in a wildcard.
      allowed_supports_wildcards: True/False for if wildcards should
          be supported in the `allowed_list` values.
    Returns:
      True/False if the identifier is covered.
    """
    for allowed in allowed_list:
      if self._does_id_match(
          entitlement_id,
          allowed,
          allowed_supports_wildcards=allowed_supports_wildcards):
        return True

    return False

  def _check_entitlements_array(self,
                                entitlements,
                                profile_entitlements,
                                key_name,
                                target,
                                supports_wildcards=False,
                                allow_wildcards_in_entitlements=False):
    """Checks if the requested entitlements against the profile for a key.

    Args:
      entitlements: The entitlements.
      profile_entitlements: The provisioning profiles entitlements (from the
          profile metadata).
      key_name: The key to check.
      target: The target to include in errors.
      supports_wildcards: True/False for if wildcards should be supported
          value from the profile_entitlements. This also means the entries
          are reverse DNS style.
      allow_wildcards_in_entitlements: True/False if wildcards are allowed.
    Raises:
      PlistToolError: For any issues found.
    """
    src_grps = entitlements.get(key_name)
    if not src_grps:
      return

    if not profile_entitlements:
      return  # Allow no profile_entitlements just for the plisttool_unittests.

    profile_grps = profile_entitlements.get(key_name)
    if not profile_grps:
      self._report(
          ENTITLEMENTS_HAS_GROUP_PROFILE_DOES_NOT % (target, key_name))
      return

    for src_grp in src_grps:
      if '*' in src_grp and not allow_wildcards_in_entitlements:
        self._report(
            ENTITLEMENTS_VALUE_HAS_WILDCARD % (target, key_name, src_grp))

      if not self._does_id_match_list(
          src_grp, profile_grps, allowed_supports_wildcards=supports_wildcards):
        self._report(
            ENTITLEMENTS_HAS_GROUP_ENTRY_PROFILE_DOES_NOT % (
                target, key_name, src_grp, '", "'.join(profile_grps)))

  def _report(self, msg):
    """Helper for reporting things.

    Args:
      msg: Message to report.
    Raises:
      PlistToolError: if 'validation_mode' flag was set to 'error'.
    """
    if self._validation_mode != 'error':
      print('WARNING: ' + msg)
    else:
      raise PlistToolError(msg)


class PlistTool(object):
  """Implements the core functionality of the plist tool."""

  def __init__(self, control):
    """Initializes PlistTool with the given control options.

    Args:
      control: The dictionary of options used to control the tool. Please see
          the moduledoc for a description of the format of this dictionary.
    """
    self._control = control

  def run(self):
    """Performs the operations requested by the control struct.

    Raises:
      PlistToolError: For any bad input (unknown control structure entries,
          missing required information, etc.) or for processing/validation
          errors.
    """
    target = self._control.get('target')
    if not target:
      raise PlistToolError('No target name in control.')
    output = self._control.get('output')
    if not output:
      raise PlistToolError('No output file specified.')

    def validate_keys(keys, expected, options_name=None):
      """Helper to validate the support keys in control structures."""
      unknown_keys = set(keys) - expected
      if unknown_keys:
        if options_name:
          raise PlistToolError(UNKNOWN_TASK_OPTIONS_KEYS_MSG % (
              target, options_name, ', '.join(sorted(unknown_keys))))
        else:
          raise PlistToolError(UNKNOWN_CONTROL_KEYS_MSG % (
              target, ', '.join(sorted(unknown_keys))))

    # Check for unknown keys in the control structure.
    validate_keys(list(self._control.keys()), _CONTROL_KEYS)

    tasks = []
    var_subs = self._control.get('variable_substitutions', {})
    raw_subs = self._control.get('raw_substitutions', {})
    unknown_var_msg_additions = {}

    task_types = (
        EntitlementsTask,
        InfoPlistTask,
    )
    for task_type in task_types:
      options_name = task_type.control_structure_options_name()
      options = self._control.get(options_name)
      if options is not None:
        validate_keys(list(options.keys()), task_type.options_keys(),
                      options_name=options_name)
        task = task_type(target, options)
        var_subs.update(task.extra_variable_substitutions())
        raw_subs.update(task.extra_raw_substitutions())
        unknown_var_msg_additions.update(
            task.unknown_variable_message_additions())
        tasks.append(task)

    subs_engine = SubstitutionEngine(target, var_subs, raw_subs)
    out_plist = {}
    for p in self._control.get('plists', []):
      plist = PlistIO.get_dict(p, target)
      self._merge_dictionaries(plist, out_plist, target, subs_engine)

    forced_plists = self._control.get('forced_plists', [])
    for p in forced_plists:
      plist = PlistIO.get_dict(p, target)
      self._merge_dictionaries(plist, out_plist, target, subs_engine,
                               override_collisions=True)

    for t in tasks:
      t.update_plist(out_plist, subs_engine)

    SubstitutionEngine.validate_no_variable_references(
        target, '', out_plist, msg_additions=unknown_var_msg_additions)

    if tasks:
      saved_copy = copy.deepcopy(out_plist)
      for t in tasks:
        t.validate_plist(out_plist)
      # Sanity check it wasn't mutated during a validate.
      assert saved_copy == out_plist

    PlistIO.write(out_plist, output, binary=self._control.get('binary'))

  @staticmethod
  def _merge_dictionaries(src, dest, target, subs_engine,
                          override_collisions=False):
    """Merge the top-level keys from src into dest.

    This method is publicly visible for testing.

    Args:
      src: The dictionary whose values will be merged into dest.
      dest: The dictionary into which the values will be merged.
      target: The name of the target for which the plist is being built.
      subs_engine: The SubstitutionEngine to use during the merge.
      override_collisions: If True, collisions will be resolved by replacing
          the previous value with the new value. If False, an error will be
          raised if old and new values do not match.
    Raises:
      PlistToolError: If the two dictionaries had different values for the
          same key.
    """
    for key in src:
      src_value = subs_engine.apply_substitutions(src[key])

      if key in dest:
        dest_value = dest[key]

        if not override_collisions and src_value != dest_value:
          raise PlistToolError(CONFLICTING_KEYS_MSG % (
              target, key, src_value, dest_value))

      dest[key] = src_value


def _main(control_path):
  """Loads JSON parameters file and runs PlistTool."""
  with open(control_path) as control_file:
    control = json.load(control_file)

  tool = PlistTool(control)
  try:
    tool.run()
  except PlistToolError as e:
    # Log tools errors cleanly for build output.
    sys.stderr.write('ERROR: %s\n' % e)
    sys.exit(1)


if __name__ == '__main__':
  if len(sys.argv) < 2:
    sys.stderr.write('ERROR: Path to control file not specified.\n')
    exit(1)

  _main(sys.argv[1])
