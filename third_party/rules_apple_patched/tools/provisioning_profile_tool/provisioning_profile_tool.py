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

"""A tool to extract information from a provisioning profile.

Provisioning profiles have embedded plist with information that is used
during the signing process. This tool is used by the bundling rules to extract
the information.

This script takes one argument, a path to a file containing the JSON
representation of a "control" structure. This control structure is a
dictionary with the following keys:

  entitlements: If present, a string that denotes the path to write out the
      default entitlements.plist contained in the profile.
  profile_metadata: If present, a string that denotes the path to write out
      some of the metadata from the plist in the profile. The full metadata
      can be rather large so only a subset that will be useful during the
      preperations to do codesigning is extracted.
  provisioning_profile: Required. The path to the provisioning profile to
      load from.
  target: Required. The target name, used for warning/error messages.
"""

import json
import platform
import plistlib
import subprocess
import sys


UNKNOWN_CONTROL_KEYS_MSG = (
    'Target "%s" used a control structure with unknown key(s): %s'
)

EXTRACT_FROM_PROFILE_FAILED = (
    'While processing target "%s", failed to extract from the provisioning '
    'profile "%s".'
)

# All valid keys in the a control structure.
_CONTROL_KEYS = frozenset([
    'entitlements', 'profile_metadata', 'provisioning_profile', 'target',
])


class ProvisioningProfileToolError(RuntimeError):
  """Raised for all errors.

  Custom RuntimeError used to allow catching (and logging) just the
  ProvisioningProfileTool errors.
  """

  def __init__(self, msg):
    """Initializes an error with the given message.

    Args:
      msg: The message for the error.
    """
    RuntimeError.__init__(self, msg)


class ProvisioningProfileTool(object):
  """Implements the clang runtime tool."""

  def __init__(self, control):
    """Initializes ProvisioningProfileTool.

    Args:
      control: The dictionary of options used to control the tool. Please see
          the moduledoc for a description of the format of this dictionary.
    """
    self._control = control

  def run(self):
    """Performs the operations requested by the control struct."""
    target = self._control.get('target')
    if not target:
      raise ProvisioningProfileToolError('No target name in control.')

    unknown_keys = set(self._control.keys()) - _CONTROL_KEYS
    if unknown_keys:
      raise ProvisioningProfileToolError(UNKNOWN_CONTROL_KEYS_MSG % (
          target, ', '.join(sorted(unknown_keys))))

    profile_path = self._control.get('provisioning_profile')
    if not profile_path:
      raise ProvisioningProfileToolError('Missing provisioning profile path')

    extracted = self._extract_from_profile(target, profile_path)

    profile_metadata_path = self._control.get('profile_metadata')
    if profile_metadata_path:
      self._write_metadata(profile_metadata_path, extracted)

    entitlements_path = self._control.get('entitlements')
    if entitlements_path:
      self._write_default_entitlements(entitlements_path, extracted)

  @classmethod
  def _extract_from_profile(self, target, profile_path):
    """Extracts the plist from the provisioning profile.

    Args:
      target: The target being built, used for error reporting.
      profile_path: Path to the provisioning profile.
    Returns:
      The plist as a dictionary.
    """
    as_str = self._extract_raw_plist(target, profile_path)
    return plistlib.loads(as_str)

  @classmethod
  def _write_default_entitlements(self, output_path, provisioning_profile):
    """Write out the default entitlements from the provisioning profile.

    Args:
      output_path: Where to write the entitlements.
      provisioning_profile: The provisioning profile.
    """
    entitlements = provisioning_profile['Entitlements']
    if hasattr(plistlib, 'dump'):
      with open(output_path, 'wb') as fp:
        plistlib.dump(entitlements, fp)
    else:
      plistlib.writePlist(entitlements, output_path)

  @classmethod
  def _write_metadata(self, output_path, provisioning_profile):
    """Write out the metadata from the profile.

    Args:
      output_path: Where to write the data.
      provisioning_profile: The provisioning profile.
    """
    # The keys likely to be useful. We use an explicit list to ensure nothing
    # extra get pulled that won't be useful and could be large. Depending
    # on the company, the ProvisionedDevices can be pretty long, so there
    # is no reason to drag that along for parsing.
    # Known to be skipping: 'DeveloperCertificates', 'ProvisionedDevices',
    #    'ProvisionsAllDevices'.
    keys = (
        'AppIDName', 'ApplicationIdentifierPrefix', 'CreationDate', 'Platform',
        'Entitlements', 'ExpirationDate', 'Name', 'TeamIdentifier', 'TeamName',
        'TimeToLive', 'UUID', 'Version',
    )
    output_data = {k: provisioning_profile[k] for k in keys}
    if hasattr(plistlib, 'dump'):
      with open(output_path, 'wb') as fp:
        plistlib.dump(output_data, fp)
    else:
      plistlib.writePlist(output_data, output_path)

  @classmethod
  def _extract_raw_plist(self, target, profile_path):
    """Extracts the raw plist from the binary provisioning profile.

    Args:
      target: The target being built, used for error reporting.
      profile_path: Path to the provisioning profile.
    Returns:
      The plist as a string.
    """
    content = open(profile_path, mode='rb').read()
    if content.startswith(b'<?xml'):
      # Back door for testing.
      return content

    # There are two possible ways to try and extract the plist from a
    # provisioning profile: security tool or openssl -
    #   El Capitan: only openssl works.
    #   Sierra: both work.
    #   High Sierra: only security tool works (as of the .1 update, Apple
    #       could change things).
    # If automating a build on a bot, folks don't always set things up to
    # provide access to the keychain, and that causes security tool to fail.
    # So, to try and make things work in as many OS versions and different
    # setups as possible; try *everything* that might work, and only log if
    # everything fails.
    commands_to_try = (
        (
            'security', 'cms', '-D', '-i', profile_path
        ),
        (
            'openssl', 'smime', '-inform', 'der', '-verify', '-noverify',
            '-in', profile_path
        ),
    )
    results = []
    for extract_cmd in commands_to_try:
      extract_process = subprocess.Popen(
          extract_cmd,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
      )
      extracted, err_str = extract_process.communicate()
      if extract_process.returncode == 0:
        # Success, return it.
        return extracted
      # Failure, capture the info to eventually report it if nothing works.
      results.append((extract_cmd, extract_process.returncode, err_str))

    # Exited the loop, nothing worked.
    for extract_cmd, ret_code, err_str in results:
      sys.stderr.write(
          'ERROR: Extraction command "%s" -- result: %d\n%s' %
          (' '.join(extract_cmd), ret_code, err_str))
    raise ProvisioningProfileToolError(EXTRACT_FROM_PROFILE_FAILED % (
        target, profile_path))


def _main(control_path):
  """Called when running the tool from a shell.

  Args:
    control_path: The path to the control file.
  """
  with open(control_path) as control_file:
    control = json.load(control_file)

  tool = ProvisioningProfileTool(control)
  try:
    tool.run()
  except ProvisioningProfileToolError as e:
    # Log tools errors cleanly for build output.
    sys.stderr.write('ERROR: %s\n' % e)
    sys.exit(1)


if __name__ == '__main__':
  if len(sys.argv) != 2:
    sys.stderr.write(
        'ERROR: Path to control file not specified.\n')
    exit(1)

  _main(sys.argv[1])
