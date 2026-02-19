# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The utility class for provisioning profile."""

import os
import pwd
import shutil
import subprocess
import tempfile
import uuid

from xctestrunner.shared import ios_errors
from xctestrunner.shared import plist_util


class ProvisiongProfile(object):
  """Handles the provisioning profile operations."""

  def __init__(self,
               provisioning_profile_path,
               work_dir=None,
               keychain_path=None):
    """Initializes the provisioning profile.

    Args:
      provisioning_profile_path: string, the path of the provisioning profile.
      work_dir: string, the path of the root temp directory.
      keychain_path: string, the path of the keychain to use.
    """
    self._provisioning_profile_path = provisioning_profile_path
    self._work_dir = work_dir
    self._decode_provisioning_profile_plist = None
    self._keychain_path = keychain_path
    self._name = None
    self._uuid = None

  @property
  def name(self):
    """Gets the name of the provisioning profile."""
    if not self._name:
      self._DecodeProvisioningProfile()
      self._name = self._decode_provisioning_profile_plist.GetPlistField('Name')
    return self._name

  @property
  def uuid(self):
    """Gets the UUID of the provisioning profile."""
    if not self._uuid:
      self._DecodeProvisioningProfile()
      self._uuid = self._decode_provisioning_profile_plist.GetPlistField('UUID')
    return self._uuid

  def Install(self):
    """Installs the provisioning profile to the current login user."""
    target_provisioning_profile_path = os.path.join(
        GetProvisioningProfilesDir(), '%s.mobileprovision' % self.uuid)
    if not os.path.exists(target_provisioning_profile_path):
      shutil.copyfile(self._provisioning_profile_path,
                      target_provisioning_profile_path)

  def _DecodeProvisioningProfile(self):
    """Decodes the provisioning profile. It only works on MacOS."""
    if self._decode_provisioning_profile_plist:
      return

    if not self._work_dir:
      self._work_dir = tempfile.mkdtemp()
    decode_provisioning_profile = os.path.join(
        self._work_dir,
        'decode_provision_%s.plist' % str(uuid.uuid1()))
    command = [
        'security', 'cms', '-D', '-i', self._provisioning_profile_path, '-o',
        decode_provisioning_profile
    ]
    if self._keychain_path:
      command.extend(['-k', self._keychain_path])
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = process.communicate()[0].decode('utf-8')
    if process.poll() != 0:
      raise ios_errors.ProvisioningProfileError(output)
    if not os.path.exists(decode_provisioning_profile):
      raise ios_errors.ProvisioningProfileError(
          'Failed to decode the provisioning profile.')

    self._decode_provisioning_profile_plist = plist_util.Plist(
        decode_provisioning_profile)


def GetProvisioningProfilesDir():
  """Gets the provisioning profiles dir in current login user."""
  home_dir = pwd.getpwuid(os.geteuid()).pw_dir
  path = os.path.join(home_dir, 'Library/MobileDevice/Provisioning Profiles')
  if not os.path.exists(path):
    os.makedirs(path)
  return path
