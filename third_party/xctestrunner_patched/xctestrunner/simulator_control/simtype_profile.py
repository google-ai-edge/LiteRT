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

"""The helper class to get information from simulator device type's profile."""

import os

from xctestrunner.shared import ios_constants
from xctestrunner.shared import ios_errors
from xctestrunner.shared import plist_util
from xctestrunner.shared import xcode_info_util


class SimTypeProfile(object):
  """The object for simulator device type's profile."""

  def __init__(self, device_type):
    """Constructor of SimulatorProfile object.

    Args:
      device_type: string, device type of the new simulator. The value
          corresponds to the output of `xcrun simctl list devicetypes`.
          E.g., iPhone 6, iPad Air, etc.
    """
    self._device_type = device_type
    self._profile_plist_obj = None
    self._min_os_version = None
    self._max_os_version = None

  @property
  def profile_plist_obj(self):
    """Gets the Plist object of the simulator device type's profile.plist.

    Returns:
      plist_util.Plist, the Plist object of the simulator device type's
      profile.plist.
    """
    if not self._profile_plist_obj:
      xcode_version = xcode_info_util.GetXcodeVersionNumber()
      platform_path = xcode_info_util.GetSdkPlatformPath(
          ios_constants.SDK.IPHONEOS)
      if xcode_version >= 1630:
        sim_profiles_dir = '/Library/Developer/CoreSimulator/Profiles'
      elif xcode_version >= 1100:
        sim_profiles_dir = os.path.join(
            platform_path, 'Library/Developer/CoreSimulator/Profiles')
      else:
        sim_profiles_dir = os.path.join(
            platform_path, 'Developer/Library/CoreSimulator/Profiles')
      profile_plist_path = os.path.join(
          sim_profiles_dir,
          'DeviceTypes/%s.simdevicetype/Contents/Resources/profile.plist' %
          self._device_type)
      if os.path.exists(profile_plist_path):
        self._profile_plist_obj = plist_util.Plist(profile_plist_path)
    return self._profile_plist_obj

  @property
  def min_os_version(self):
    """Gets the min supported OS version.

    Returns:
      float, the min supported OS version.
    """
    if not self._min_os_version:
      if self.profile_plist_obj:
        try:
          min_os_version_str = self.profile_plist_obj.GetPlistField(
              'minRuntimeVersion')
          self._min_os_version = _extra_os_version(min_os_version_str)
        except ios_errors.PlistError:
          self._min_os_version = 12.0
      else:
        # Fallback for newer Xcode where profile.plist is missing.
        self._min_os_version = 12.0
    return self._min_os_version

  @property
  def max_os_version(self):
    """Gets the max supported OS version.

    Returns:
      float, the max supported OS version or None if it is not found.
    """
    if not self._max_os_version:
      # If the profile.plist does not have maxRuntimeVersion field, it means
      # it supports the max OS version of current iphonesimulator platform.
      try:
        if self.profile_plist_obj:
          max_os_version_str = self.profile_plist_obj.GetPlistField(
              'maxRuntimeVersion')
          self._max_os_version = _extra_os_version(max_os_version_str)
      except ios_errors.PlistError:
        return None
    return self._max_os_version


def _extra_os_version(os_version_str):
  """Extracts os version float value from a given string."""
  # Cut build version. E.g., cut 9.3.3 to 9.3.
  if os_version_str.count('.') > 1:
    os_version_str = os_version_str[:os_version_str.rfind('.')]
  # We need to round the os version string in the simulator profile. E.g.,
  # the maxRuntimeVersion of iPhone 5 is 10.255.255 and we could create iOS 10.3
  # for iPhone 5.
  return round(float(os_version_str), 1)
