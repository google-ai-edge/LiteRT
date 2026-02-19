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
"""The utility class for simulator."""

import json
import logging
import os
import pwd
import re
import shutil
import subprocess
import time

from xctestrunner.shared import ios_constants
from xctestrunner.shared import ios_errors
from xctestrunner.shared import plist_util
from xctestrunner.shared import xcode_info_util
from xctestrunner.simulator_control import simtype_profile

_SIMULATOR_STATES_MAPPING = {
    0: ios_constants.SimState.CREATING,
    1: ios_constants.SimState.SHUTDOWN,
    3: ios_constants.SimState.BOOTED
}
_PREFIX_RUNTIME_ID = 'com.apple.CoreSimulator.SimRuntime.'
_SIM_OPERATION_MAX_ATTEMPTS = 3
_SIMCTL_MAX_ATTEMPTS = 2
_SIMULATOR_CREATING_TO_SHUTDOWN_TIMEOUT_SEC = 10
_SIMULATOR_BOOTED_TIMEOUT_SEC = 10
_SIMULATOR_SHUTDOWN_TIMEOUT_SEC = 30
_SIM_ERROR_RETRY_INTERVAL_SEC = 2
_SIM_CHECK_STATE_INTERVAL_SEC = 0.5
_PATTERN_APP_CRASH_ON_SIM = (
    r'com\.apple\.CoreSimulator\.SimDevice\.[A-Z0-9\-]+(.+) '
    r'\(UIKitApplication:%s(.+)\): Service exited '
    '(due to (signal|Terminated|Killed|Abort trap)|with abnormal code)')
_PATTERN_XCTEST_PROCESS_CRASH_ON_SIM = (
    r'com\.apple\.CoreSimulator\.SimDevice\.[A-Z0-9\-]+(.+) '
    r'\((.+)xctest\[[0-9]+\]\): Service exited '
    '(due to (signal|Terminated|Killed|Abort trap)|with abnormal code)')
_PATTERN_CORESIMULATOR_CRASH = (
    r'com\.apple\.CoreSimulator\.SimDevice\.[A-Z0-9\-]+(.+) '
    r'\(com\.apple\.CoreSimulator(.+)\): Service exited due to ')


class Simulator(object):
  """The object for simulator in MacOS."""

  def __init__(self, simulator_id):
    """Constructor of Simulator object.

    Args:
      simulator_id: string, the identity of the simulator.
    """
    self._simulator_id = simulator_id
    self._simulator_root_dir = None
    self._simulator_log_root_dir = None
    self._device_plist_object = None

  @property
  def simulator_id(self):
    if not self._simulator_id:
      raise ios_errors.SimError(
          'The simulator has not been created or has been deleted.')
    return self._simulator_id

  @property
  def simulator_system_log_path(self):
    return os.path.join(self.simulator_log_root_dir, 'system.log')

  @property
  def simulator_root_dir(self):
    """Gets the simulator's root directory."""
    if not self._simulator_root_dir:
      home_dir = pwd.getpwuid(os.geteuid()).pw_dir
      self._simulator_root_dir = os.path.join(
          '%s/Library/Developer/CoreSimulator/Devices/%s' %
          (home_dir, self.simulator_id))
    return self._simulator_root_dir

  @property
  def simulator_log_root_dir(self):
    """Gets the root directory of the simulator's logs."""
    if not self._simulator_log_root_dir:
      home_dir = pwd.getpwuid(os.geteuid()).pw_dir
      self._simulator_log_root_dir = os.path.join(
          '%s/Library/Logs/CoreSimulator/%s' % (home_dir, self.simulator_id))
    return self._simulator_log_root_dir

  @property
  def device_plist_object(self):
    """Gets the plist_util.Plist object of device.plist of the simulator.

    Returns:
      a plist_util.Plist object of device.plist of the simulator or None when
      the simulator does not exist or is being created.
    """
    if not self._device_plist_object:
      device_plist_path = os.path.join(self.simulator_root_dir, 'device.plist')
      if not os.path.exists(device_plist_path):
        return None
      self._device_plist_object = plist_util.Plist(device_plist_path)
    return self._device_plist_object

  def Boot(self):
    """Boots the simulator as asynchronously.

    Returns:
      A subprocess.Popen object of the boot process.
    """
    RunSimctlCommand(['xcrun', 'simctl', 'boot', self.simulator_id])
    self.WaitUntilStateBooted()
    logging.info('The simulator %s is booted.', self.simulator_id)

  def BootStatus(self):
    """Monitor the simulator boot status asynchronously.

    Returns:
      A subprocess.Popen object of the boot status process.
    """
    return subprocess.Popen(
        ['xcrun', 'simctl', 'bootstatus', self.simulator_id, '-b'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8')

  def Shutdown(self):
    """Shuts down the simulator."""
    sim_state = self.GetSimulatorState()
    if sim_state == ios_constants.SimState.SHUTDOWN:
      logging.info('Simulator %s has already shut down.', self.simulator_id)
      return
    if sim_state == ios_constants.SimState.CREATING:
      raise ios_errors.SimError(
          'Can not shut down the simulator in state CREATING.')
    logging.info('Shutting down simulator %s.', self.simulator_id)
    try:
      RunSimctlCommand(['xcrun', 'simctl', 'shutdown', self.simulator_id])
    except ios_errors.SimError as e:
      if 'Unable to shutdown device in current state: Shutdown' in str(e):
        logging.info('Simulator %s has already shut down.', self.simulator_id)
        return
      raise ios_errors.SimError('Failed to shutdown simulator %s: %s' %
                                (self.simulator_id, str(e)))
    self.WaitUntilStateShutdown()
    logging.info('Shut down simulator %s.', self.simulator_id)

  def Delete(self, asynchronously=True):
    """Deletes the simulator.

    The simulator state should be SHUTDOWN when deleting it. Otherwise, it will
    raise exception.

    Args:
      asynchronously: whether deleting the simulator asynchronously.
    Raises:
      ios_errors.SimError: The simulator's state is not SHUTDOWN.
    """
    command = ['xcrun', 'simctl', 'delete', self.simulator_id]
    if asynchronously:
      logging.info('Deleting simulator %s asynchronously.', self.simulator_id)
      subprocess.Popen(
          command,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          preexec_fn=os.setpgrp)
    else:
      try:
        RunSimctlCommand(command)
        logging.info('Deleted simulator %s.', self.simulator_id)
      except ios_errors.SimError as e:
        raise ios_errors.SimError('Failed to delete simulator %s: %s' %
                                  (self.simulator_id, str(e)))
    # The delete command won't delete the simulator log directory.
    if os.path.exists(self.simulator_log_root_dir):
      shutil.rmtree(self.simulator_log_root_dir, ignore_errors=True)
    self._simulator_id = None

  def FetchLogToFile(self, output_file_path, start_time=None, end_time=None):
    """Gets simulator log via running `log` tool on simulator.

    Args:
      output_file_path: string, the path of the stdout file.
      start_time: datetime, the start time of the simulatro log.
      end_time: datetime, the end time of the simulatro log.
    """
    command = [
        'xcrun', 'simctl', 'spawn', self._simulator_id, 'log', 'show',
        '--style', 'syslog'
    ]
    if start_time:
      command.extend(('--start', start_time.strftime('%Y-%m-%d %H:%M:%S')))
    if end_time:
      command.extend(('--end', end_time.strftime('%Y-%m-%d %H:%M:%S')))
    with open(output_file_path, 'w') as stdout_file:
      try:
        subprocess.Popen(command, stdout=stdout_file, stderr=subprocess.STDOUT)
      except ios_errors.SimError as e:
        raise ios_errors.SimError('Failed to get log on simulator %s: %s' %
                                  (self.simulator_id, str(e)))

  def GetAppDocumentsPath(self, app_bundle_id):
    """Gets the path of the app's Documents directory."""
    try:
      app_data_container = RunSimctlCommand([
          'xcrun', 'simctl', 'get_app_container', self._simulator_id,
          app_bundle_id, 'data'
      ])
      return os.path.join(app_data_container, 'Documents')
    except ios_errors.SimError as e:
      raise ios_errors.SimError(
          'Failed to get data container of the app %s in simulator %s: %s' %
          (app_bundle_id, self._simulator_id, str(e)))

    apps_dir = os.path.join(self.simulator_root_dir,
                            'data/Containers/Data/Application')
    for sub_dir_name in os.listdir(apps_dir):
      container_manager_plist = plist_util.Plist(
          os.path.join(apps_dir, sub_dir_name,
                       '.com.apple.mobile_container_manager.metadata.plist'))
      current_app_bundle_id = container_manager_plist.GetPlistField(
          'MCMMetadataIdentifier')
      if current_app_bundle_id == app_bundle_id:
        return os.path.join(apps_dir, sub_dir_name, 'Documents')
    raise ios_errors.SimError(
        'Failed to get Documents directory of the app %s in simulator %s' %
        (app_bundle_id, self._simulator_id))

  def IsAppInstalled(self, app_bundle_id):
    """Checks if the simulator has installed the app with given bundle id."""
    try:
      RunSimctlCommand([
          'xcrun', 'simctl', 'get_app_container', self._simulator_id,
          app_bundle_id
      ])
      return True
    except ios_errors.SimError:
      return False

  def WaitUntilStateBooted(self, timeout_sec=_SIMULATOR_BOOTED_TIMEOUT_SEC):
    """Waits until the simulator state becomes BOOTED.

    Args:
      timeout_sec: int, timeout of waiting simulator state for becoming BOOTED
        in seconds.

    Raises:
      ios_errors.SimError: when it is timeout to wait the simulator state
          becomes BOOTED.
    """
    start_time = time.time()
    while start_time + timeout_sec >= time.time():
      time.sleep(_SIM_CHECK_STATE_INTERVAL_SEC)
      if self.GetSimulatorState() == ios_constants.SimState.BOOTED:
        return
    raise ios_errors.SimError('Timeout to wait for simulator booted in %ss.' %
                              timeout_sec)

  def WaitUntilStateShutdown(self, timeout_sec=_SIMULATOR_SHUTDOWN_TIMEOUT_SEC):
    """Waits until the simulator state becomes SHUTDOWN.

    Args:
      timeout_sec: int, timeout of waiting simulator state for becoming SHUTDOWN
        in seconds.

    Raises:
      ios_errors.SimError: when it is timeout to wait the simulator state
          becomes SHUTDOWN.
    """
    start_time = time.time()
    while start_time + timeout_sec >= time.time():
      time.sleep(_SIM_CHECK_STATE_INTERVAL_SEC)
      if self.GetSimulatorState() == ios_constants.SimState.SHUTDOWN:
        return
    raise ios_errors.SimError('Timeout to wait for simulator shutdown in %ss.' %
                              timeout_sec)

  def GetSimulatorState(self):
    """Gets the state of the simulator in real time.

    Returns:
      shared.ios_constants.SimState, the state of the simulator.

    Raises:
      ios_errors.SimError: The state can not be recognized.
    """
    if self.device_plist_object is None:
      return ios_constants.SimState.CREATING
    state_num = self.device_plist_object.GetPlistField('state')
    if state_num not in _SIMULATOR_STATES_MAPPING.keys():
      logging.warning('The state %s of simulator %s can not be recognized.',
                      state_num, self.simulator_id)
      return ios_constants.SimState.UNKNOWN
    return _SIMULATOR_STATES_MAPPING[state_num]


def CreateNewSimulator(device_type=None, os_version=None, name_prefix=None):
  """Creates a new simulator according to arguments.

  If neither device_type nor os_version is given, will use the latest iOS
  version and latest iPhone type.
  If os_version is given but device_type is not, will use latest iPhone type
  according to the OS version limitation. E.g., if the given os_version is 9.3,
  the latest simulator type is iPhone 6s Plus. Because the min OS version of
  iPhone 7 is 10.0.
  If device_type is given but os_version is not, will use the min value
  between max OS version of the simulator type and current latest OS version.
  E.g., if the given device_type is iPhone 5 and latest OS version is 10.3,
  will use 10.2. Because the max OS version of iPhone 5 is 10.2.

  Args:
    device_type: string, device type of the new simulator. The value corresponds
      to the output of `xcrun simctl list devicetypes`. E.g., iPhone 6, iPad
      Air, etc.
    os_version: string, OS version of the new simulator. The format is
      {major}.{minor}, such as 9.3, 10.2.
    name_prefix: string, name prefix of the new simulator. By default, it is
      "New".

  Returns:
     a tuple with four items:
        string, id of the new simulator.
        string, simulator device type of the new simulator.
        string, OS version of the new simulator.
        string, name of the new simulator.

  Raises:
    ios_errors.SimError: when failed to create new simulator.
    ios_errors.IllegalArgumentError: when the given argument is invalid.
  """
  if not device_type:
    os_type = ios_constants.OS.IOS
  else:
    _ValidateSimulatorType(device_type)
    os_type = GetOsType(device_type)
  if not os_version:
    os_version = GetLastSupportedSimOsVersion(os_type, device_type=device_type)
  else:
    supported_sim_os_versions = GetSupportedSimOsVersions(os_type)
    if os_version not in supported_sim_os_versions:
      raise ios_errors.IllegalArgumentError(
          'The simulator os version %s is not supported. Supported simulator '
          'os versions are %s.' % (os_version, supported_sim_os_versions))
  if not device_type:
    device_type = GetLastSupportedIphoneSimType(os_version)
  else:
    _ValidateSimulatorTypeWithOsVersion(device_type, os_version)
  if not name_prefix:
    name_prefix = 'New'
  name = '%s-%s-%s' % (name_prefix, device_type, os_version)

  # Example
  # Runtime ID of iOS 10.2: com.apple.CoreSimulator.SimRuntime.iOS-10-2
  runtime_id = _PREFIX_RUNTIME_ID + os_type + '-' + os_version.replace('.', '-')
  logging.info('Creating a new simulator:\nName: %s\nOS: %s %s\nType: %s', name,
               os_type, os_version, device_type)
  for i in range(0, _SIM_OPERATION_MAX_ATTEMPTS):
    try:
      new_simulator_id = RunSimctlCommand(
          ['xcrun', 'simctl', 'create', name, device_type, runtime_id])
    except ios_errors.SimError as e:
      raise ios_errors.SimError('Failed to create simulator: %s' % str(e))
    new_simulator_obj = Simulator(new_simulator_id)
    # After creating a new simulator, its state is CREATING. When the
    # simulator's state becomes SHUTDOWN, the simulator is created.
    try:
      new_simulator_obj.WaitUntilStateShutdown(
          _SIMULATOR_CREATING_TO_SHUTDOWN_TIMEOUT_SEC)
      logging.info('Created new simulator %s.', new_simulator_id)
      return new_simulator_id, device_type, os_version, name
    except ios_errors.SimError as error:
      logging.debug('Failed to create simulator %s: %s.', new_simulator_id,
                    error)
      logging.debug('Deleted half-created simulator %s.', new_simulator_id)
      new_simulator_obj.Delete()
      if i != _SIM_OPERATION_MAX_ATTEMPTS - 1:
        logging.debug('Will sleep %ss and retry again.',
                      _SIM_ERROR_RETRY_INTERVAL_SEC)
        # If the simulator's state becomes SHUTDOWN, there may be something
        # wrong in CoreSimulatorService. Sleeps a short interval(2s) can help
        # reduce flakiness.
        time.sleep(_SIM_ERROR_RETRY_INTERVAL_SEC)
  raise ios_errors.SimError('Failed to create simulator in %d attempts.' %
                            _SIM_OPERATION_MAX_ATTEMPTS)


def GetSupportedSimDeviceTypes(os_type=None):
  """Gets the name list of supported simulator device types of given OS type.

  If os_type is not provided, it will return all supported simulator device
  types. The names are got from command result of `xcrun simctl list devices`.
  So some simulator device types' names may be different in different Xcode.
  E.g., the name of iPad Pro (12.9-inch) in Xcode 7.2.1 is "iPad Pro", but it is
  "iPad Pro (12.9-inch)" in Xcode 8+.

  Args:
    os_type: shared.ios_constants.OS, OS type of simulator, such as iOS,
      watchOS, tvOS.

  Returns:
    a list of string, each item is a simulator device type.
    E.g., ["iPhone 5", "iPhone 6 Plus"]
  """
  # Example output:
  # {
  #   "devicetypes" : [
  #    {
  #      "name" : "iPhone 5",
  #      "identifier" : "com.apple.CoreSimulator.SimDeviceType.iPhone-5"
  #    }
  #   ]
  # }
  #
  # See more examples in testdata/simctl_list_devicetypes.json
  sim_types_infos_json = json.loads(
      RunSimctlCommand(('xcrun', 'simctl', 'list', 'devicetypes', '-j')))
  sim_types = []
  for sim_types_info in sim_types_infos_json['devicetypes']:
    sim_type = sim_types_info['name']
    if (os_type is None or
        (os_type == ios_constants.OS.IOS and sim_type.startswith('i')) or
        (os_type == ios_constants.OS.TVOS and 'TV' in sim_type) or
        (os_type == ios_constants.OS.WATCHOS and 'Watch' in sim_type)):
      sim_types.append(sim_type)
  return sim_types


def GetLastSupportedIphoneSimType(os_version):
  """"Gets the last supported iPhone simulator type of the given OS version.

  Currently, the last supported iPhone simulator type is the last iPhone from
  the output of `xcrun simctl list devicetypes`.

  Args:
    os_version: string, OS version of the new simulator. The format is
      {major}.{minor}, such as 9.3, 10.2.

  Returns:
    a string, the last supported iPhone simulator type.

  Raises:
    ios_errors.SimError: when there is no supported iPhone simulator type.
  """
  supported_sim_types = GetSupportedSimDeviceTypes(ios_constants.OS.IOS)
  os_version_float = float(os_version)
  for sim_type in supported_sim_types:
    if sim_type.startswith('iPhone'):
      min_os_version = simtype_profile.SimTypeProfile(sim_type).min_os_version
      if os_version_float >= min_os_version:
        return sim_type
  raise ios_errors.SimError('Can not find supported iPhone simulator type.')


def GetSupportedSimOsVersions(os_type=ios_constants.OS.IOS):
  """Gets the supported version of given simulator OS type.

  Args:
    os_type: shared.ios_constants.OS, OS type of simulator, such as iOS,
      watchOS, tvOS.

  Returns:
    a list of string, each item is an OS version number. E.g., ["10.1", "11.0"]
  """
  if os_type is None:
    os_type = ios_constants.OS.IOS
  # Example output:
  # {
  # "runtimes" : [
  #   {
  #     "bundlePath" : "\/Applications\/Xcode10.app\/Contents\/Developer\
  #                     /Platforms\/iPhoneOS.platform\/Developer\/Library\
  #                     /CoreSimulator\/Profiles\/Runtimes\/iOS.simruntime",
  #     "availabilityError" : "",
  #     "buildversion" : "16A366",
  #     "availability" : "(available)",
  #     "isAvailable" : true,
  #     "identifier" : "com.apple.CoreSimulator.SimRuntime.iOS-12-0",
  #     "version" : "12.0",
  #     "name" : "iOS 12.0"
  #   }
  # }
  # See more examples in testdata/simctl_list_runtimes.json
  xcode_version_num = xcode_info_util.GetXcodeVersionNumber()
  sim_runtime_infos_json = json.loads(
      RunSimctlCommand(('xcrun', 'simctl', 'list', 'runtimes', '-j')))
  sim_versions = []
  for sim_runtime_info in sim_runtime_infos_json['runtimes']:
    # Normally, the json does not contain unavailable runtimes. To be safe,
    # also checks the 'availability' field.
    if 'availability' in sim_runtime_info and sim_runtime_info[
        'availability'].find('unavailable') >= 0:
      continue
    elif 'isAvailable' in sim_runtime_info and not sim_runtime_info[
        'isAvailable']:
      continue

    listed_os_type, listed_os_version = sim_runtime_info['name'].split(' ', 1)
    if listed_os_type == os_type:
      # `bundlePath` key may not exist in the old Xcode/macOS version.
      if 'bundlePath' in sim_runtime_info:
        runtime_path = sim_runtime_info['bundlePath']
        info_plist_object = plist_util.Plist(
            os.path.join(runtime_path, 'Contents/Info.plist'))
        min_xcode_version_num = int(info_plist_object.GetPlistField('DTXcode'))
        if xcode_version_num >= min_xcode_version_num:
          sim_versions.append(listed_os_version)
      else:
        if os_type == ios_constants.OS.IOS:
          ios_major_version, ios_minor_version = listed_os_version.split('.', 1)
          # Ingores the potential build version
          ios_minor_version = ios_minor_version[0]
          ios_version_num = int(ios_major_version) * 100 + int(
              ios_minor_version) * 10
          # One Xcode version always maps to one max simulator's iOS version.
          # The rules is almost max_sim_ios_version <= xcode_version + 200.
          # E.g., Xcode 8.3.1/8.3.3 maps to iOS 10.3, Xcode 7.3.1 maps to iOS
          # 9.3.
          if ios_version_num > xcode_version_num + 200:
            continue
        sim_versions.append(listed_os_version)
  return sim_versions


def GetLastSupportedSimOsVersion(os_type=ios_constants.OS.IOS,
                                 device_type=None):
  """Gets the last supported version of given arguments.

  If device_type is given, will return the last supported OS version of the
  device type. Otherwise, will return the last supported OS version of the
  OS type.

  Args:
    os_type: shared.ios_constants.OS, OS type of simulator, such as iOS,
      watchOS, tvOS.
    device_type: string, device type of the new simulator. The value corresponds
      to the output of `xcrun simctl list devicetypes`. E.g., iPhone 6, iPad
      Air, etc.

  Returns:
    a string, the last supported version.

  Raises:
    ios_errors.SimError: when there is no supported OS version of the given OS.
    ios_errors.IllegalArgumentError: when the supported OS version can not match
        the given simulator type.
  """
  supported_os_versions = GetSupportedSimOsVersions(os_type)
  if not supported_os_versions:
    raise ios_errors.SimError('Can not find supported OS version of %s.' %
                              os_type)
  if not device_type:
    return supported_os_versions[-1]

  max_os_version = simtype_profile.SimTypeProfile(device_type).max_os_version
  # The supported os versions will be from latest to older after reverse().
  supported_os_versions.reverse()
  if not max_os_version:
    return supported_os_versions[0]

  for os_version in supported_os_versions:
    if float(os_version) <= max_os_version:
      return os_version
  raise ios_errors.IllegalArgumentError(
      'The supported OS version %s can not match simulator type %s. Because '
      'its max OS version is %s' %
      (supported_os_versions, device_type, max_os_version))


def GetOsType(device_type):
  """Gets the OS type of the given simulator.

  This method can not work fine if the device_type is invalid. Please calls
  simulator_util.ValidateSimulatorType(device_type, os_version) to validate
  it first.

  Args:
    device_type: string, device type of the new simulator. The value corresponds
      to the output of `xcrun simctl list devicetypes`. E.g., iPhone 6, iPad
      Air, etc.

  Returns:
    shared.ios_constants.OS.

  Raises:
    ios_errors.IllegalArgumentError: when the OS type of the given simulator
        device type can not be recognized.
  """
  if device_type.startswith('i'):
    return ios_constants.OS.IOS
  if 'TV' in device_type:
    return ios_constants.OS.TVOS
  if 'Watch' in device_type:
    return ios_constants.OS.WATCHOS
  raise ios_errors.IllegalArgumentError(
      'Failed to recognize the os type for simulator device type %s.' %
      device_type)


def _ValidateSimulatorType(device_type):
  """Checks if the simulator type is valid.

  Args:
    device_type: string, device type of the new simulator. The value corresponds
      to the output of `xcrun simctl list devicetypes`. E.g., iPhone 6, iPad
      Air, etc.

  Raises:
    ios_errors.IllegalArgumentError: when the given simulator device type is
    invalid.
  """
  supported_sim_device_types = GetSupportedSimDeviceTypes()
  if device_type not in supported_sim_device_types:
    raise ios_errors.IllegalArgumentError(
        'The simulator device type %s is not supported. Supported simulator '
        'device types are %s.' % (device_type, supported_sim_device_types))


def _ValidateSimulatorTypeWithOsVersion(device_type, os_version):
  """Checks if the simulator type with the given os version is valid.

  Args:
    device_type: string, device type of the new simulator. The value corresponds
      to the output of `xcrun simctl list devicetypes`. E.g., iPhone 6, iPad
      Air, etc.
    os_version: string, OS version of the new simulator. The format is
      {major}.{minor}, such as 9.3, 10.2.

  Raises:
    ios_errors.IllegalArgumentError: when the given simulator device type can
        not match the given OS version.
  """
  os_version_float = float(os_version)
  sim_profile = simtype_profile.SimTypeProfile(device_type)
  min_os_version = sim_profile.min_os_version
  if min_os_version > os_version_float:
    raise ios_errors.IllegalArgumentError(
        'The min OS version of %s is %f. But current OS version is %s' %
        (device_type, min_os_version, os_version))
  max_os_version = sim_profile.max_os_version
  if max_os_version:
    if max_os_version < os_version_float:
      raise ios_errors.IllegalArgumentError(
          'The max OS version of %s is %f. But current OS version is %s' %
          (device_type, max_os_version, os_version))


def QuitSimulatorApp():
  """Quits the Simulator.app."""
  subprocess.Popen(['killall', 'Simulator'],
                   stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT)


def IsAppFailedToLaunchOnSim(sim_sys_log, app_bundle_id=''):
  """Checks if the app failed to launch on simulator.

  If app_bundle_id is not provided, will check if any UIKitApplication failed
  to launch on simulator.

  Args:
    sim_sys_log: string, the content of the simulator's system.log.
    app_bundle_id: string, the bundle id of the app.

  Returns:
    True if the app failed to launch on simulator.
  """
  pattern = re.compile(_PATTERN_APP_CRASH_ON_SIM % app_bundle_id)
  return pattern.search(sim_sys_log) is not None


def IsXctestFailedToLaunchOnSim(sim_sys_log):
  """Checks if the xctest process failed to launch on simulator.

  Args:
    sim_sys_log: string, the content of the simulator's system.log.

  Returns:
    True if the xctest process failed to launch on simulator.
  """
  pattern = re.compile(_PATTERN_XCTEST_PROCESS_CRASH_ON_SIM)
  return pattern.search(sim_sys_log) is not None


def IsCoreSimulatorCrash(sim_sys_log):
  """Checks if CoreSimulator crashes.

  Args:
    sim_sys_log: string, the content of the simulator's system.log.

  Returns:
    True if the CoreSimulator crashes.
  """
  pattern = re.compile(_PATTERN_CORESIMULATOR_CRASH)
  return pattern.search(sim_sys_log) is not None


def RunSimctlCommand(command):
  """Runs simctl command."""
  for i in range(_SIMCTL_MAX_ATTEMPTS):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8')
    stdout, stderr = process.communicate()
    all_output = '\n'.join([stdout, stderr])
    output = stdout.strip()
    if process.poll() != 0:
      if (i < (_SIMCTL_MAX_ATTEMPTS - 1) and
          ios_constants.CORESIMULATOR_INTERRUPTED_ERROR in all_output):
        continue
      raise ios_errors.SimError(output)
    return output
