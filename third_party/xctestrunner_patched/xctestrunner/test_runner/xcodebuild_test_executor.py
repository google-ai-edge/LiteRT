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

"""Helper class for running test by xcodebuild tool."""

import io
import logging
import os
import random
import re
import shutil
import subprocess
import threading
import time

from xctestrunner.shared import ios_constants
from xctestrunner.shared import ios_errors
from xctestrunner.shared import xcode_info_util
from xctestrunner.simulator_control import simulator_util
from xctestrunner.test_runner import runner_exit_codes


_XCODEBUILD_TEST_STARTUP_TIMEOUT_SEC = 150
_SIM_TEST_MAX_ATTEMPTS = 3
_DEVICE_TEST_MAX_ATTEMPTS = 2
_TAIL_SIM_LOG_LINE = 200
_BACKGROUND_TEST_RUNNER_ERROR = 'Failed to background test runner'
_PROCESS_EXISTED_OR_CRASHED_ERROR = ('The process did launch, but has since '
                                     'exited or crashed.')
_REQUEST_DENIED_ERROR = ('The request was denied by service delegate '
                         '(SBMainWorkspace) for reason')
_APP_UNKNOWN_TO_FRONTEND_PATTERN = re.compile(
    'Application ".*" is unknown to FrontBoard.')
_INIT_SIM_SERVICE_ERROR = 'Failed to initiate service connection to simulator'
_DEVICE_TYPE_WAS_NULL_PATTERN = re.compile(
    'DTDeviceKit: deviceType from .* was NULL')
_TOO_MANY_INSTANCES_ALREADY_RUNNING = ('Too many instances of this service are '
                                       'already running.')
_LOST_CONNECTION_ERROR = 'Lost connection to testmanagerd'
_LOST_CONNECTION_TO_DTSERVICEHUB_ERROR = 'Lost connection to DTServiceHub'
_DEVICE_NO_LONGER_CONNECTED = 'This device is no longer connected'
_UNABLE_FIND_DEVICE_IDENTIFIER = 'Unable to find device with identifier'
_BUNDLE_DAMAGED = 'The bundle is damaged or missing necessary resources.'


class CheckXcodebuildStuckThread(threading.Thread):
  """The thread class that checking if xcodebuild process stucks.

  The thread will stop gracefully when it is called Terminate() or the given
  process is terminated or reaching the given timeout. If reaching the given
  timeout, this thread will also kill the given xcodebuild process.
  """

  def __init__(self, xcodebuild_test_popen, startup_timeout_sec):
    super(CheckXcodebuildStuckThread, self).__init__()
    self._xcodebuild_test_popen = xcodebuild_test_popen
    self._terminate = False
    self._is_xcodebuild_stuck = False
    self._startup_timeout_sec = startup_timeout_sec

  def run(self):
    start_time = time.time()
    while (not self._terminate and
           start_time + self._startup_timeout_sec >= time.time() and
           self._xcodebuild_test_popen.poll() is None):
      time.sleep(2)
    if (not self._terminate and
        start_time + self._startup_timeout_sec < time.time()):
      logging.warning(
          'The xcodebuild command got stuck and has not started test in %d. '
          'Will kill the command directly.', self._startup_timeout_sec)
      self._is_xcodebuild_stuck = True
      self._xcodebuild_test_popen.terminate()

  def Terminate(self):
    """Terminates this thread."""
    self._terminate = True

  @property
  def is_xcodebuild_stuck(self):
    """If the xcodebuild test command is stuck in timeout."""
    return self._is_xcodebuild_stuck


class XcodebuildTestExecutor(object):
  """A class to execute testing command by xcodebuild tool."""

  # TODO(albertdai): change the argument succeeded_signal and failed_signal to
  # be required.
  def __init__(self,
               command,
               sdk=None,
               test_type=None,
               device_id=None,
               succeeded_signal=None,
               failed_signal=None,
               app_bundle_id=None,
               startup_timeout_sec=None):
    """Initializes the XcodebuildTestExecutor object.

    The optional argument sdk, test_type and device_id can provide more
    specific handler.

    Args:
      command: array, the testing command of `xcodebuild`.
      sdk: ios_constants.SDK, the sdk of the target device to run test.
      test_type: ios_constants.TestType, the type of the test.
      device_id: string, the id of the device to run test.
      succeeded_signal: string, the signal of command succeeded.
      failed_signal: string, the signal of command failed.
      app_bundle_id: string, the bundle id of the app under test.
      startup_timeout_sec: int, seconds until the xcodebuild is deemed stuck.
    """
    self._command = command
    self._sdk = sdk
    self._test_type = test_type
    self._device_id = device_id
    self._succeeded_signal = succeeded_signal
    self._failed_signal = failed_signal
    self._app_bundle_id = app_bundle_id
    self._startup_timeout_sec = (
        startup_timeout_sec or _XCODEBUILD_TEST_STARTUP_TIMEOUT_SEC)

  def Execute(self, return_output=True, result_bundle_path=None):
    """Executes the xcodebuild test command.

    Args:
      return_output: bool, whether save output in the execution result.
      result_bundle_path: string, path of the result bundle.

    Returns:
      a tuple of two fields:
        exit_code: A value of type runner_exit_codes.EXITCODE.
        output: the output of xcodebuild test command or None if return_output
            is False.
    """
    run_env = dict(os.environ)
    run_env['NSUnbufferedIO'] = 'YES'
    max_attempts = 1
    sim_log_path = None
    if self._sdk == ios_constants.SDK.IPHONESIMULATOR:
      max_attempts = _SIM_TEST_MAX_ATTEMPTS
      if self._device_id:
        sim_log_path = simulator_util.Simulator(
            self._device_id).simulator_system_log_path
    elif self._sdk == ios_constants.SDK.IPHONEOS:
      max_attempts = _DEVICE_TEST_MAX_ATTEMPTS

    test_started = False
    test_succeeded = False
    test_failed = False

    for i in range(max_attempts):
      if result_bundle_path and os.path.exists(result_bundle_path):
        shutil.rmtree(result_bundle_path, ignore_errors=True)
      process = subprocess.Popen(
          self._command, env=run_env, stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT, encoding='ascii', errors='ignore')
      check_xcodebuild_stuck = CheckXcodebuildStuckThread(
          process, self._startup_timeout_sec)
      check_xcodebuild_stuck.start()
      output = io.StringIO()
      for stdout_line in process.stdout:
        if not test_started:
          # Terminates the CheckXcodebuildStuckThread when test has started
          # or XCTRunner.app has started.
          # But XCTRunner.app start does not mean test start.
          if ios_constants.TEST_STARTED_SIGNAL in stdout_line:
            test_started = True
            check_xcodebuild_stuck.Terminate()
          # Only terminate the check_xcodebuild_stuck thread when running on
          # iphonesimulator device. When running on iphoneos device, the
          # XCTRunner.app may not launch the test session sometimes
          # (error rate < 1%).
          if (self._test_type == ios_constants.TestType.XCUITEST and
              ios_constants.XCTRUNNER_STARTED_SIGNAL in stdout_line and
              self._sdk == ios_constants.SDK.IPHONESIMULATOR):
            check_xcodebuild_stuck.Terminate()
        else:
          if self._succeeded_signal and self._succeeded_signal in stdout_line:
            test_succeeded = True
          if self._failed_signal and self._failed_signal in stdout_line:
            test_failed = True

        print(stdout_line, flush=True, end='')
        # If return_output is false, the output is only used for checking error
        # cause and deleting cached files (_DeleteTestCacheFileDirs method).
        if return_output or not test_started:
          output.write(stdout_line)

      try:
        if test_started:
          if test_succeeded:
            exit_code = runner_exit_codes.EXITCODE.SUCCEEDED
          elif test_failed:
            exit_code = runner_exit_codes.EXITCODE.FAILED
          else:
            exit_code = runner_exit_codes.EXITCODE.ERROR
          return exit_code, output.getvalue() if return_output else None

        check_xcodebuild_stuck.Terminate()
        if check_xcodebuild_stuck.is_xcodebuild_stuck:
          return self._GetResultForXcodebuildStuck(output, return_output)

        output_str = output.getvalue()
        # Don't need to retry the case for the damaged test bundle.
        if _BUNDLE_DAMAGED in output_str:
          return (runner_exit_codes.EXITCODE.TEST_NOT_START,
                  output_str if return_output else None)

        if self._sdk == ios_constants.SDK.IPHONEOS:
          if (self._NeedRetryForDeviceTesting(output_str) and
              i < max_attempts - 1):
            logging.warning(
                'Failed to launch test on the device. Will relaunch again '
                'after 5s.'
            )
            time.sleep(5)
            continue
          if _TOO_MANY_INSTANCES_ALREADY_RUNNING in output_str:
            return (runner_exit_codes.EXITCODE.NEED_REBOOT_DEVICE,
                    output_str if return_output else None)

        if self._sdk == ios_constants.SDK.IPHONESIMULATOR:
          if self._NeedRebootSim(output_str):
            return (runner_exit_codes.EXITCODE.NEED_REBOOT_DEVICE,
                    output_str if return_output else None)
          if self._NeedRecreateSim(output_str):
            return (runner_exit_codes.EXITCODE.NEED_RECREATE_SIM,
                    output_str if return_output else None)

          # The following error can be fixed by relaunching the test again.
          try:
            if sim_log_path and os.path.exists(sim_log_path):
              # Sleeps short time. Then the tail simulator log can get more log.
              time.sleep(0.5)
              tail_sim_log = _ReadFileTailInShell(
                  sim_log_path, _TAIL_SIM_LOG_LINE)
              if (self._test_type == ios_constants.TestType.LOGIC_TEST and
                  simulator_util.IsXctestFailedToLaunchOnSim(tail_sim_log) or
                  self._test_type != ios_constants.TestType.LOGIC_TEST and
                  simulator_util.IsAppFailedToLaunchOnSim(tail_sim_log) or
                  simulator_util.IsCoreSimulatorCrash(tail_sim_log)):
                raise ios_errors.SimError('')
            if _PROCESS_EXISTED_OR_CRASHED_ERROR in output_str:
              raise ios_errors.SimError('')
            if ios_constants.CORESIMULATOR_INTERRUPTED_ERROR in output_str:
              # Sleep random[0,2] seconds to avoid race condition. It is a known
              # issue that CoreSimulatorService connection will be interrupted
              # if two simulators are booting at the same time.
              time.sleep(random.uniform(0, 2))
              raise ios_errors.SimError('')
            if (self._app_bundle_id and
                not simulator_util.Simulator(self._device_id).IsAppInstalled(
                    self._app_bundle_id)):
              raise ios_errors.SimError('')
          except ios_errors.SimError:
            if i < max_attempts - 1:
              logging.warning(
                  'Failed to launch test on simulator. Will relaunch again.')
              continue

        return (runner_exit_codes.EXITCODE.TEST_NOT_START,
                output_str if return_output else None)
      finally:
        _DeleteTestCacheFileDirs(output.getvalue(), self._sdk, self._test_type)

  def _GetResultForXcodebuildStuck(self, output, return_output):
    """Gets the execution result for the xcodebuild stuck case."""
    error_message = ('xcodebuild command can not launch test on '
                     'device/simulator in %ss.' % self._startup_timeout_sec)
    logging.error(error_message)
    output.write(error_message)
    output_str = output.getvalue()
    if self._sdk == ios_constants.SDK.IPHONEOS:
      return (runner_exit_codes.EXITCODE.NEED_REBOOT_DEVICE,
              output_str if return_output else None)
    return (runner_exit_codes.EXITCODE.TEST_NOT_START,
            output_str if return_output else None)

  def _NeedRebootSim(self, output_str):
    """Checks if need reboot the simulator."""
    if (self._test_type == ios_constants.TestType.XCUITEST and
        _BACKGROUND_TEST_RUNNER_ERROR in output_str):
      return True

  def _NeedRecreateSim(self, output_str):
    """Checks if need recreate a new simulator."""
    if re.search(_APP_UNKNOWN_TO_FRONTEND_PATTERN, output_str):
      return True
    if _REQUEST_DENIED_ERROR in output_str:
      return True
    if _INIT_SIM_SERVICE_ERROR in output_str:
      return True
    return False

  def _NeedRetryForDeviceTesting(self, output_str):
    """Returns true if the device testing needs retry."""
    return (re.search(_DEVICE_TYPE_WAS_NULL_PATTERN, output_str) or
            _LOST_CONNECTION_ERROR in output_str or
            _LOST_CONNECTION_TO_DTSERVICEHUB_ERROR in output_str or
            _DEVICE_NO_LONGER_CONNECTED in output_str or
            _UNABLE_FIND_DEVICE_IDENTIFIER in output_str)


def _DeleteTestCacheFileDirs(xcodebuild_test_output, sdk, test_type):
  """Deletes the cache files of the test session according to arguments."""
  if sdk == ios_constants.SDK.IPHONEOS:
    max_cache_dir_num = 1
    if test_type == ios_constants.TestType.XCUITEST:
      # Because XCUITest will install two apps (app under test and
      # XCTRunner.app) on the device.
      max_cache_dir_num = 2
    test_cache_file_dirs = _FetchTestCacheFileDirs(xcodebuild_test_output,
                                                   max_cache_dir_num)
    for cache_dir in test_cache_file_dirs:
      if os.path.exists(cache_dir):
        logging.info('Removing cache files directory: %s', cache_dir)
        shutil.rmtree(cache_dir)


def _FetchTestCacheFileDirs(xcodebuild_test_output, max_dir_num=1):
  """Fetches the cache file directories of this test session.

  When using `xcodebuild` to run test on iOS real device, it will generate some
  cache files under
  DARWIN_USER_CACHE_DIR/com.apple.DeveloperTools/All/Xcode/EmbeddedAppDeltas.

  Args:
    xcodebuild_test_output: string, the `xcodebuild test` output of this test
        session.
    max_dir_num: int, the max number of test cache files potentially.

  Returns:
    an array of this test's EmbeddedAppDeltas directories.
  """
  xcode_cache_dir = xcode_info_util.GetXcodeEmbeddedAppDeltasDir()
  pattern = re.compile('(%s/[a-z0-9]+)/' % re.escape(xcode_cache_dir))
  start_pos = 0
  cache_file_dirs = set()
  while (start_pos < len(xcodebuild_test_output) and
         len(cache_file_dirs) < max_dir_num):
    match = pattern.search(xcodebuild_test_output, start_pos)
    if not match:
      break
    cache_file_dirs.add(match.group(1))
    start_pos = match.end(1)
  return cache_file_dirs


def _ReadFileTailInShell(file_path, line):
  """Tails the file in the last several lines."""
  return subprocess.check_output(['tail', '-%d' % line,
                                  file_path]).decode('utf-8')
