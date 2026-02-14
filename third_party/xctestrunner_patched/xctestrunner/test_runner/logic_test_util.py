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

"""The helper classes to run logic test."""

import os
import subprocess
import sys

from xctestrunner.shared import ios_constants
from xctestrunner.shared import version_util
from xctestrunner.shared import xcode_info_util
from xctestrunner.test_runner import runner_exit_codes

_SIMCTL_ENV_VAR_PREFIX = 'SIMCTL_CHILD_'


def RunLogicTestOnSim(sim_id,
                      test_bundle_path,
                      env_vars=None,
                      args=None,
                      tests_to_run=None,
                      os_version=None):
  """Runs logic tests on the simulator. The output prints on system stdout.

  Args:
    sim_id: string, the id of the simulator.
    test_bundle_path: string, the path of the logic test bundle.
    env_vars: dict, the additionl environment variables passing to test's
        process.
    args: array, the additional arguments passing to test's process.
    tests_to_run: array, the format of each item is TestClass[/TestMethod].
        If it is empty, then runs with All methods.
    os_version: string, the OS version of the simulator.

  Returns:
    exit_code: A value of type runner_exit_codes.EXITCODE.

  Raises:
    ios_errors.SimError: The command to launch logic test has error.
  """
  simctl_env_vars = {}
  if env_vars:
    for key in env_vars:
      simctl_env_vars[_SIMCTL_ENV_VAR_PREFIX + key] = env_vars[key]
  simctl_env_vars['NSUnbufferedIO'] = 'YES'
  # When running tests on iOS 12.1 or earlier simulator under Xcode 11 or later,
  # it is required to add swift5 fallback libraries to environment variable.
  # See https://github.com/bazelbuild/rules_apple/issues/684 for context.
  if (xcode_info_util.GetXcodeVersionNumber() >= 1100 and
      os_version and
      version_util.GetVersionNumber(os_version) < 1220):
    key = _SIMCTL_ENV_VAR_PREFIX + 'DYLD_FALLBACK_LIBRARY_PATH'
    simctl_env_vars[key] = xcode_info_util.GetSwift5FallbackLibsDir()
  # We need to set the DEVELOPER_DIR to ensure xcrun works correctly
  developer_dir = os.environ.get('DEVELOPER_DIR')
  if developer_dir:
    simctl_env_vars['DEVELOPER_DIR'] = developer_dir
  command = [
      'xcrun', 'simctl', 'spawn', '-s', sim_id,
      xcode_info_util.GetXctestToolPath(ios_constants.SDK.IPHONESIMULATOR)]
  if args:
    command += args
  if not tests_to_run:
    tests_to_run_str = 'All'
  else:
    tests_to_run_str = ','.join(tests_to_run)

  return_code = subprocess.Popen(
      command + ['-XCTest', tests_to_run_str, test_bundle_path],
      env=simctl_env_vars, stdout=sys.stdout, stderr=subprocess.STDOUT).wait()
  if return_code != 0:
    return runner_exit_codes.EXITCODE.FAILED
  return runner_exit_codes.EXITCODE.SUCCEEDED
