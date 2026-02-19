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

"""The module to run XCTEST based tests."""

import logging
import os
import shutil
import subprocess
import tempfile

from xctestrunner.shared import bundle_util
from xctestrunner.shared import ios_constants
from xctestrunner.shared import ios_errors
from xctestrunner.shared import xcode_info_util
from xctestrunner.test_runner import logic_test_util
from xctestrunner.test_runner import xcresult_util
from xctestrunner.test_runner import xctestrun


class XctestSession(object):
  """The class that runs XCTEST based tests."""

  def __init__(self, sdk, device_arch, work_dir=None, output_dir=None):
    """Initializes the XctestSession object.

    If work_dir is not provdied, will create a temp direcotry to be work_dir and
    remove it after test ends. If output_dir is not provided, will create a temp
    direcotry to be output_dir and remove it after test ends.

    Args:
      sdk: ios_constants.SDK. The sdk of the target device.
      device_arch: ios_constants.ARCH. The architecture of the target device.
      work_dir: string, the working directory contains runfiles.
      output_dir: string, The directory where derived data will go, including:
          1) the detailed test session log which includes test output and the
          communication log between host machine and device;
          2) the screenshots of every test stages (XCUITest). If directory is
          specified, the directory will not be deleted after test ends.'
    """
    self._sdk = sdk
    self._device_arch = device_arch
    self._work_dir = work_dir
    self._delete_work_dir = True
    self._output_dir = output_dir
    self._delete_output_dir = True
    self._startup_timeout_sec = None
    self._destination_timeout_sec = None
    self._xctestrun_obj = None
    self._prepared = False
    self._keep_xcresult_data = True
    # The following fields are only for Logic Test.
    self._logic_test_bundle = None
    self._logic_test_env_vars = None
    self._logic_test_args = None
    self._logic_tests_to_run = None
    # The following fields are only for XCUITest.
    self._disable_uitest_auto_screenshots = True

  def __enter__(self):
    return self

  def __exit__(self, unused_type, unused_value, unused_traceback):
    """Deletes the temp directories."""
    self.Close()

  # TODO(albertdai): Support bundle id as the value of app_under_test and
  # test_bundle.
  def Prepare(self, app_under_test=None, test_bundle=None,
              xctestrun_file_path=None, test_type=None, signing_options=None):
    """Prepares the test session.

    If xctestrun_file is not provided, will use app under test and test bundle
    path to generate a new xctest file.

    Args:
      app_under_test: string, the path of the application to be tested. It can
          be .ipa or .app.
      test_bundle: string, the path of the test bundle to be tested. It can
          be .ipa or .xctest.
      xctestrun_file_path: string, the path of the xctestrun file. It is the
          configure file to launch test in Xcode 8+.
      test_type: ios_constants.TestType. The type of test bundle.
      signing_options: dict, the signing app options. See
          ios_constants.SIGNING_OPTIONS_JSON_HELP for details.

    Raises:
      ios_errors.IllegalArgumentError:
          1) the app under test/test bundle does not exist;
          2) the app under test/test bundle's extension is invaild.
    """
    if not signing_options:
      signing_options = {}

    if self._work_dir:
      if not os.path.exists(self._work_dir):
        os.mkdir(self._work_dir)
      self._work_dir = os.path.abspath(self._work_dir)
      self._delete_work_dir = False
    else:
      self._work_dir = tempfile.mkdtemp()
      self._delete_work_dir = True

    if self._output_dir:
      if not os.path.exists(self._output_dir):
        os.mkdir(self._output_dir)
      self._delete_output_dir = False
    else:
      self._output_dir = tempfile.mkdtemp()
      self._delete_output_dir = True

    if xctestrun_file_path:
      self._xctestrun_obj = xctestrun.XctestRun(
          xctestrun_file_path, test_type)
    else:
      if not test_bundle:
        raise ios_errors.IllegalArgumentError(
            'Without providing xctestrun file, test bundle is required.')
      app_under_test_dir, test_bundle_dir = _PrepareBundles(
          self._work_dir, app_under_test, test_bundle)
      test_type = _FinalizeTestType(
          test_bundle_dir, self._sdk, app_under_test_dir=app_under_test_dir,
          original_test_type=test_type)

      if test_type not in ios_constants.SUPPORTED_TEST_TYPES:
        raise ios_errors.IllegalArgumentError(
            'The test type %s is not supported. Supported test types are %s' %
            (test_type, ios_constants.SUPPORTED_TEST_TYPES))

      if test_type != ios_constants.TestType.LOGIC_TEST:
        xctestrun_factory = xctestrun.XctestRunFactory(
            app_under_test_dir, test_bundle_dir, self._sdk, self._device_arch,
            test_type, signing_options, self._work_dir)
        self._xctestrun_obj = xctestrun_factory.GenerateXctestrun()
      else:
        self._logic_test_bundle = test_bundle_dir
    self._prepared = True

  def SetLaunchOptions(self, launch_options):
    """Set the launch options to xctest session.

    Args:
      launch_options: dict, the signing app options. See
          ios_constants.LAUNCH_OPTIONS_JSON_HELP for details.
    """
    if not self._prepared:
      raise ios_errors.XcodebuildTestError(
          'The session has not been prepared. Please call '
          'XctestSession.Prepare first.')
    if not launch_options:
      return
    self._keep_xcresult_data = launch_options.get('keep_xcresult_data', True)
    self._startup_timeout_sec = launch_options.get('startup_timeout_sec')
    self._destination_timeout_sec = launch_options.get(
        'destination_timeout_sec')
    if self._xctestrun_obj:
      self._xctestrun_obj.SetTestEnvVars(launch_options.get('env_vars'))
      self._xctestrun_obj.SetTestArgs(launch_options.get('args'))
      self._xctestrun_obj.SetTestsToRun(launch_options.get('tests_to_run'))
      self._xctestrun_obj.SetSkipTests(launch_options.get('skip_tests'))
      self._xctestrun_obj.SetAppUnderTestEnvVars(
          launch_options.get('app_under_test_env_vars'))
      self._xctestrun_obj.SetAppUnderTestArgs(
          launch_options.get('app_under_test_args'))

      if launch_options.get('uitest_auto_screenshots'):
        self._disable_uitest_auto_screenshots = False
        # By default, this SystemAttachmentLifetime field is in the generated
        # test.xctestrun.
        try:
          self._xctestrun_obj.DeleteXctestrunField('SystemAttachmentLifetime')
        except ios_errors.PlistError:
          pass
    elif self._logic_test_bundle:
      self._logic_test_env_vars = launch_options.get('env_vars')
      self._logic_test_args = launch_options.get('args')
      self._logic_tests_to_run = launch_options.get('tests_to_run')

  def RunTest(self, device_id, os_version=None):
    """Runs test on the target device with the given device_id.

    Args:
      device_id: string, id of the device.
      os_version: string, OS version of the device.

    Returns:
      A value of type runner_exit_codes.EXITCODE.

    Raises:
      XcodebuildTestError: when the XctestSession.Prepare has not been called.
    """
    if not self._prepared:
      raise ios_errors.XcodebuildTestError(
          'The session has not been prepared. Please call '
          'XctestSession.Prepare first.')

    if self._xctestrun_obj:
      result_bundle_path = os.path.join(self._output_dir, 'test.xcresult')
      exit_code = self._xctestrun_obj.Run(device_id, self._sdk,
                                          self._output_dir,
                                          self._startup_timeout_sec,
                                          self._destination_timeout_sec,
                                          os_version=os_version,
                                          result_bundle_path=result_bundle_path)
      # The xcresult only contains raw data in Xcode 11 or later.
      if xcode_info_util.GetXcodeVersionNumber() >= 1100:
        expose_xcresult = os.path.join(self._output_dir, 'ExposeXcresult')
        try:
          xcresult_util.ExposeXcresult(result_bundle_path, expose_xcresult)
          if not self._keep_xcresult_data:
            shutil.rmtree(result_bundle_path)
        except subprocess.CalledProcessError as e:
          logging.warning(e.output)
      return exit_code
    elif self._logic_test_bundle:
      return logic_test_util.RunLogicTestOnSim(
          device_id,
          self._logic_test_bundle,
          self._logic_test_env_vars,
          self._logic_test_args,
          self._logic_tests_to_run,
          os_version=os_version)
    else:
      raise ios_errors.XcodebuildTestError('Unexpected runtime error.')

  def Close(self):
    """Deletes the temp directories."""
    if (self._delete_work_dir and self._work_dir and
        os.path.exists(self._work_dir)):
      shutil.rmtree(self._work_dir)
    if (self._delete_output_dir and self._output_dir and
        os.path.exists(self._output_dir)):
      shutil.rmtree(self._output_dir)


def _PrepareBundles(working_dir, app_under_test_path, test_bundle_path):
  """Prepares the bundles in work directory.

  If the original bundle is .ipa, the .ipa file will be unzipped under
  working_dir. If the original bundle is .app/.xctest and the bundle file is not
  in working_dir, the bundle file will be copied to working_dir.

  Args:
    working_dir: string, the working directory.
    app_under_test_path: string, the path of the application to be tested.
        It can be .ipa or .app. It can be None.
    test_bundle_path: string, the path of the test bundle to be tested. It can
        be .ipa or .xctest.

  Returns:
    a tuple with two items:
      a path of app under test directory (.app) under work directory.
      a path of test bundle directory (.xctest) under work directory.

  Raises:
    ios_errors.IllegalArgumentError: if the app under test/test bundle does not
      exist or its extension is invaild.
  """
  working_dir = os.path.abspath(working_dir)
  app_under_test_dir = None
  if app_under_test_path:
    if not os.path.exists(app_under_test_path):
      raise ios_errors.IllegalArgumentError(
          'The app under test does not exists: %s' % app_under_test_path)
    if not (app_under_test_path.endswith('.app') or
            app_under_test_path.endswith('.ipa')):
      raise ios_errors.IllegalArgumentError(
          'The app under test %s should be with .app or .ipa extension.'
          % app_under_test_path)

    app_under_test_dir = os.path.join(
        working_dir,
        os.path.splitext(os.path.basename(app_under_test_path))[0] + '.app')
    if not os.path.exists(app_under_test_dir):
      if app_under_test_path.endswith('.ipa'):
        extract_app_under_test_dir = bundle_util.ExtractApp(
            app_under_test_path, working_dir)
        shutil.move(extract_app_under_test_dir, app_under_test_dir)
      elif not os.path.abspath(app_under_test_path).startswith(working_dir):
        # Only copies the app under test if it is not in working directory.
        shutil.copytree(app_under_test_path, app_under_test_dir)
      else:
        app_under_test_dir = app_under_test_path

  if not os.path.exists(test_bundle_path):
    raise ios_errors.IllegalArgumentError(
        'The test bundle does not exists: %s' % test_bundle_path)
  if not (test_bundle_path.endswith('.xctest') or
          test_bundle_path.endswith('.ipa') or
          test_bundle_path.endswith('.zip')):
    raise ios_errors.IllegalArgumentError(
        'The test bundle %s should be with .xctest, .ipa or .zip extension.'
        % test_bundle_path)

  test_bundle_dir = os.path.join(
      working_dir,
      os.path.splitext(os.path.basename(test_bundle_path))[0] + '.xctest')
  if not os.path.exists(test_bundle_dir):
    if test_bundle_path.endswith('.ipa') or test_bundle_path.endswith('.zip'):
      extract_test_bundle_dir = bundle_util.ExtractTestBundle(
          test_bundle_path, working_dir)
      shutil.move(extract_test_bundle_dir, test_bundle_dir)
    elif not os.path.abspath(test_bundle_path).startswith(working_dir):
      # Only copies the test bundle if it is not in working directory.
      shutil.copytree(test_bundle_path, test_bundle_dir)
    else:
      test_bundle_dir = test_bundle_path

  return app_under_test_dir, test_bundle_dir


def _FinalizeTestType(
    test_bundle_dir, sdk, app_under_test_dir=None, original_test_type=None):
  """Finalizes the test type of the test session according to the args.

  If original_test_type is not given, will auto detect the test bundle.
  If

  Args:
    test_bundle_dir: string, the path of test bundle folder.
    sdk: ios_constants.SDK, the sdk of testing device.
    app_under_test_dir: string, the path of app under test folder.
    original_test_type: ios_constants.TestType, the original test type.

  Returns:
    a ios_constants.TestType object.

  Raises:
    ios_errors.IllegalArgumentError: The given arguments are unmatch.
  """
  if not original_test_type:
    test_type = _DetectTestType(test_bundle_dir)
    if (test_type == ios_constants.TestType.XCTEST and
        not app_under_test_dir and sdk == ios_constants.SDK.IPHONESIMULATOR):
      test_type = ios_constants.TestType.LOGIC_TEST
    logging.info('Will consider the test as test type %s to run.', test_type)
  else:
    test_type = original_test_type
    if (test_type == ios_constants.TestType.LOGIC_TEST and
        sdk != ios_constants.SDK.IPHONESIMULATOR):
      if app_under_test_dir:
        test_type = ios_constants.TestType.XCTEST
        logging.info(
            'Will consider the test as test type XCTest to run. Because '
            'it is only support running Logic Test on iOS simulator and the '
            'sdk of testing device is %s.', sdk)
      else:
        raise ios_errors.IllegalArgumentError(
            'It is only support running Logic Test on iOS simulator.'
            'The sdk of testing device is %s.' % sdk)
    elif (test_type == ios_constants.TestType.XCTEST and
          not app_under_test_dir and sdk == ios_constants.SDK.IPHONESIMULATOR):
      test_type = ios_constants.TestType.LOGIC_TEST
      logging.info(
          'Will consider the test as test type Logic Test to run. Because the '
          'app under test is not given.')
  if (not app_under_test_dir and
      test_type != ios_constants.TestType.LOGIC_TEST):
    raise ios_errors.IllegalArgumentError(
        'The app under test is required in test type %s.' % test_type)
  return test_type


def _DetectTestType(test_bundle_dir):
  """Detects if the test bundle is XCUITest or XCTest."""
  test_bundle_exec_path = os.path.join(
      test_bundle_dir,
      os.path.splitext(os.path.basename(test_bundle_dir))[0])
  output = subprocess.check_output(['nm',
                                    test_bundle_exec_path]).decode('utf-8')
  if 'XCUIApplication' in output:
    return ios_constants.TestType.XCUITEST
  else:
    return ios_constants.TestType.XCTEST
