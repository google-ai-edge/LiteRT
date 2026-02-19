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

"""Helper class for parsing xcresult under Xcode 11 or later."""

import json
import os
import subprocess

from xctestrunner.shared import ios_errors
from xctestrunner.shared import xcode_info_util


def ExposeXcresult(xcresult_path, output_path):
  """Exposes the files from xcresult.

  The files includes the diagnostics files and attachments files.

  Args:
    xcresult_path: string, path of xcresult bundle.
    output_path: string, path of output directory.
  """
  root_result_bundle = _GetResultBundleObject(xcresult_path, bundle_id=None)
  actions = root_result_bundle['actions']['_values']
  action_result = None
  for action in actions:
    if action['_type']['_name'] == 'ActionRecord':
      action_result = action['actionResult']
      break
  if action_result is None:
    raise ios_errors.XcresultError(
        'Failed to get "ActionResult" from result bundle %s' %
        root_result_bundle)
  _ExposeDiagnostics(xcresult_path, output_path, action_result)
  _ExposeAttachments(xcresult_path, output_path, action_result)


def _ExposeDiagnostics(xcresult_path, output_path, action_result):
  """Exposes the diagnostics files from the given xcresult file."""
  if 'diagnosticsRef' not in action_result:
    return
  diagnostics_id = action_result['diagnosticsRef']['id']['_value']
  export_command = _MakeXcresulttoolCommand([
    'export', '--path', xcresult_path,
    '--output-path', output_path, '--type', 'directory', '--id',
    diagnostics_id
  ])
  subprocess.check_call(export_command)


def _ExposeAttachments(xcresult_path, output_path, action_result):
  """Exposes the attachments files from the given xcresult file."""
  testsref_id = action_result['testsRef']['id']['_value']
  test_plan_summaries = _GetResultBundleObject(
      xcresult_path, bundle_id=testsref_id)
  test_plan_summary = test_plan_summaries['summaries']['_values'][0]
  testable_summary = test_plan_summary['testableSummaries']['_values'][0]
  # If the app under test crashes in unit test (XCTest) before loading the
  # tests, the testable summary won't have tests summary.
  if 'tests' not in testable_summary:
    return
  root_tests_summary = testable_summary['tests']['_values'][0]
  failure_test_ref_ids = _GetFailureTestRefs(root_tests_summary)
  for test_ref_id in failure_test_ref_ids:
    test_summary_result = _GetResultBundleObject(xcresult_path, test_ref_id)
    # if the test results in an `expectedFailures` entry, there might be an
    # `activitySummaries` field present.
    if 'activitySummaries' not in test_summary_result:
      continue

    activity_summaries = test_summary_result['activitySummaries']['_values']
    for activity_summary in activity_summaries:
      if 'attachments' in activity_summary:
        test_identifier = test_summary_result['identifier']['_value']
        for attachment in activity_summary['attachments']['_values']:
          file_name = attachment['filename']['_value']
          target_file_dir = os.path.join(output_path, 'Attachments',
                                         test_identifier)
          if not os.path.exists(target_file_dir):
            os.makedirs(target_file_dir)
          target_file_path = os.path.join(target_file_dir, file_name)

          payload_ref_id = attachment['payloadRef']['id']['_value']
          export_command = _MakeXcresulttoolCommand([
            'export', '--path', xcresult_path,
            '--output-path', target_file_path, '--type', 'file', '--id',
            payload_ref_id
          ])
          subprocess.check_call(export_command)


def _GetResultBundleObject(xcresult_path, bundle_id=None):
  """Gets the result bundle object in json format.

  Args:
    xcresult_path: string, path of xcresult bundle.
    bundle_id: string, id of the result bundle object. If it is None, it is
        rootID.
  Returns:
    A dict, result bundle object in json format.
  """
  command = _MakeXcresulttoolCommand([
    'get', '--format', 'json', '--path', xcresult_path
  ])
  if bundle_id:
    command.extend(['--id', bundle_id])
  return json.loads(subprocess.check_output(command).decode('utf-8'))


def _GetFailureTestRefs(test_summary):
  """Gets a list of test summaryRef id of all failure test.

  Args:
    test_summary: dict, a dict of test summary object.
  Returns:
    A list of failure test case's summaryRef id.
  """
  failure_test_refs = []
  if 'subtests' in test_summary:
    for sub_test_summary in test_summary['subtests']['_values']:
      failure_test_refs.extend(_GetFailureTestRefs(sub_test_summary))
  else:
    if (('testStatus' not in test_summary or
         test_summary['testStatus']['_value'] != 'Success') and
        'summaryRef' in test_summary):
      summary_ref_id = test_summary['summaryRef']['id']['_value']
      failure_test_refs.append(summary_ref_id)
  return failure_test_refs

def _MakeXcresulttoolCommand(args):
  """Constructs xcresulttool command for selected Xcode version.

  Args:
    args: array, a list of arguments to pass to xcresulttool.
  Returns:
    The xcresulttool command.
  """
  command = ['xcrun', 'xcresulttool'] + args
  xcode_version = xcode_info_util.GetXcodeVersionNumber()
  if xcode_version >= 1600:
    command.extend(['--legacy'])
  return command
