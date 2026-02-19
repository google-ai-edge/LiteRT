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
"""Tests for wrapper_common.execute."""

import contextlib
import io
import signal
import unittest

from tools.wrapper_common import execute

_INVALID_UTF8 = b'\xa0\xa1'

def _cmd_filter(cmd_result, stdout, stderr):
  # Concat the input to a native string literal, to make sure
  # it doesn't trigger a unicode encode/decode error
  return cmd_result, stdout + ' filtered', stderr + ' filtered'


class ExecuteTest(unittest.TestCase):

  def test_execute_unicode(self):
    bytes_out = u'\u201d '.encode('utf8') + _INVALID_UTF8
    args = ['echo', '-n', bytes_out]

    with contextlib.redirect_stdout(io.StringIO()) as mock_stdout, \
      contextlib.redirect_stderr(io.StringIO()) as mock_stderr:
      execute.execute_and_filter_output(
          args,
          filtering=_cmd_filter,
          print_output=True,
          raise_on_failure=False)
    stdout = mock_stdout.getvalue()
    stderr = mock_stderr.getvalue()

    expected = bytes_out.decode('utf8', 'replace')

    expected += ' filtered'
    self.assertEqual(expected, stdout)
    self.assertIn('filtered', stderr)

  def test_execute_timeout(self):
    args = ['sleep', '30']
    result, stdout, stderr = execute.execute_and_filter_output(
        args, timeout=1, raise_on_failure=False)
    self.assertEqual(-signal.SIGKILL, result)

  def test_execute_inputstr(self):
    args = ['cat', '-']
    result, stdout, stderr = execute.execute_and_filter_output(
        args, inputstr=b'foo', raise_on_failure=False)
    self.assertEqual(0, result)
    self.assertEqual('foo', stdout)

  @contextlib.contextmanager
  def _mock_streams(self):
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    mock_stdout = io.StringIO()
    mock_stderr = io.StringIO()

    try:
      sys.stdout = mock_stdout
      sys.stderr = mock_stderr
      yield mock_stdout, mock_stderr
    finally:
      sys.stdout = orig_stdout
      sys.stderr = orig_stderr

if __name__ == '__main__':
  unittest.main()
