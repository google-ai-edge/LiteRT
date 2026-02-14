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
"""Tests for xctoolrunner."""

import tempfile
import unittest

from unittest import mock

from tools.wrapper_common import execute
from tools.xctoolrunner import xctoolrunner


_UNHANDLED_DESTINATION_METRICS_MSG = (
    "2017-11-15 08:41:21.507 ibtoold[11658:7455845]"
    "WARNING: Unhandled destination metrics: (null)\n")
_CHANGE_PROPERTY_MSG = (
    "2018-08-07 13:23:26.386 ibtoold[51132:942471]"
    " <CATransformLayer: 0x7f8b000e2af0> - "
    "changing property masksToBounds in transform-only layer, "
    "will have no effect\n")
_MODEL_CHECKSUM_MSG = (
    "PersistentQueue.xcdatamodel: note: Model PersistentQueue "
    "version checksum: +liGtY4aOXak0pynyLB0+SQanU4nvQ3MLg5dwh2PE0o=\n")
_MODEL_FAILURE_MSG = (
    "CoreData: warning: no NSValueTransformer with class name "
    "'SomeTransformer' was found for attribute 'value' on entity "
    "'SomeEntity'\n")


class TestIBTOOL(unittest.TestCase):

  def testFiltering(self):
    stdout = _UNHANDLED_DESTINATION_METRICS_MSG + _CHANGE_PROPERTY_MSG
    stderr = ""
    tool_exit_status = 0

    (tool_exit_status, out, _) = xctoolrunner.ibtool_filtering(
        tool_exit_status,
        stdout,
        stderr)

    self.assertEqual(out, _CHANGE_PROPERTY_MSG)


class TestMomcTool(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.momc_input = tempfile.mkdtemp()
    self.momc_output = tempfile.mkdtemp()

    execute_patch = mock.patch.object(
        execute, "execute_and_filter_output", autospec=True)
    self.execute_patch = execute_patch.start()
    self.addCleanup(execute_patch.stop)

  def testFiltering(self):
    stderr = _MODEL_CHECKSUM_MSG + _MODEL_FAILURE_MSG
    tool_exit_status = 0

    (tool_exit_status, _, err) = xctoolrunner.momc_filtering(
        tool_exit_status, "", stderr)

    self.assertEqual(err, _MODEL_FAILURE_MSG)

  def testRaisesFileNotFoundError(self):
    args = [
        "momc",
        "--action",
        "generate",
        self.momc_input,
        self.momc_output,
        "--xctoolrunner_assert_nonempty_dir",
        self.momc_output,
    ]
    self.execute_patch.return_value = (0, None, None)

    with self.assertRaisesRegex(
        FileNotFoundError, "xcrun momc did not generate artifacts.*"):
      xctoolrunner.main(args)

  def testRaisesSystemExit(self):
    args = ["momc", "--action", "generate", self.momc_input, self.momc_output]
    self.execute_patch.return_value = (0, None, None)

    # empty directory
    with self.assertRaises(SystemExit):
      xctoolrunner.main(args)

    # non-empty directory
    args = [
        "momc",
        "--action",
        "generate",
        self.momc_input,
        self.momc_output,
        "--xctoolrunner_assert_nonempty_dir",
        self.momc_output,
    ]
    tempfile.mkstemp(dir=self.momc_output)
    with self.assertRaises(SystemExit):
      xctoolrunner.main(args)

if __name__ == "__main__":
  unittest.main()
