# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test mmap_utils with both local and non-local filesystems."""

import logging
import os
import sys
import tempfile

import os # import gfile
from absl.testing import absltest as googletest
from litert.python.tools import mmap_utils


# A `tuple[event: str, exception: Exception]` that will be used by
# `_sys_audit_hook` to raise the given `exception` the next time an `event` is
# detected.
_SYS_AUDIT_HOOK_RAISE_ON_NEXT = None


def _sys_audit_hook(event: str, *_):
  global _SYS_AUDIT_HOOK_RAISE_ON_NEXT
  if (
      _SYS_AUDIT_HOOK_RAISE_ON_NEXT is not None
      and event == _SYS_AUDIT_HOOK_RAISE_ON_NEXT[0]
  ):
    exception = _SYS_AUDIT_HOOK_RAISE_ON_NEXT[1]
    _SYS_AUDIT_HOOK_RAISE_ON_NEXT = None
    raise exception

sys.addaudithook(_sys_audit_hook)


class MmapUtilsTest(googletest.TestCase):

  _temp_dir: str
  _mappable_path: str
  _test_data: bytes = (
      b'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod'
      b' tempor incididunt ut labore et dolore magna aliqua.'
  )

  def _raise_on_next_event(self, event: str, exception: Exception):
    global _SYS_AUDIT_HOOK_RAISE_ON_NEXT
    _SYS_AUDIT_HOOK_RAISE_ON_NEXT = (event, exception)

  def setUp(self):
    super().setUp()
    self._temp_dir = tempfile.mkdtemp()
    logging.info('Created test directory "%s".', self._temp_dir)

    # Create a regular binary file containing the test data.
    self._mappable_path = os.path.join(self._temp_dir, 'mappable_file')
    with open(self._mappable_path, 'wb') as f:
      f.write(self._test_data)

  def test_get_mapped_buffer_new_mappable_file(self):
    path = os.path.join(self._temp_dir, 'new_mappable_file')
    buffer = mmap_utils.get_mapped_buffer(path, len(self._test_data))
    self.assertLen(buffer, len(self._test_data))
    buffer[:] = self._test_data
    buffer.flush()

    with open(path, 'rb') as f:
      self.assertEqual(self._test_data, f.read())

  def test_get_mapped_buffer_existing_mappable_file(self):
    data_reversed = bytes(reversed(self._test_data))
    buffer = mmap_utils.get_mapped_buffer(
        self._mappable_path, len(self._test_data)
    )
    self.assertEqual(self._test_data, buffer[:])
    buffer[:] = data_reversed
    buffer.flush()

    with open(self._mappable_path, 'rb') as f:
      self.assertEqual(data_reversed, f.read())

  def test_get_mapped_buffer_existing_non_mappable_file(self):
    # Make the next call to `mmap.mmap` fails.
    self._raise_on_next_event('mmap.__new__', OSError())

    with self.assertRaises(OSError):
      mmap_utils.get_mapped_buffer(
          self._mappable_path, len(self._test_data)
      )

  def test_get_mapped_buffer_or_none_new_mappable_file(self):
    path = os.path.join(self._temp_dir, 'new_mappable_file')
    buffer = mmap_utils.get_mapped_buffer_or_none(path, len(self._test_data))
    self.assertIsNotNone((buffer))
    self.assertLen(buffer, len(self._test_data))
    buffer[:] = self._test_data
    buffer.flush()

    with open(path, 'rb') as f:
      self.assertEqual(self._test_data, f.read())

  def test_get_mapped_buffer_or_none_existing_mappable_file(self):
    data_reversed = bytes(reversed(self._test_data))
    buffer = mmap_utils.get_mapped_buffer_or_none(
        self._mappable_path, len(self._test_data)
    )
    self.assertIsNotNone((buffer))
    self.assertEqual(self._test_data, buffer[:])
    buffer[:] = data_reversed
    buffer.flush()

    with open(self._mappable_path, 'rb') as f:
      self.assertEqual(data_reversed, f.read())

  def test_get_mapped_buffer_or_none_existing_non_mappable_file(self):
    # Make the next call to `mmap.mmap` fails.
    self._raise_on_next_event('mmap.__new__', OSError())

    self.assertIsNone(
        mmap_utils.get_mapped_buffer_or_none(
            self._mappable_path, len(self._test_data)
        )
    )

  def test_get_file_contents_mappable_file(self):
    self.assertEqual(
        self._test_data, mmap_utils.get_file_contents(self._mappable_path)
    )

  def test_get_file_contents_non_mappable_file(self):
    # Make the next call to `mmap.mmap` fails.
    self._raise_on_next_event('mmap.__new__', OSError())

    self.assertEqual(
        self._test_data, mmap_utils.get_file_contents(self._mappable_path)
    )

  def test_set_file_contents_mappable_file(self):
    mmap_utils.set_file_contents(self._mappable_path, self._test_data)
    with open(self._mappable_path, 'rb') as f:
      self.assertEqual(self._test_data, f.read())

  def test_set_file_contents_non_mappable_file(self):
    # Make the next call to `mmap.mmap` fails.
    self._raise_on_next_event('mmap.__new__', OSError())

    mmap_utils.set_file_contents(self._mappable_path, self._test_data)
    with open(self._mappable_path, 'rb') as f:
      self.assertEqual(self._test_data, f.read())


if __name__ == '__main__':
  googletest.main()
