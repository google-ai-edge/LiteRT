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
import mmap
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
    # Get whole file.
    self.assertEqual(
        self._test_data, mmap_utils.get_file_contents(self._mappable_path)
    )

    # Get with an offset.
    self.assertEqual(
        self._test_data[10:],
        mmap_utils.get_file_contents(self._mappable_path, offset=10),
    )

    # Get with offset and size.
    self.assertEqual(
        self._test_data[10:20],
        mmap_utils.get_file_contents(self._mappable_path, offset=10, size=10),
    )

  def test_get_file_contents_non_mappable_file(self):
    # Make the next call to `mmap.mmap` fails.
    self._raise_on_next_event('mmap.__new__', OSError())

    self.assertEqual(
        self._test_data, mmap_utils.get_file_contents(self._mappable_path)
    )

    # Get with an offset.
    self._raise_on_next_event('mmap.__new__', OSError())
    self.assertEqual(
        self._test_data[10:],
        mmap_utils.get_file_contents(self._mappable_path, offset=10),
    )

    # Get with offset and size.
    self._raise_on_next_event('mmap.__new__', OSError())
    self.assertEqual(
        self._test_data[10:20],
        mmap_utils.get_file_contents(self._mappable_path, offset=10, size=10),
    )

  def test_get_file_contents_os_open_fails_once(self):
    # Make the next call to `os.open` fail.
    self._raise_on_next_event('open', OSError())

    self.assertEqual(
        self._test_data, mmap_utils.get_file_contents(self._mappable_path)
    )

    # Get with an offset.
    self._raise_on_next_event('open', OSError())
    self.assertEqual(
        self._test_data[10:],
        mmap_utils.get_file_contents(self._mappable_path, offset=10),
    )

    # Get with offset and size.
    self._raise_on_next_event('open', OSError())
    self.assertEqual(
        self._test_data[10:20],
        mmap_utils.get_file_contents(self._mappable_path, offset=10, size=10),
    )

  def test_set_file_contents_mappable_file(self):
    mmap_utils.set_file_contents(self._mappable_path, self._test_data)
    with open(self._mappable_path, 'rb') as f:
      self.assertEqual(self._test_data, f.read())

    # Set with an offset.
    mmap_utils.set_file_contents(
        self._mappable_path, self._test_data, offset=10
    )
    with open(self._mappable_path, 'rb') as f:
      f.seek(10)
      self.assertEqual(self._test_data, f.read())

  def test_set_file_contents_non_mappable_file(self):
    # Make the next call to `mmap.mmap` fails.
    self._raise_on_next_event('mmap.__new__', OSError())

    mmap_utils.set_file_contents(self._mappable_path, self._test_data)
    with open(self._mappable_path, 'rb') as f:
      self.assertEqual(self._test_data, f.read())

    # Set with an offset.
    self._raise_on_next_event('mmap.__new__', OSError())
    mmap_utils.set_file_contents(
        self._mappable_path, self._test_data, offset=10
    )
    with open(self._mappable_path, 'rb') as f:
      f.seek(10)
      self.assertEqual(self._test_data, f.read())

  def test_advise_sequential_mmap(self):
    buffer = mmap_utils.get_mapped_buffer(
        self._mappable_path, len(self._test_data)
    )
    # This should not raise any exception.
    mmap_utils.advise_sequential(buffer)

  def test_advise_sequential_memoryview_mmap(self):
    buffer = mmap_utils.get_file_contents(self._mappable_path)
    self.assertIsInstance(buffer, memoryview)
    # This should not raise any exception.
    mmap_utils.advise_sequential(buffer)

  def test_advise_sequential_non_mmap(self):
    buffer = bytearray(self._test_data)
    # This should not raise any exception and should just return.
    mmap_utils.advise_sequential(buffer)

  def test_advise_sequential_bytearray(self):
    # bytearray supports the buffer protocol but is not a memoryview.
    buffer = bytearray(self._test_data)
    # This should not raise any exception.
    mmap_utils.advise_sequential(buffer)

  def test_advise_sequential_nested_memoryview(self):
    buffer = mmap_utils.get_file_contents(self._mappable_path)
    nested_buffer = memoryview(buffer)
    # This should not raise any exception.
    mmap_utils.advise_sequential(nested_buffer)

  def test_advise_dont_need_mmap(self):
    buffer = mmap_utils.get_mapped_buffer(
        self._mappable_path, len(self._test_data)
    )
    # This should not raise any exception.
    mmap_utils.advise_dont_need(buffer, 0, len(self._test_data))

  def test_advise_dont_need_memoryview_mmap(self):
    buffer = mmap_utils.get_file_contents(self._mappable_path)
    self.assertIsInstance(buffer, memoryview)
    # This should not raise any exception.
    mmap_utils.advise_dont_need(buffer, 0, len(self._test_data))

  def test_advise_dont_need_memoryview_slice(self):
    content = b'0123456789' * 10
    path = self.create_tempfile(content=content).full_path
    mv = mmap_utils.get_file_contents(path)
    # Create a slice
    slice_offset = 25
    slice_length = 10
    mv_slice = mv[slice_offset : slice_offset + slice_length]

    try:
      # This should not raise any exceptions and should correctly call
      # m.madvise with inferred offset=25 and length=10.
      mmap_utils.advise_dont_need(mv_slice)

      # Test with explicit offset relative to slice
      # This should result in m.madvise(MADV_DONTNEED, 25 + 2, 5)
      mmap_utils.advise_dont_need(mv_slice, offset=2, length=5)
    finally:
      mv_slice.release()
      mv.release()

  def test_advise_sequential_memoryview_slice(self):
    content = b'0123456789' * 10
    path = self.create_tempfile(content=content).full_path
    mv = mmap_utils.get_file_contents(path)
    slice_offset = 25
    mv_slice = mv[slice_offset:]
    try:
      # This should not raise any exceptions.
      mmap_utils.advise_sequential(mv_slice)
    finally:
      mv_slice.release()
      mv.release()

  def test_advise_dont_need_non_mmap(self):
    buffer = bytearray(self._test_data)
    # This should not raise any exception and should just return.
    mmap_utils.advise_dont_need(buffer)


if __name__ == '__main__':
  googletest.main()
