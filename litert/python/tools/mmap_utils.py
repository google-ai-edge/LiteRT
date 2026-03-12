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
"""A collection of utility functions for reading/writing memory-mapped data."""

import logging
import mmap
import os
import pathlib

import os # import gfile


Path = str | pathlib.Path
BufferType = bytes | bytearray | memoryview | mmap.mmap


def get_mapped_buffer(path: Path, size: int) -> mmap.mmap:
  """Creates an `mmap.mmap` of a file of the given `size` at `path`.

  Any data written to the resulting buffer will also be stored at the given
  path.

  Args:
    path: The path of the underlying file. If the file already exists, it will
      be overwritten.
    size: The size, in bytes, of the file to create.

  Returns:
    A `mmap.mmap` of the mutable contents of the file.

  Raises:
    An `OSError` if creating/opening the file fails, or if `mmap.mmap` fails.
  """
  fd = os.open(path, flags=os.O_RDWR | os.O_CREAT)
  try:
    os.truncate(fd, size)
    output_map = mmap.mmap(
        fd, 0, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE
    )
  except OSError as e:
    os.close(fd)
    raise e
  else:
    os.close(fd)

  return output_map


def get_mapped_buffer_or_none(
    path: Path, size: int
) -> mmap.mmap | None:
  """Creates an `mmap.mmap` of a file of the given `size` at `path`.

  Same as `get_file_as_buffer`, but catches all exceptions an return `None` if
  anything goes wrong.

  Args:
    path: The path of the underlying file. If the file already exists, it will
      be overwritten.
    size: The size, in bytes, of the file to create.

  Returns:
    A `mmap.mmap` of the mutable contents of the file, or `None` if it could not
    be created.
  """
  try:
    return get_mapped_buffer(path, size)
  except OSError as e:
    logging.info(
        'Failed to create/open an `mmap` of size %i for the file "%s": %s',
        size,
        path,
        e,
    )
  return None


def get_file_contents(path: Path) -> memoryview:
  """Creates a `mmap.mmap` or `bytearray` of the contents of the given file.

  If the file is memory mapped, then it will be mapped as `mmap.MAP_PRIVATE`
  which does copy-on-write, i.e. without altering the underlying file.

  Args:
    path: The path of the new file.

  Returns:
    A `memoryview` wrapping either an `mmap.mmap` or `bytearray` of the contents
    of the file.
  """
  data = None

  # Try to mmap the file first if it is local.
  try:
    fd = os.open(path, os.O_RDONLY)
  except OSError as e:
    logging.info('Failed to open the file "%s": %s', path, e)
  else:
    try:
      data = mmap.mmap(
          fd, 0, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE
      )
    except OSError as e:
      logging.info(
          'Failed to create an `mmap` of the file "%s": %s',
          path,
          e,
      )
    os.close(fd)

  # If mapping failed (path might refer to a special file that either `os.open`
  # or `mmap.mmap` can't handle, go at it conventionally.
  if data is None:
    size = os.stat(path).st_size
    data = bytearray(size)
    with open(path, 'rb') as f:
      assert f.readinto(data) == size

  return memoryview(data)


def set_file_contents(
    path: Path, data: BufferType
):
  """Write the `data` to the given `path`.

  Args:
    path: The path of the new file.
    data: The binary data to write.

  Raises:
    An `OSError` if creating/opening the file fails, or if `mmap.mmap` fails.
  """
  # Try to mmap the file first if it is local.
  if (output_map := get_mapped_buffer_or_none(path, len(data))) is not None:
    output_map[:] = data
    output_map.close()

  else:
    # If mapping failed (path might refer to a special file that either
    # `os.open` or `mmap.mmap` can't handle, go at it conventionally.
    with open(path, 'wb') as f:
      # Write the file in chunks to avoid creating large internal buffers.
      finger = 0
      chunk_size = max(len(data) // 10, 10 * 1024 * 1024)
      while finger < len(data):
        f.write(data[finger : finger + chunk_size])
        finger += chunk_size
