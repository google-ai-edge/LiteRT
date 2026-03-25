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

import ctypes
import logging
import mmap
import os
import pathlib
from typing import Any

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


def get_file_contents(
    path: Path, *, offset: int = 0, size: int = 0
) -> memoryview:
  """Creates a `mmap.mmap` or `bytearray` of the contents of the given file.

  If the file is memory mapped, then it will be mapped as `mmap.MAP_PRIVATE`
  which does copy-on-write, i.e. without altering the underlying file.

  Args:
    path: The path of the new file.
    offset: Optional offset in bytes within the file.
    size: Optional number of bytes to read (until EOF if `0`).

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
          fd,
          size,
          flags=mmap.MAP_PRIVATE,
          prot=mmap.PROT_READ | mmap.PROT_WRITE,
          offset=offset,
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
    size = size or (os.stat(path).st_size - offset)
    data = bytearray(size)
    with open(path, 'rb') as f:
      if offset:
        f.seek(offset)
      assert f.readinto(data) == size

  return memoryview(data)


def set_file_contents(path: Path, data: BufferType, *, offset: int = 0):
  """Write the `data` to the given `path`.

  Args:
    path: The path of the new file.
    data: The binary data to write.
    offset: Optional offset in bytes within the file.

  Raises:
    An `OSError` if creating/opening the file fails, or if `mmap.mmap` fails.
  """
  # Try to mmap the file first if it is local.
  if (
      output_map := get_mapped_buffer_or_none(path, offset + len(data))
  ) is not None:
    output_map[offset:] = data
    output_map.close()

  else:
    # If mapping failed (path might refer to a special file that either
    # `os.open` or `mmap.mmap` can't handle, go at it conventionally.
    with open(path, 'wb') as f:
      if offset:
        f.seek(offset)
      # Write the file in chunks to avoid creating large internal buffers.
      finger = 0
      chunk_size = max(len(data) // 10, 10 * 1024 * 1024)
      while finger < len(data):
        f.write(data[finger : finger + chunk_size])
        finger += chunk_size


def _get_mmap_offset_and_length(
    buffer: Any,
) -> tuple[mmap.mmap | None, int, int]:
  """Returns the underlying mmap, offset, and length for a buffer."""
  if isinstance(buffer, mmap.mmap):
    return buffer, 0, buffer.size()

  # Find the root mmap object if it exists.
  try:
    mv = memoryview(buffer)
  except TypeError:
    return None, 0, 0

  root = mv
  while isinstance(root, memoryview):
    root = root.obj

  if not isinstance(root, mmap.mmap):
    return None, 0, 0

  try:
    # We use ctypes to get the memory address of the buffers.
    root_addr = ctypes.addressof(ctypes.c_char.from_buffer(root))
    slice_addr = ctypes.addressof(ctypes.c_char.from_buffer(mv))
    return root, slice_addr - root_addr, mv.nbytes
  except (TypeError, ValueError):
    return None, 0, 0


def advise_sequential(buffer: Any, offset: int = 0, length: int = 0):
  """Advises the kernel that the `buffer` will be read sequentially.

  Args:
    buffer: The buffer to set `MADV_SEQUENTIAL` on.
    offset: The offset into the buffer to start from.
    length: The length of the chunk to advise on. If 0, the entire buffer from
      `offset` is used.
  """
  m, m_offset, m_length = _get_mmap_offset_and_length(buffer)
  if  m is not None:
    if length == 0:
      length = m_length - offset

    # Linux requires page alignment for madvise.
    # For MADV_SEQUENTIAL, we can be more liberal and include partial pages.
    start = m_offset + offset
    end = start + length
    aligned_start = start & ~(mmap.PAGESIZE - 1)
    aligned_end = (end + mmap.PAGESIZE - 1) & ~(mmap.PAGESIZE - 1)

    try:
      m.madvise(
          mmap.MADV_SEQUENTIAL, aligned_start, aligned_end - aligned_start
      )
    except (OSError, ValueError) as e:
      logging.warning('Failed to set MADV_SEQUENTIAL: %s', e)


def advise_dont_need(buffer: Any, offset: int = 0, length: int = 0):
  """Advises the kernel that the `buffer` is no longer needed.

  Args:
    buffer: The buffer to set `MADV_DONTNEED` on.
    offset: The offset into the buffer to start from.
    length: The length of the chunk to advise on. If 0, the entire buffer from
      `offset` is used.
  """
  m, m_offset, m_length = _get_mmap_offset_and_length(buffer)
  if m is not None:
    if length == 0:
      length = m_length - offset

    # Linux requires page alignment for madvise.
    # For MADV_DONTNEED, we MUST stay within the requested range to avoid
    # reclaiming memory that might still be needed by adjacent tensors sharing
    # pages.
    start = m_offset + offset
    end = start + length
    aligned_start = (start + mmap.PAGESIZE - 1) & ~(mmap.PAGESIZE - 1)
    aligned_end = end & ~(mmap.PAGESIZE - 1)

    if aligned_end > aligned_start:
      try:
        m.madvise(
            mmap.MADV_DONTNEED, aligned_start, aligned_end - aligned_start
        )
      except (OSError, ValueError) as e:
        logging.warning('Failed to set MADV_DONTNEED: %s', e)
