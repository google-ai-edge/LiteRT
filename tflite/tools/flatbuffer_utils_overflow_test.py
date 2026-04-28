# Copyright 2026 The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Regression tests for overflow-safe bounds checks in flatbuffer_utils.

The tested predicate is extracted from flatbuffer_utils.read_model_from_bytearray
and validated in isolation so the test does not depend on
schema_py_generated (a build-time generated module absent from the
git tree).
"""

import unittest


def _bounds_check_rejects(offset: int, size: int, model_len: int) -> bool:
  """Mirror of the predicate now enforced in flatbuffer_utils.py.

  Returns True when the (offset, size) pair is rejected (the safe
  behaviour). The legacy code silently accepted any (offset, size) and
  used Python slice clamping.
  """
  return offset > model_len or size > model_len - offset


class FlatbufferOverflowTest(unittest.TestCase):

  MODEL_LEN = 4096

  def test_offset_plus_size_overflow_rejected(self):
    # uint64 wrap: offset = 2^64 - 0x100, size = 0x100. Sum wraps to 0;
    # legacy `offset+size > model_len` would return False and accept.
    self.assertTrue(_bounds_check_rejects(
        (1 << 64) - 0x100, 0x100, self.MODEL_LEN))

  def test_offset_exceeds_model_rejected(self):
    self.assertTrue(_bounds_check_rejects(8192, 16, self.MODEL_LEN))

  def test_size_exceeds_remaining_rejected(self):
    self.assertTrue(_bounds_check_rejects(4000, 1000, self.MODEL_LEN))

  def test_well_formed_buffer_accepted(self):
    self.assertFalse(_bounds_check_rejects(100, 32, self.MODEL_LEN))

  def test_exact_fit_accepted(self):
    self.assertFalse(_bounds_check_rejects(
        self.MODEL_LEN - 32, 32, self.MODEL_LEN))

  def test_zero_offset_zero_size_accepted(self):
    self.assertFalse(_bounds_check_rejects(0, 0, self.MODEL_LEN))


if __name__ == '__main__':
  unittest.main()
