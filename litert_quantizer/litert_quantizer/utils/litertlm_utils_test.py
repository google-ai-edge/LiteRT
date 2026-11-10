# Copyright 2024 Google LLC
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

"""Unit tests for reading/parsing LiteRT-LM files."""

import os
import random

from absl.testing import absltest

from litert_lm_builder import litertlm_core
from litert_quantizer.utils import litertlm_utils
from litert_quantizer.utils import test_utils


class LitertlmUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._litertlm_path = os.path.join(
        test_utils.get_path_to_datafile("../tests/models"),
        "test_tok_tfl_llm.litertlm",
    )

  def test_gets_model_type_from_litertlm(self):
    litertlm_file = litertlm_utils.LiteRTLMFile(self._litertlm_path)
    sections = litertlm_file.sections
    self.assertLen(sections, 4)
    self.assertIsNone(litertlm_file.get_model_type(0))
    self.assertEqual(litertlm_file.get_model_type(1), "tf_lite_prefill_decode")
    self.assertIsNone(litertlm_file.get_model_type(2))
    self.assertIsNone(litertlm_file.get_model_type(3))

  def test_reads_model_from_litertlm(self):
    litertlm_file = litertlm_utils.LiteRTLMFile(self._litertlm_path)
    self.assertIsNotNone(litertlm_file.read_model(1))

  def test_serializes_with_data_overrides(self):
    # Load the original file.
    litertlm_file = litertlm_utils.LiteRTLMFile(self._litertlm_path)

    # Re-serialize the file overriding two of its sections. The section sizes
    # are chosen to _not_ be integer multiples of the LiteRT-LM block size to
    # verify that they are padded correctly.
    output_path = self.create_tempfile("modified.litertlm").full_path
    overrides = {
        0: random.randbytes(10 * litertlm_core.BLOCK_SIZE + 1),
        2: random.randbytes(10 * litertlm_core.BLOCK_SIZE - 1),
    }
    litertlm_file.serialize(output_path, overrides)

    # Read the modified LiteRT-LM file.
    modified_litertlm_file = litertlm_utils.LiteRTLMFile(output_path)

    # Compare the sections.
    self.assertEqual(
        len(litertlm_file.sections), len(modified_litertlm_file.sections)
    )
    for section_id in range(len(litertlm_file.sections)):
      # Check that the key value pairs are the same.
      self.assertEqual(
          litertlm_file.get_section_metadata(section_id),
          modified_litertlm_file.get_section_metadata(section_id),
      )

      # Check whether the buffers match the original or override data.
      if buff := overrides.get(section_id):
        self.assertEqual(
            buff, modified_litertlm_file.get_section_buffer(section_id)
        )
      else:
        self.assertEqual(
            litertlm_file.get_section_buffer(section_id),
            modified_litertlm_file.get_section_buffer(section_id),
        )


if __name__ == "__main__":
  absltest.main()
