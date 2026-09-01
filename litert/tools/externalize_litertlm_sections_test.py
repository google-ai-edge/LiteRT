# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib
import tempfile
import tomllib
import unittest

from litert.tools import externalize_litertlm_sections

_INPUT_TOML = """\
[system_metadata]
entries = []

[[section]]
section_type = "LlmMetadata"
data_path = "metadata.pbtext"

[[section]]
model_type = "vision_encoder"
section_type = "TFLiteModel"
data_path = "vision.tflite"

[[section]]
model_type = "prefill_decode"
section_type = "TFLiteModel"
data_path = "prefill.tflite"

[[section]]
model_type = "embedder"
section_type = "TFLiteModel"
data_path = "embedder.tflite"

[[section]]
model_type = "embedder"
section_type = "TFLiteWeights"
data_path = "old_embedder.weights"
"""


class ExternalizeLitertlmSectionsTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._temp_dir = tempfile.TemporaryDirectory()
    self.root = pathlib.Path(self._temp_dir.name)
    self.input_toml = self.root / "model.toml"
    self.input_toml.write_text(_INPUT_TOML, encoding="utf-8")
    for name in (
        "metadata.pbtext",
        "vision.tflite",
        "prefill.tflite",
        "embedder.tflite",
        "old_embedder.weights",
    ):
      (self.root / name).write_bytes(name.encode())

  def tearDown(self):
    self._temp_dir.cleanup()
    super().tearDown()

  def test_manifest_includes_every_tflite_model(self):
    manifest = self.root / "manifest.tsv"
    count = externalize_litertlm_sections.build_manifest(
        self.input_toml, manifest
    )

    self.assertEqual(count, 3)
    text = manifest.read_text(encoding="utf-8")
    self.assertIn("vision_encoder", text)
    self.assertIn("prefill_decode", text)
    self.assertIn("embedder", text)

  def test_rewrite_preserves_section_order_and_appends_all_weights(self):
    processed_prefill = self.root / "processed_prefill.tflite"
    processed_prefill.write_bytes(b"processed")
    prefill_weights = self.root / "prefill.weights"
    prefill_weights.write_bytes(b"weights")
    results = self.root / "results.tsv"
    results.write_text(
        "section_index\tmodel_type\tmodel_path\tweights_path\n"
        f"1\tvision_encoder\t{self.root / 'vision.tflite'}\t\n"
        f"2\tprefill_decode\t{processed_prefill}\t{prefill_weights}\n"
        f"3\tembedder\t{self.root / 'embedder.tflite'}\t\n",
        encoding="utf-8",
    )
    output = self.root / "rewritten.toml"

    model_count, weight_count = externalize_litertlm_sections.rewrite_toml(
        self.input_toml, results, output
    )

    self.assertEqual((model_count, weight_count), (3, 2))
    sections = tomllib.loads(output.read_text(encoding="utf-8"))["section"]
    model_types = [section.get("model_type") for section in sections]
    self.assertEqual(
        model_types[:4], [None, "vision_encoder", "prefill_decode", "embedder"]
    )
    first_weight = next(
        index
        for index, section in enumerate(sections)
        if section["section_type"] == "TFLiteWeights"
    )
    self.assertEqual(
        {section["section_type"] for section in sections[first_weight:]},
        {"TFLiteWeights"},
    )
    self.assertEqual(
        [section["model_type"] for section in sections[first_weight:]],
        ["prefill_decode", "embedder"],
    )

  def test_verify_peek_rejects_nonweight_after_weight(self):
    peek = self.root / "peek.txt"
    peek.write_text(
        "Data Type:    TFLiteModel\n"
        "Data Type:    TFLiteWeights\n"
        "Data Type:    SP_Tokenizer\n",
        encoding="utf-8",
    )
    with self.assertRaisesRegex(ValueError, "contiguous EOF suffix"):
      externalize_litertlm_sections.verify_peek_layout(peek, 1, 1)


if __name__ == "__main__":
  unittest.main()
