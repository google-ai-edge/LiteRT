#!/usr/bin/env python3
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
"""CPU-only checks of teacher_forced.py on synthetic captures (numpy only)."""

import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np
import teacher_forced


def write_safetensors(path, name, array):
  dtype = {np.dtype(np.float32): "F32", np.dtype(np.int32): "I32"}[array.dtype]
  header = json.dumps({
      name: {
          "dtype": dtype,
          "shape": list(array.shape),
          "data_offsets": [0, array.nbytes],
      }
  }).encode()
  with open(path, "wb") as f:
    f.write(struct.pack("<Q", len(header)))
    f.write(header)
    f.write(array.tobytes())


class TeacherForcedTest(unittest.TestCase):

  def test_compare_steps(self):
    ref = np.array(
        [[3.0, 1.0, 0.0, -1.0], [0.0, 2.0, 1.9, 0.0]], dtype=np.float32
    )
    # Step 0 agrees with a margin 0.5 smaller; step 1 flips the near-tie.
    test = np.array(
        [[2.5, 1.0, 0.0, -1.0], [0.0, 1.9, 2.0, 0.0]], dtype=np.float32
    )
    rows = teacher_forced.compare_steps(ref, test, [1, None], [10, 11])
    self.assertTrue(rows[0]["agree"])
    self.assertAlmostEqual(rows[0]["ref_margin"], 2.0)
    self.assertAlmostEqual(rows[0]["margin_error"], -0.5)
    self.assertAlmostEqual(rows[0]["top_logit_error"], -0.5)
    self.assertFalse(rows[1]["agree"])
    self.assertEqual(rows[1]["test_top"], 2)
    self.assertAlmostEqual(rows[1]["flip_margin"], 0.1, places=5)
    self.assertEqual(rows[1]["position"], 11)
    # KL of identical rows is zero, of different ones positive.
    same = teacher_forced.compare_steps(ref, ref, [None, None], [0, 1])
    self.assertAlmostEqual(same[0]["kl"], 0.0, places=12)
    self.assertGreater(rows[0]["kl"], 0.0)
    # The NLL of the scored token under each backend.
    p = np.exp(ref[0].astype(np.float64))
    self.assertAlmostEqual(
        rows[0]["ref_nll"], -np.log(p[1] / p.sum()), places=6
    )
    self.assertNotIn("ref_nll", rows[1])
    summary = io.StringIO()
    teacher_forced.summarize(rows, [0.5, 1.0], [11], out=summary)
    text = summary.getvalue()
    self.assertIn("top-1 agreement 0.5000", text)
    self.assertIn("[0, 0.5): 1 steps, 1 disagree", text)
    self.assertIn("[11, inf): 1 steps", text)

  def test_collect_from_the_debugger(self):
    with tempfile.TemporaryDirectory() as tmp:
      session = Path(tmp) / "capture" / "0"
      session.mkdir(parents=True)
      for step, position in ((7, 6), (8, 7)):
        write_safetensors(
            session / f"decode_post_logits_step_{step}.safetensors",
            "logits",
            np.full((1, 1, 4), step, dtype=np.float32),
        )
        write_safetensors(
            session / f"decode_pre_input_pos_step_{step}.safetensors",
            "input_pos",
            np.array([position], dtype=np.int32),
        )
      prefix = str(Path(tmp) / "run")
      teacher_forced.collect(str(Path(tmp) / "capture"), prefix, scored=[3, 1])
      logits = np.load(prefix + ".logits.npy")
      self.assertEqual(logits.shape, (2, 4))
      self.assertEqual(logits[1, 0], 8.0)
      steps = json.loads(Path(prefix + ".steps.json").read_text())
      self.assertEqual(
          steps,
          [
              {"step": 7, "position": 6, "scored": 3},
              {"step": 8, "position": 7, "scored": 1},
          ],
      )
      with self.assertRaises(ValueError):
        teacher_forced.collect(str(Path(tmp) / "capture"), prefix, scored=[3])

  def test_collect_from_the_dispatch_dumps(self):
    with tempfile.TemporaryDirectory() as tmp:
      dump = Path(tmp) / "dump"
      lines = []
      vocab = 8
      for seq, (step, values) in enumerate((
          (1, np.arange(vocab, dtype=np.float32)),
          (2, -np.arange(vocab, dtype=np.float16)),
      )):
        name = f"c0/f/step{step}/out_0.raw"
        (dump / name).parent.mkdir(parents=True)
        (dump / name).write_bytes(values.tobytes())
        lines.append({
            "seq": seq,
            "context": 0,
            "function": "f",
            "step": step,
            "dims": [1, 1, vocab],
            "skipped": "",
            "file": name,
        })
      # A cache output of another width is not a step's logits.
      lines.append({
          "seq": 9,
          "context": 0,
          "function": "f",
          "step": 1,
          "dims": [1, 2, 4],
          "skipped": "",
          "file": "",
      })
      (dump / "index.jsonl").write_text(
          "".join(json.dumps(l) + "\n" for l in lines)
      )
      prefix = str(Path(tmp) / "run")
      teacher_forced.collect_dispatch(
          str(dump), prefix, scored=[5, 6], vocab=vocab
      )
      logits = np.load(prefix + ".logits.npy")
      self.assertEqual(logits.dtype, np.float32)
      np.testing.assert_array_equal(logits[0], np.arange(vocab))
      np.testing.assert_array_equal(logits[1], -np.arange(vocab))
      self.assertEqual(
          json.loads(Path(prefix + ".steps.json").read_text()),
          [{"step": 1, "scored": 5}, {"step": 2, "scored": 6}],
      )

  def test_pad_to_shape(self):
    # An INT4 cache of 3 positions out of the model's 5, two values a byte,
    # low nibble first, over the flat tensor: [[1, 2, 3], [-1, 7, 0]].
    raw, length = teacher_forced.pad_to_shape(
        bytes([0x21, 0xF3, 0x07]), (1, 2, 5), 4
    )
    self.assertEqual(length, 3)
    self.assertEqual(raw, bytes([0x21, 0x03, 0xF0, 0x07, 0x00]))
    raw, length = teacher_forced.pad_to_shape(bytes([1, 1]), (1, 1, 1, 4), 8)
    self.assertEqual((raw, length), (bytes([1, 1, 0, 0]), 2))
    with self.assertRaises(ValueError):
      teacher_forced.pad_to_shape(bytes(3), (1, 1, 1, 2), 8)

  def test_replay_of_recorded_steps(self):
    with tempfile.TemporaryDirectory() as tmp:
      dump = Path(tmp) / "io"
      log = Path(tmp) / "run.log"
      lines = [
          "NVIDIA dispatch io in tensorrt_partition_2=input_0_tensor_0 "
          "f32[0 0 0 0] midsum=1\n"
      ]
      for step, position in ((1, 30), (2, 31)):
        for i, tensor in enumerate((1, 0, 2)):
          lines.append(
              "I1115 NVIDIA dispatch io in tensorrt_partition_0="
              f"input_{i}_tensor_{tensor} f32[0 0 0 0] midsum=2\n"
          )
        step_dir = dump / "tensorrt_partition_0" / f"step{step}"
        step_dir.mkdir(parents=True)
        np.array([position], dtype=np.int32).tofile(step_dir / "input_0.raw")
        np.zeros(4, dtype=np.float32).tofile(step_dir / "input_1.raw")
        (step_dir / "input_2.raw").write_bytes(bytes([0x21, 0xF3, 0x07]))
      log.write_text("".join(lines))
      inputs = {
          0: ("embeddings", (1, 4), 32),
          1: ("input_pos", (1,), 32),
          2: ("kv_cache", (1, 2, 5), 4),
      }
      out = Path(tmp) / "replay"
      with mock.patch.object(
          teacher_forced, "signature_inputs", return_value=inputs
      ):
        teacher_forced.replay_manifest(
            str(dump),
            str(log),
            str(out),
            "tensorrt_partition_0",
            "model.tflite",
        )
      manifest = (out / "manifest.txt").read_text().splitlines()
      self.assertEqual(len(manifest), 2)
      fields = manifest[1].split()
      recorded = (dump / "tensorrt_partition_0" / "step2").resolve()
      padded = (out / "inputs" / "step2" / "kv_cache.raw").resolve()
      self.assertEqual(
          fields,
          [
              str((out / "step2").resolve()),
              f"input_pos={recorded / 'input_0.raw'}",
              f"embeddings={recorded / 'input_1.raw'}",
              f"kv_cache={padded}",
          ],
      )
      self.assertEqual(padded.read_bytes(), bytes([0x21, 0x03, 0xF0, 0x07, 0]))
      # What cpu_replay writes for each line.
      for step in (1, 2):
        np.full(4, step, dtype=np.float32).tofile(
            out / f"step{step}.logits.raw"
        )
      prefix = str(Path(tmp) / "cpu")
      teacher_forced.collect_replay(str(out), prefix, scored=[7, 8])
      np.testing.assert_array_equal(
          np.load(prefix + ".logits.npy")[1], np.full(4, 2)
      )
      self.assertEqual(
          json.loads(Path(prefix + ".steps.json").read_text()),
          [
              {"step": 1, "position": 30, "scored": 7},
              {"step": 2, "position": 31, "scored": 8},
          ],
      )


if __name__ == "__main__":
  unittest.main()
