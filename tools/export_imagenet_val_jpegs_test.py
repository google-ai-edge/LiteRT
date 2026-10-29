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

"""Local Parquet export checks; no dataset download or image decoding."""
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from export_imagenet_val_jpegs import export


class ExportTest(unittest.TestCase):
    def test_bounded_export_preserves_bytes_and_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shards = root / "shards"
            shards.mkdir()
            for number in range(2):
                rows = [{"image": {"bytes": f"image-{number}-{i}".encode(),
                                   "path": "same-name.JPEG"}, "label": number * 2 + i}
                        for i in range(2)]
                pq.write_table(pa.Table.from_pylist(rows),
                               shards / f"validation-{number:05d}.parquet")
            self.assertEqual(export(shards, root / "out", 3), 3)
            self.assertEqual((root / "out/manifest.tsv").read_text(),
                             "id\tlabel\tpath\n0\t0\timages/00000000.JPEG\n"
                             "1\t1\timages/00000001.JPEG\n2\t2\timages/00000002.JPEG\n")
            self.assertEqual((root / "out/images/00000002.JPEG").read_bytes(), b"image-1-0")
            self.assertEqual(export(shards, root / "all", 0), 4)
            with self.assertRaises(ValueError):
                export(shards, root / "bad", -1)
            with self.assertRaises(ValueError):
                export(root / "missing", root / "bad", 1)

    def test_invalid_labels_or_missing_bytes_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for row in [{"image": {"bytes": b"", "path": ""}, "label": 0},
                        {"image": {"bytes": b"image", "path": ""}, "label": 1000}]:
                pq.write_table(pa.Table.from_pylist([row]), root / "validation-00000.parquet")
                with self.assertRaises(ValueError):
                    export(root, root / "output", 1)


if __name__ == "__main__":
    unittest.main()
