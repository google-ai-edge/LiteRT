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

"""Extract original validation image bytes from downloaded HF Parquet shards."""

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def export(shards, out_dir, max_samples):
    """Write encoded images and a manifest; zero selects all rows."""
    shards = sorted(Path(shards).glob("validation-*.parquet"))
    if not shards:
        raise ValueError("No validation-*.parquet shards found")
    if max_samples < 0:
        raise ValueError("max_samples must be nonnegative")
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with (out_dir / "manifest.tsv").open("w", encoding="utf-8") as manifest:
        manifest.write("id\tlabel\tpath\n")
        for shard in shards:
            for batch in pq.ParquetFile(shard).iter_batches(
                    batch_size=32, columns=["image", "label"]):
                for row in batch.to_pylist():
                    encoded = row["image"]["bytes"]
                    label = int(row["label"])
                    if not encoded or not 0 <= label < 1000:
                        raise ValueError("Expected encoded image and label 0..999")
                    relative = f"images/{count:08d}.JPEG"
                    (out_dir / relative).write_bytes(encoded)
                    manifest.write(f"{count}\t{label}\t{relative}\n")
                    count += 1
                    if max_samples and count >= max_samples:
                        return count
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing downloaded validation Parquet shards")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=3,
                        help="Number to extract; 0 extracts all downloaded validation rows")
    args = parser.parse_args()
    print(f"Wrote {export(args.input_dir, args.out_dir, args.max_samples)} images")
    print(Path(args.out_dir) / "manifest.tsv")


if __name__ == "__main__":
    main()
