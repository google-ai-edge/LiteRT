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

"""Materialize ImageNet validation JPEGs and a device-friendly manifest.

The Hugging Face dataset cache stores ImageNet as Arrow shards. This script
exports the original encoded JPEG bytes so Android eval can decode and
preprocess images on-device without storing model-specific tensor copies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Image
from datasets import load_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--cache_dir", default="/data/bt/models/datasets")
    parser.add_argument("--token", default=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "imagenet-1k",
        split="validation",
        cache_dir=args.cache_dir,
        token=args.token,
    )
    ds = ds.cast_column("image", Image(decode=False))
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    manifest = out_dir / "manifest.tsv"
    with manifest.open("w", encoding="utf-8") as f:
        f.write("id\tlabel\tpath\n")
        for i, row in enumerate(ds):
            image = row["image"]
            rel = image["path"] or f"{i:08d}.JPEG"
            rel = Path(rel).name
            data = image["bytes"]
            if data is None:
                raise ValueError(f"Image {i} has no embedded bytes")
            (images_dir / rel).write_bytes(data)
            f.write(f"{i}\t{int(row['label'])}\timages/{rel}\n")

    print(f"Wrote {len(ds)} images")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
