# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Module for extracting object files from a compiled archive (.a) file.

This module provides functionality almost identical to the 'ar -x' command,
which extracts out all object files from a given archive file. This module
assumes the archive is in the BSD variant format used in Apple platforms.

See: https://en.wikipedia.org/wiki/Ar_(Unix)#BSD_variant

This extractor has two important differences compared to the 'ar -x' command
shipped with Xcode.

1.  When there are multiple object files with the same name in a given archive,
    each file is renamed so that they are all correctly extracted without
    overwriting each other.

2.  This module takes the destination directory as an additional parameter.

    Example Usage:

    archive_path = ...
    dest_dir = ...
    extract_object_files(archive_path, dest_dir)
"""

import hashlib
import io
import itertools
from pathlib import Path
import struct
from typing import Iterator, Tuple


ARCHIVE_SIGNATURE = b"!<arch>\n"
HEADER_STRUCT = struct.Struct("=16s12s6s6s8s10s2s")


def extract_object_files(archive_file: io.BufferedIOBase, dest_dir: str) -> None:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    _check_archive_signature(archive_file)

    extracted_files = {}

    for name, content in _extract_next_file(archive_file):
        digest = hashlib.md5(content).digest()

        for final_name in _generate_modified_filenames(name):
            if final_name not in extracted_files:
                extracted_files[final_name] = digest
                (dest / final_name).write_bytes(content)
                break

            if extracted_files[final_name] == digest:
                break


def _generate_modified_filenames(filename: str) -> Iterator[str]:
    yield filename

    base, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")

    for i in itertools.count(1):
        yield f"{base}_{i}.{ext}" if ext else f"{base}_{i}"


def _check_archive_signature(archive_file: io.BufferedIOBase) -> None:
    if archive_file.read(8) != ARCHIVE_SIGNATURE:
        raise RuntimeError("Invalid archive file format.")


def _extract_next_file(
    archive_file: io.BufferedIOBase
) -> Iterator[Tuple[str, bytes]]:

    while True:
        header = archive_file.read(60)

        if not header:
            return

        if len(header) != 60:
            raise RuntimeError("Invalid file header format.")

        name, _, _, _, _, size, end = HEADER_STRUCT.unpack(header)

        if end != b"`\n":
            raise RuntimeError("Invalid file header format.")

        name = name.decode("ascii").strip()
        size = int(size)
        odd = size & 1

        if name.startswith("#1/"):
            filename_size = int(name[3:])
            name = archive_file.read(filename_size).decode("utf-8").strip(" \x00")
            size -= filename_size

        content = archive_file.read(size)

        if odd:
            archive_file.read(1)

        yield name, content
