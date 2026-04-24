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
"""Build helper for LiteRT acceleration configuration schemas."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _read_text(path):
  return Path(path).read_text(encoding="utf-8")


def _write_ascii(path, contents):
  output_path = Path(path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", encoding="ascii", newline="\n") as output_file:
    output_file.write(contents)


def _run_generate_schema(args):
  source_path = Path(args.src)
  proto_name = args.proto_name or source_path.name
  if not proto_name.endswith(".proto"):
    raise ValueError("--proto_name must end with .proto")

  with tempfile.TemporaryDirectory(prefix="litert_configuration_schema_") as tmp:
    tmp_path = Path(tmp)
    proto_path = tmp_path / proto_name
    shutil.copyfile(source_path, proto_path)

    subprocess.run(
        [args.flatc, "--proto", "-o", str(tmp_path), str(proto_path)],
        check=True,
    )

    generated_name = proto_name[:-len(".proto")] + ".fbs"
    contents = _read_text(tmp_path / generated_name)
    contents = contents.replace("tflite.proto", "tflite")
    _write_ascii(args.out, contents)


def _run_generate_header(args):
  schema_contents = _read_text(args.src)
  header_contents = (
      f'constexpr char {args.variable_name}[] = R"Delimiter(\n'
      f"{schema_contents})Delimiter\";\n"
  )
  _write_ascii(args.out, header_contents)


def main(argv):
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="command")

  generate_schema = subparsers.add_parser("generate_schema")
  generate_schema.add_argument("--flatc", required=True)
  generate_schema.add_argument("--src", required=True)
  generate_schema.add_argument("--out", required=True)
  generate_schema.add_argument("--proto_name")
  generate_schema.set_defaults(func=_run_generate_schema)

  generate_header = subparsers.add_parser("generate_header")
  generate_header.add_argument("--src", required=True)
  generate_header.add_argument("--out", required=True)
  generate_header.add_argument("--variable_name", required=True)
  generate_header.set_defaults(func=_run_generate_header)

  args = parser.parse_args(argv)
  if not hasattr(args, "func"):
    parser.error("missing command")
  args.func(args)
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
