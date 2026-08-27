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
"""Plans and verifies all-TFLite weight externalization for LiteRT-LM."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import tomllib

_SECTION_MARKER = "[[section]]"
_DATA_PATH_RE = re.compile(r'(?m)^data_path\s*=\s*"(?:[^"\\]|\\.)*"')
_PEEK_DATA_TYPE_RE = re.compile(r"Data Type:\s+(\S+)")


def _split_toml(text: str) -> tuple[str, list[str]]:
  parts = text.split(_SECTION_MARKER)
  return parts[0].rstrip(), [
      (_SECTION_MARKER + part).strip() for part in parts[1:]
  ]


def _parse_section(block: str) -> dict[str, object]:
  parsed = tomllib.loads(block)
  sections = parsed.get("section", [])
  if len(sections) != 1:
    raise ValueError(f"Expected one section in TOML block:\n{block}")
  return sections[0]


def _toml_path(path: pathlib.Path, output_toml: pathlib.Path) -> str:
  relative = os.path.relpath(path.resolve(), output_toml.parent.resolve())
  return json.dumps(pathlib.Path(relative).as_posix())


def _replace_data_path(
    block: str, path: pathlib.Path, output_toml: pathlib.Path
) -> str:
  replacement = f"data_path = {_toml_path(path, output_toml)}"
  rewritten, count = _DATA_PATH_RE.subn(replacement, block, count=1)
  if count != 1:
    raise ValueError(f"Expected exactly one data_path in section:\n{block}")
  return rewritten


def build_manifest(
    input_toml: pathlib.Path, output_manifest: pathlib.Path
) -> int:
  """Writes one manifest row for every TFLiteModel section."""
  _, blocks = _split_toml(input_toml.read_text(encoding="utf-8"))
  rows: list[tuple[int, str, str]] = []
  seen_model_types: set[str] = set()
  for section_index, block in enumerate(blocks):
    section = _parse_section(block)
    if section.get("section_type") != "TFLiteModel":
      continue
    model_type = section.get("model_type")
    data_path = section.get("data_path")
    if not isinstance(model_type, str) or not model_type:
      raise ValueError(f"TFLiteModel section {section_index} has no model_type")
    if model_type in seen_model_types:
      raise ValueError(f"Duplicate TFLiteModel model_type: {model_type}")
    if not isinstance(data_path, str) or not data_path:
      raise ValueError(f"TFLiteModel section {section_index} has no data_path")
    seen_model_types.add(model_type)
    source_path = (input_toml.parent / data_path).resolve()
    if not source_path.is_file():
      raise FileNotFoundError(
          f"TFLiteModel data_path does not exist: {source_path}"
      )
    rows.append((section_index, model_type, str(source_path)))

  output_manifest.parent.mkdir(parents=True, exist_ok=True)
  with output_manifest.open("w", encoding="utf-8", newline="") as output:
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow(("section_index", "model_type", "input_model"))
    writer.writerows(rows)
  return len(rows)


def _read_results(
    results_path: pathlib.Path,
) -> dict[str, tuple[pathlib.Path, pathlib.Path | None]]:
  """Reads the externalization results TSV keyed by model_type."""
  results: dict[str, tuple[pathlib.Path, pathlib.Path | None]] = {}
  with results_path.open("r", encoding="utf-8", newline="") as source:
    reader = csv.DictReader(source, delimiter="\t")
    expected = {"section_index", "model_type", "model_path", "weights_path"}
    if set(reader.fieldnames or []) != expected:
      raise ValueError(
          f"Unexpected results columns {reader.fieldnames}; expected"
          f" {sorted(expected)}"
      )
    for row in reader:
      model_type = row["model_type"]
      if model_type in results:
        raise ValueError(f"Duplicate externalization result: {model_type}")
      model_path = pathlib.Path(row["model_path"])
      weights_path = (
          pathlib.Path(row["weights_path"]) if row["weights_path"] else None
      )
      if not model_path.is_file():
        raise FileNotFoundError(
            f"Processed TFLite does not exist: {model_path}"
        )
      if weights_path is not None and not weights_path.is_file():
        raise FileNotFoundError(f"Weights file does not exist: {weights_path}")
      results[model_type] = (model_path, weights_path)
  return results


def rewrite_toml(
    input_toml: pathlib.Path,
    results_path: pathlib.Path,
    output_toml: pathlib.Path,
) -> tuple[int, int]:
  """Preserves section order and appends external-weight sections at EOF."""
  prefix, blocks = _split_toml(input_toml.read_text(encoding="utf-8"))
  results = _read_results(results_path)
  model_sections: list[str] = []
  weight_sections_by_model: dict[str, str] = {}
  encountered_models: set[str] = set()

  for block in blocks:
    section = _parse_section(block)
    section_type = section.get("section_type")
    model_type = section.get("model_type")

    if section_type == "TFLiteWeights":
      if not isinstance(model_type, str) or not model_type:
        raise ValueError("TFLiteWeights section has no model_type")
      data_path = section.get("data_path")
      if not isinstance(data_path, str) or not data_path:
        raise ValueError("TFLiteWeights section has no data_path")
      block = _replace_data_path(
          block,
          (input_toml.parent / data_path).resolve(),
          output_toml,
      )
      weight_sections_by_model[model_type] = block
      continue

    data_path = section.get("data_path")
    if isinstance(data_path, str):
      block = _replace_data_path(
          block, (input_toml.parent / data_path).resolve(), output_toml
      )

    if section_type == "TFLiteModel":
      if not isinstance(model_type, str) or not model_type:
        raise ValueError("TFLiteModel section has no model_type")
      encountered_models.add(model_type)
      if model_type not in results:
        raise ValueError(
            f"No externalization result for TFLiteModel {model_type}"
        )
      model_path, weights_path = results[model_type]
      if weights_path is not None and weights_path.stat().st_size > 0:
        block = _replace_data_path(block, model_path, output_toml)
        weight_sections_by_model[model_type] = "\n".join((
            _SECTION_MARKER,
            f"model_type = {json.dumps(model_type)}",
            'section_type = "TFLiteWeights"',
            f"data_path = {_toml_path(weights_path, output_toml)}",
        ))

    model_sections.append(block)

  missing = sorted(set(results) - encountered_models)
  if missing:
    raise ValueError(
        f"Results do not match a TFLiteModel: {', '.join(missing)}"
    )

  ordered_weights = [
      weight_sections_by_model[model_type]
      for model_type in results
      if model_type in weight_sections_by_model
  ]
  output_blocks = [
      block
      for block in (
          prefix,
          *model_sections,
          *ordered_weights,
      )
      if block
  ]
  output_toml.parent.mkdir(parents=True, exist_ok=True)
  output_toml.write_text(
      "\n\n".join(output_blocks).rstrip() + "\n", encoding="utf-8"
  )
  return len(encountered_models), len(ordered_weights)


def verify_peek_layout(
    peek_path: pathlib.Path, expected_models: int, expected_weights: int
) -> None:
  """Verifies peek output has the expected, EOF-contiguous weight layout."""
  data_types = _PEEK_DATA_TYPE_RE.findall(peek_path.read_text(encoding="utf-8"))
  actual_models = data_types.count("TFLiteModel")
  actual_weights = data_types.count("TFLiteWeights")
  if actual_models != expected_models:
    raise ValueError(
        f"Expected {expected_models} TFLiteModel sections, found"
        f" {actual_models}"
    )
  if actual_weights != expected_weights:
    raise ValueError(
        f"Expected {expected_weights} TFLiteWeights sections, found"
        f" {actual_weights}"
    )
  if actual_weights:
    first_weight = data_types.index("TFLiteWeights")
    trailing_types = data_types[first_weight:]
    if any(data_type != "TFLiteWeights" for data_type in trailing_types):
      raise ValueError(
          "All TFLiteWeights sections must form a contiguous EOF suffix"
      )


def _parse_args() -> argparse.Namespace:
  """Parses command-line arguments for the manifest/rewrite/verify commands."""
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="command", required=True)

  manifest = subparsers.add_parser("manifest")
  manifest.add_argument("--toml", type=pathlib.Path, required=True)
  manifest.add_argument("--output", type=pathlib.Path, required=True)

  rewrite = subparsers.add_parser("rewrite")
  rewrite.add_argument("--toml", type=pathlib.Path, required=True)
  rewrite.add_argument("--results", type=pathlib.Path, required=True)
  rewrite.add_argument("--output", type=pathlib.Path, required=True)

  verify = subparsers.add_parser("verify-peek")
  verify.add_argument("--peek", type=pathlib.Path, required=True)
  verify.add_argument("--expected-models", type=int, required=True)
  verify.add_argument("--expected-weights", type=int, required=True)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.command == "manifest":
    count = build_manifest(args.toml, args.output)
    print(f"TFLiteModel sections: {count}")
  elif args.command == "rewrite":
    model_count, weight_count = rewrite_toml(
        args.toml, args.results, args.output
    )
    print(f"TFLiteModel sections: {model_count}")
    print(f"Externalized weight sections: {weight_count}")
  else:
    verify_peek_layout(args.peek, args.expected_models, args.expected_weights)
    print("Verified all TFLiteWeights sections form a contiguous EOF suffix")


if __name__ == "__main__":
  main()
