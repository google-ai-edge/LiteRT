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

"""A command-line tool to quantize TFLite models using a quantization recipe."""

import argparse
from collections.abc import Sequence
import logging
import pathlib
import tempfile
import sys

import os
import io
from litert_quantizer import qtyping
from litert_quantizer import quantizer
from litert_quantizer.utils import litertlm_utils
from litert_quantizer.utils import progress_utils
from litert_quantizer.utils import recipe_utils


def _verify_output_path(
    output_path: qtyping.Path, overwrite_outputs: bool = False
) -> bool:
  """Checks whether the output file exists and whether it's OK to overwrite it."""
  if os.path.exists(output_path) and not overwrite_outputs:
    overwrite_input = input(
        f"Output file {output_path} already exists. Overwrite? (y/N): "
    )
    return overwrite_input.lower() == "y"
  return True


def _quantize_model(
    model: qtyping.Path | qtyping.BufferType,
    recipe: str | qtyping.ModelQuantizationRecipe,
    serialize_to_path: qtyping.Path | None = None,
    enable_progress_report: bool = True,
) -> quantizer.QuantizationResult:
  """Quantizes the given model with the given recipe and maybe write it to disk."""
  qt = quantizer.Quantizer(model)
  qt.load_quantization_recipe(recipe)
  return qt.quantize(
      serialize_to_path=serialize_to_path,
      enable_progress_report=enable_progress_report,
  )


def quantize_litertlm(
    litertlm_path: str,
    recipe: str,
    output_dir: str | None = None,
    overwrite_outputs: bool = False,
) -> int:
  """Quantizes a TFLite model using a recipe.

  Args:
    litertlm_path: Path to the `.litertlm` file containing the models to be
      quantized.
    recipe: Optional recipe name or path to a .json file containing either a
      recipe or a mapping of model_types to quantization recipes.
    output_dir: Optional path to the directory to save the quantized model(s).
      If `None`, the base directory of `litertlm_path` will be used.
    overwrite_outputs: Output files overwrite exisiting files without requiring
      user input.

  Returns:
    `0` on success and a non-zero exit code otherwise.
  """

  litertlm_path = pathlib.Path(litertlm_path)  # pyrefly: ignore[bad-assignment]
  litertlm_basename = litertlm_path.stem  # pyrefly: ignore[missing-attribute]

  # Load the quantization recipe.
  recipe_map = recipe_utils.resolve_litertlm_recipe_or_mapping(recipe)
  recipe_basename = pathlib.Path(recipe).stem
  default_recipe = recipe_map.get("default")

  # Load the `.litertlm` file.
  litertlm_file = litertlm_utils.LiteRTLMFile(litertlm_path)

  # Create the output directory if it doesn't already exist.
  if not output_dir:
    output_dir = litertlm_path.parent  # pyrefly: ignore[missing-attribute]
  else:
    output_dir = pathlib.Path(output_dir)  # pyrefly: ignore[bad-assignment]
    if not os.path.exists(output_dir):  # pyrefly: ignore[bad-argument-type]
      os.makedirs(output_dir)  # pyrefly: ignore[bad-argument-type]

  # Create temporary files for the `mmap`ed converted TFLite models and clean
  # them up when we're done.
  temp_dir = pathlib.Path(tempfile.gettempdir())
  temp_files: list[pathlib.Path] = []

  # Track progress.
  progress_report = progress_utils.ProgressReport(litertlm_path.name)  # pyrefly: ignore[missing-attribute]
  progress_report.capture_progess_start()

  # Loop over the models selected for quantization and populate a dictionary
  # mapping section IDs to serialized quantized models.
  quantized_sections: dict[int, qtyping.BufferType] = {}
  for section_id, section in enumerate(litertlm_file.sections):

    # Skip non-TFLite sections.
    if section.dataType != litertlm_utils.AnySectionDataType.TFLiteModel:
      continue

    # Try to identify the model type.
    if (model_type := litertlm_file.get_model_type(section_id)) is None:
      logging.warning(
          "Could not get the model_type for the TFLiteModel in section %d,"
          " skipping.",
          section_id,
      )
      continue

    # Get the recipe for this model type, skip if none is given.
    if (
        model_recipe := recipe_map.get(model_type, default_recipe)
    ) is None:
      logging.info(
          "No quantization recipe specified for model_type '%s', skipping.",
          model_type,
      )
      continue

    # Create a filename for the quantized TFLite model.
    output_filename = (
        f"{litertlm_basename}_{section_id:03d}_{model_type}_"
        f"{recipe_basename}.tflite"
    )
    temp_files.append(output_file_path := temp_dir / output_filename)
    if not _verify_output_path(output_file_path, overwrite_outputs):
      logging.error("Aborting.")
      return 1

    # Quantize the TFLite model and write it to disk.
    quantized_sections[section_id] = _quantize_model(  # pyrefly: ignore[unsupported-operation]
        model=litertlm_file.get_section_buffer(section_id),  # pyrefly: ignore[bad-argument-type]
        recipe=model_recipe,
        serialize_to_path=output_file_path,
        enable_progress_report=False,
    ).quantized_model

  if not quantized_sections:
    logging.error("No models were quantized, not creating output file.")
    return 1

  # Rebuild the LiteRT-LM file with the quantized models swapped in.
  output_filename = f"{litertlm_basename}_{recipe_basename}.litertlm"
  output_file_path = output_dir / output_filename  # pyrefly: ignore[unsupported-operation]
  if not _verify_output_path(output_file_path, overwrite_outputs):
    logging.error("Aborting.")
    return 1
  output_file_size = litertlm_file.serialize(
      output_file_path, quantized_sections
  )

  # Clean up temporary files.
  for path in temp_files:
    path.unlink(missing_ok=True)

  progress_report.generate_progress_report(
      os.path.getsize(litertlm_path), output_file_size
  )

  print(f"Wrote quantized model to {output_file_path}.")

  return 0


def quantize_tflite(
    model_file: str,
    recipe_file: str,
    output_dir: str | None = None,
    overwrite_outputs: bool = False,
) -> int:
  """Quantizes a TFLite model using a recipe.

  Args:
    model_file: Path to the .tflite file to be quantized.
    recipe_file: Path to the .json file with the quantization recipe.
    output_dir: Optional path to the directory to save the quantized model(s).
      If `None`, the base directory of `model_file` will be used.
    overwrite_outputs: Output files overwrite exisiting files without requiring
      user input.

  Returns:
    `0` on success and a non-zero exit code otherwise.
  """
  model_file = pathlib.Path(model_file)  # pyrefly: ignore[bad-assignment]
  model_basename = model_file.stem  # pyrefly: ignore[missing-attribute]
  recipe_basename = pathlib.Path(recipe_file).stem
  output_filename = f"{model_basename}_{recipe_basename}.tflite"

  # Create the output directory if it doesn't already exist.
  if not output_dir:
    output_dir = model_file.parent  # pyrefly: ignore[missing-attribute]
  else:
    output_dir = pathlib.Path(output_dir)  # pyrefly: ignore[bad-assignment]
    if not os.path.exists(output_dir):  # pyrefly: ignore[bad-argument-type]
      os.makedirs(output_dir)  # pyrefly: ignore[bad-argument-type]

  output_file_path = str(output_dir / output_filename)  # pyrefly: ignore[unsupported-operation]

  if not _verify_output_path(output_file_path, overwrite_outputs):
    logging.error("Aborting.")
    return 1

  _quantize_model(model_file, recipe_file, serialize_to_path=output_file_path)

  print(f"Wrote quantized model to {output_file_path}.")

  return 0


def parse_args(args: Sequence[str]) -> argparse.Namespace:
  """Parses command-line arguments.

  Args:
    args: A list of strings to parse. If None, sys.argv is used.

  Returns:
    An argparse.Namespace containing the parsed arguments,
  """
  parser = argparse.ArgumentParser(
      description="Quantize TFLite models using a quantization recipe.",
  )
  parser.add_argument(
      "--model_file",
      required=True,
      help="Path to the .tflite or .litertlm file to be quantized.",
  )
  parser.add_argument(
      "--recipe",
      required=True,
      help=(
          "Recipe name or path to the .json file with the quantization recipe."
      ),
  )
  parser.add_argument(
      "--output_dir",
      help=(
          "Path to the directory in which to save the quantized model(s). If"
          " not specified, the directory of the `model_file` argument will be"
          " used."
      ),
  )
  parser.add_argument(
      "--overwrite_outputs",
      action="store_true",
      help="Overwrite exisiting output files without requesting user input.",
  )
  return parser.parse_args(args[1:])


def main(parsed_args: argparse.Namespace) -> int:
  if parsed_args.model_file.endswith(".tflite"):
    return quantize_tflite(
        model_file=parsed_args.model_file,
        recipe_file=parsed_args.recipe,
        output_dir=parsed_args.output_dir,
        overwrite_outputs=parsed_args.overwrite_outputs,
    )
  elif parsed_args.model_file.endswith(".litertlm"):
    return quantize_litertlm(
        litertlm_path=parsed_args.model_file,
        recipe=parsed_args.recipe,
        output_dir=parsed_args.output_dir,
        overwrite_outputs=parsed_args.overwrite_outputs,
    )
  logging.error(
      'File passed to `--model_file` must end in either ".tflite" or'
      ' ".litertlm".'
  )
  return 1


# Wrapper to play nicely with uv tool install litert-quantizer
def cli_main():
  main(parse_args(sys.argv))


if __name__ == "__main__":
  cli_main()
