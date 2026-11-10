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

"""Unit tests for reading/parsing quantization recipes."""

import json
import os
import pathlib

from absl.testing import absltest

import os
import io
from litert_quantizer import qtyping
from litert_quantizer.utils import recipe_utils
from litert_quantizer.utils import test_utils


class RecipeUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._recipe_file_path = (
        pathlib.Path(test_utils.get_path_to_datafile("."))
        / "../recipes/dynamic_wi8_afp32_recipe.json"
    )
    with open(self._recipe_file_path, "r") as f:
      self._recipe = json.load(f)

  def test_resolves_recipe_from_absolute_path(self):
    self.assertCountEqual(
        self._recipe,
        recipe_utils.resolve_recipe(self._recipe_file_path.absolute()),
    )

  def test_resolves_recipe_from_relative_path(self):
    self.assertCountEqual(
        self._recipe,
        recipe_utils.resolve_recipe(
            self._recipe_file_path.relative_to(os.getcwd())
        ),
    )

  def test_resolves_recipe_from_fileame_in_recipe_dir(self):
    self.assertCountEqual(
        self._recipe,
        recipe_utils.resolve_recipe(self._recipe_file_path.name),
    )

  def test_resolves_recipe_from_name(self):
    self.assertCountEqual(
        self._recipe, recipe_utils.resolve_recipe("dynamic_wi8_afp32")
    )

  def test_resolve_litertlm_quant_recipe_files_correctly(self):
    # Load a single quantization recipe.
    with open(self._recipe_file_path, "r") as f:
      orig_recipe: qtyping.ModelQuantizationRecipe = json.load(f)

    # Create a copy of the original recipe and mark it as such.
    copied_recipe = [{"copied": True}] + orig_recipe

    # Create an output directory outside of `cwd` and `recipes/` that will
    # contain the LiteRT-LM recipe mapping along with a recipe file.
    output_dir = pathlib.Path(self.create_tempdir()) / "recipes"
    os.mkdir(output_dir)
    litertlm_recipe_path = output_dir / "generated_litertlm_recipe.json"
    copied_recipe_path = output_dir / "copied_recipe.json"

    # Create a litertlm recipe mapping.
    orig_litertlm_recipe = {
        # The original recipe's name (will load from `recipes/`).
        "name": "dynamic_wi8_afp32",
        # The recipe itself (will parse from JSON).
        "json": [{"json": True}] + orig_recipe,
        # The full path of the copied recipe.
        "abspath": str(copied_recipe_path),
        # The name of the file in the `recipes/` directory,
        "recipes": self._recipe_file_path.name,
        # The name of the file in the mapping file's directory,
        "basename": copied_recipe_path.name,
    }

    # Write the copied recipe and the mapping to disk.
    with open(copied_recipe_path, "w") as f:
      json.dump(copied_recipe, f)
    with open(litertlm_recipe_path, "w") as f:
      json.dump(orig_litertlm_recipe, f)

    # Load the LiteRT-LM recipe mapping.
    self.assertIsNotNone(
        litertlm_recipe_mapping := recipe_utils.resolve_litertlm_recipe_or_mapping(
            litertlm_recipe_path
        )
    )

    # Check that the mapped recipes resolve to the corresponding quantization
    # recipes.
    for model_type, recipe in litertlm_recipe_mapping.items():
      match model_type:
        case "name" | "recipes":
          self.assertCountEqual(orig_recipe, recipe)
        case "json":
          self.assertEqual({"json": True}, recipe[0])
          self.assertCountEqual(orig_recipe, recipe[1:])
        case "abspath" | "basename":
          self.assertEqual({"copied": True}, recipe[0])
          self.assertCountEqual(orig_recipe, recipe[1:])
        case _:
          self.fail(f"Unexpected model_type {model_type}.")


if __name__ == "__main__":
  absltest.main()
