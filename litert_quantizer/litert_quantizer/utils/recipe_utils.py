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

"""Utilities to read named recipe files."""

from collections.abc import Callable, Collection
import copy
import inspect
import json
import logging
import pathlib
from typing import TypeGuard

import os
import io
from litert_quantizer import qtyping
from litert_quantizer import recipe as recipes

UnresolvedQuantRecipeMapping = dict[str, qtyping.ModelQuantizationRecipe | str]
QuantRecipeMapping = dict[str, qtyping.ModelQuantizationRecipe]
QuantRecipeFileContents = (
    qtyping.ModelQuantizationRecipe | UnresolvedQuantRecipeMapping
)


# Create a mapping of recipe names to parsed JSON files.
_RECIPE_REPO_PATH = pathlib.Path(__file__).parent / '../recipes'
_NAMED_RECIPES: dict[str, Callable[..., QuantRecipeFileContents]] | None = None


def _get_recipe_from_path(
    path: qtyping.Path, extra_dirs: Collection[qtyping.Path] | None = None
) -> QuantRecipeFileContents:
  """Loads a JSON file from an absolute or relative path."""
  path = pathlib.Path(path)

  # Check whether the absolute or relative path exists.
  if (
      os.path.exists(  # Assume this is an absolute or relative (to cwd) path.
          actual_path := path
      )
      or os.path.exists(  # Assume it is relative to the recipes directory.
          actual_path := _RECIPE_REPO_PATH / path
      )
      or (
          extra_dirs
          and any(
              os.path.exists(actual_path := pathlib.Path(dir) / path)
              for dir in extra_dirs
          )
      )
  ):
    logging.info('Loading recipe from %s.', actual_path)
    with open(actual_path, 'r') as f:
      return json.load(f)

  raise ValueError(f'Failed to load/resolve recipe "{path}".')


def _get_named_recipe(
    name: str, default: QuantRecipeFileContents | None = None
) -> QuantRecipeFileContents | None:
  """Looks up a recipe from the `recipes/` directory by its name."""
  global _NAMED_RECIPES

  # Populate the named recipes if this has not been done yet.
  if _NAMED_RECIPES is None:
    # Initialize the named recipes with the functions in the `recipe` module.
    _NAMED_RECIPES = {
        k: v
        for k, v in inspect.getmembers(recipes, inspect.isfunction)
        if not k.startswith('_')
    }

    # Look for additional named recipes in the `recipes/` directory.
    for recipe_path in _RECIPE_REPO_PATH.iterdir():
      if (recipe_name := recipe_path.name).endswith('.json'):
        recipe_name = recipe_name.removesuffix('.json')
        recipe_name = recipe_name.removesuffix('_recipe')
        if recipe_name not in _NAMED_RECIPES:
          with open(recipe_path, 'r') as f:
            recipe = json.load(f)
          _NAMED_RECIPES[recipe_name] = lambda recipe=recipe: copy.deepcopy(
              recipe
          )

    logging.info('Loaded named recipes: %s', _NAMED_RECIPES.keys())

  if recipe_fun := _NAMED_RECIPES.get(name):
    return recipe_fun()
  return default


def resolve_recipe(
    recipe_name_or_path: qtyping.Path,
    extra_dirs: Collection[qtyping.Path] | None = None,
) -> QuantRecipeFileContents:
  """Loads a quantization recipe given by name or path.

  Recipes can be identified either:

    1. By a canonical name, where `name` corresponds to a file
      `recipes/name_recipe.json` in the code's base directory,
    2. By absolute path,
    3. By relative path,
    4. By a file name relative to the code's `recipes/` directory,
    5. By a file name relative to any of the directories in `extra_dirs`,

  in the above order.

  Args:
    recipe_name_or_path: The name of, or path to, a recipe file.
    extra_dirs: An optional collection of extra directories in which to look for
      `recipe_name_or_path`.

  Returns:
    Either a:
      * `qtyping.ModelQuantizationRecipe` if the name or path refered to a
        quantization recipe, or a
      * `dict[str, qtyping.ModelQuantizationRecipe | str]` if the name or path
        refered to a LiteRT-LM mapping from `model_type` to quantization recipe.

  Raises:
    ValueError if the recipe name or path could not be resolved.
  """
  # Check whether this is a named recipe.
  if recipe := _get_named_recipe(recipe_name_or_path):  # pyrefly: ignore[bad-argument-type]
    logging.info(
        'Loading named recipe "%s" from %s.',
        recipe_name_or_path,
        _RECIPE_REPO_PATH,
    )
    return recipe

  # Otherwise, assume we were given a file name/path.
  return _get_recipe_from_path(recipe_name_or_path, extra_dirs)


def _is_quant_recipe(
    recipe: QuantRecipeFileContents,
) -> TypeGuard[qtyping.ModelQuantizationRecipe]:
  return isinstance(recipe, list) and all(
      isinstance(val, dict) and all(isinstance(key, str) for key in val)
      for val in recipe
  )


def _as_quant_recipe_or_none(
    recipe: QuantRecipeFileContents,
) -> qtyping.ModelQuantizationRecipe | None:
  if _is_quant_recipe(recipe):
    return recipe
  else:
    return None


def _is_unresolved_recipe_mapping(
    recipe: QuantRecipeFileContents,
) -> TypeGuard[UnresolvedQuantRecipeMapping]:
  return isinstance(recipe, dict) and all(
      isinstance(key, str)
      and (isinstance(value, str) or _is_quant_recipe(value))
      for key, value in recipe.items()
  )


def resolve_litertlm_recipe_or_mapping(
    recipe_name_or_path: qtyping.Path,
) -> QuantRecipeMapping:
  """Loads a LiteRT-LM quantization recipe given by name or path.

  The named recipe or JSON file is either:
  1. A quantization recipe, which will be used for all models in the LiteRT-LM
    file.
  2. A mapping of strings, corresponding to a `model_type` or `'default'`, to
    either:
    1. A `list[dict[str, Any]]` corresponding to a quantization recipe for the
      `model_type`,
    2. A canonical name, where `name` corresponds to a file
      `recipes/name_recipe.json` in the code's base directory,
    3. A `str` containing the absolute path to a JSON file containing the
      quantization recipe for the `model_type`,
    4. A `str` containing the  path, relative to the current working directory,
      to a JSON file containing the quantization recipe for the `model_type`,
    5. A `str` containing the  path, relative to the directory containing
      `recipe_mapping_path`, to a JSON file containing the quantization recipe
      for the `model_type`.

  Args:
    recipe_name_or_path: The name of, or path to, a JSON file containing the
      default recipe or a per-`model_type` recipe mapping.

  Returns:
    A `QuantRecipeMapping` mapping `model_type` or `'default'` to a quantization
    recipe.

  Raises:
    ValueError if the recipe name or path could not be resolved.
  """
  # Load the recipe or per model_type dictionary of recipes.
  recipe_or_mapping: QuantRecipeFileContents = resolve_recipe(
      recipe_name_or_path
  )

  # If we got a quant recipe, then return a mapping with it set as the default.
  if recipe := _as_quant_recipe_or_none(recipe_or_mapping):  # pyrefly: ignore[bad-assignment]
    logging.info(
        'Using recipe "%s" as the default recipe for all models in the'
        ' LiteRT-LM file.',
        recipe_name_or_path,
    )
    return {'default': recipe}  # pyrefly: ignore[bad-assignment, bad-return]

  if not _is_unresolved_recipe_mapping(recipe_or_mapping):
    raise ValueError(
        f'Expected LiteRT-LM recipe {recipe_name_or_path} to contain a mapping'
        ' of `model_type` to recipes, recipe names, or recipe file paths.'
    )

  # Loop over the map of model_type to recipe and resolve any names/paths.
  extra_dirs = [pathlib.Path(recipe_name_or_path).parent]
  recipe_mapping: QuantRecipeMapping = {}
  for model_type, recipe_or_path in recipe_or_mapping.items():
    if (recipe := _as_quant_recipe_or_none(recipe_or_path)) is None:  # pyrefly: ignore[bad-argument-type, bad-assignment]
      recipe: QuantRecipeFileContents = resolve_recipe(
          recipe_or_path, extra_dirs  # pyrefly: ignore[bad-argument-type]
      )
      if not _is_quant_recipe(recipe):
        raise ValueError(
            f'Expected LiteRT-LM recipe {recipe_name_or_path} to contain a'
            f' recipe name or recipe file path for model_type {model_type}, but'
            f' got object of type {type(recipe)} instead.'
        )
    recipe_mapping[model_type] = recipe  # pyrefly: ignore[unsupported-operation]

  return recipe_mapping
