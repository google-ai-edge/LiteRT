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

"""Command-line interface for LiteRT Python ahead-of-time (AOT) compilation."""

from collections.abc import Callable, Sequence
import json
import pathlib
import tempfile
from typing import Any

from absl import app
from absl import flags

from litert.python.aot import aot_compile as aot_compile_lib
from litert.python.aot.core import aot_types
from litert.python.aot.vendors import fallback_backend
from litert.python.aot.vendors import import_vendor
from litert.python.aot.vendors.google_tensor import target as google_tensor_target
from litert.python.aot.vendors.intel_openvino import target as intel_openvino_target
from litert.python.aot.vendors.mediatek import target as mediatek_target
from litert.python.aot.vendors.qualcomm import target as qualcomm_target
from litert.python.aot.vendors.samsung import target as samsung_target

_INPUT_MODEL = flags.DEFINE_string(
    "input_model",
    "",
    "Path to the input .tflite model file to compile.",
)
_MODEL = flags.DEFINE_string(
    "model",
    "",
    "Alias for --input_model (matching apply_plugin_main).",
)
_OUTPUT_MODEL = flags.DEFINE_string(
    "output_model",
    "",
    "Exact output .tflite file path when compiling for a single target.",
)
_OUTPUT_SHORT = flags.DEFINE_string(
    "o",
    "",
    "Alias for --output_model (matching apply_plugin_main).",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "",
    "Directory to write compiled .tflite model(s) to.",
)
_BACKEND = flags.DEFINE_string(
    "backend",
    "",
    "Backend ID or manufacturer name (e.g. 'qualcomm', 'mediatek',"
    " 'google_tensor', 'samsung', 'intel_openvino', 'fallback', 'all')."
    " Defaults to 'all'.",
)
_SOC_MANUFACTURER = flags.DEFINE_string(
    "soc_manufacturer",
    "",
    "Alias for --backend (matching apply_plugin_main).",
)
_SOC_MODEL = flags.DEFINE_string(
    "soc_model",
    "ALL",
    "Target SoC model (e.g. 'SM8750', 'MT6989', 'Tensor_G5', 'E9965', 'LNL',"
    " 'ALL').",
)
_SUBGRAPHS_TO_COMPILE = flags.DEFINE_list(
    "subgraphs_to_compile",
    [],
    "Optional comma-separated subgraph indices to compile (e.g. '0,1').",
)
_SUBGRAPHS = flags.DEFINE_list(
    "subgraphs",
    [],
    "Alias for --subgraphs_to_compile (matching apply_plugin_main).",
)
_KEEP_GOING = flags.DEFINE_bool(
    "keep_going",
    False,
    "Whether to continue compiling remaining targets when a target fails.",
)
_BACKEND_OPTION = flags.DEFINE_multi_string(
    "backend_option",
    [],
    "Repeatable key=value option forwarded to aot_compile (e.g."
    " --backend_option=quantize_recipe=...).",
)

_BACKEND_ALIASES: dict[str, str] = {
    "": "all",
    "google": "GOOGLE",
    "google_tensor": "GOOGLE",
    "intel": "intel_openvino",
    "intelopenvino": "intel_openvino",
}

_VENDOR_TARGETS: dict[str, tuple[Callable[[Any], aot_types.Target], Any]] = {
    "GOOGLE": (google_tensor_target.Target, google_tensor_target.SocModel),
    "intel_openvino": (
        intel_openvino_target.Target,
        intel_openvino_target.SocModel,
    ),
    "mediatek": (mediatek_target.Target, mediatek_target.SocModel),
    "qualcomm": (qualcomm_target.Target, qualcomm_target.SocModel),
    "samsung": (samsung_target.Target, samsung_target.SocModel),
}


def normalize_backend_id(raw_backend: str) -> str:
  """Returns the canonical backend ID for `raw_backend`."""
  lowered = raw_backend.strip().lower()
  return _BACKEND_ALIASES.get(lowered, lowered)


def _parse_option_value(raw_value: str) -> Any:
  """Parses a CLI option value string into bool, None, JSON, or str."""
  stripped = raw_value.strip()
  lowered = stripped.lower()
  if lowered == "true":
    return True
  if lowered == "false":
    return False
  if lowered in ("none", "null"):
    return None
  if stripped.startswith(("{", "[")):
    try:
      return json.loads(stripped)
    except json.JSONDecodeError as error:
      raise ValueError(
          f"Invalid JSON in backend option value {raw_value!r}: {error}"
      ) from error
  return stripped


def parse_backend_options(
    entries: Sequence[str],
    *,
    backend_id: str = "",
) -> dict[str, Any]:
  """Parses repeatable `key=value` CLI options into a dictionary.

  Args:
    entries: Sequence of raw `key=value` option strings from the CLI.
    backend_id: Target backend ID used to determine whether vendor-specific key
      prefixes should be applied.

  Returns:
    A mapping of option keys to coerced option values (bool, None, JSON object,
    or str).

  Raises:
    ValueError: If an entry is missing '=', has an empty key, or contains
      invalid JSON.
  """
  _ = backend_id  # Used only in vendor-specific blocks below.
  parsed: dict[str, Any] = {}
  for entry in entries:
    if "=" not in entry:
      raise ValueError(
          f"Invalid --backend_option {entry!r}; expected 'key=value'."
      )
    raw_key, raw_value = entry.split("=", 1)
    key = raw_key.strip()
    if not key:
      raise ValueError(
          f"Invalid --backend_option {entry!r}; option key must not be empty."
      )
    parsed[key] = _parse_option_value(raw_value)
  return parsed


def parse_subgraph_indices(entries: Sequence[str]) -> list[int] | None:
  """Parses subgraph index strings into a list of non-negative integers.

  Args:
    entries: Raw subgraph index tokens from the command line.

  Returns:
    A list of non-negative subgraph indices, or `None` if `entries` is empty or
    contains only whitespace tokens (indicating all subgraphs should be
    compiled).

  Raises:
    ValueError: If any non-empty token is not a non-negative integer.
  """
  indices: list[int] = []
  for token in entries:
    cleaned = token.strip()
    if not cleaned:
      continue
    try:
      index = int(cleaned)
    except ValueError as error:
      raise ValueError(
          f"Invalid subgraph index {cleaned!r}; expected a non-negative"
          " integer."
      ) from error
    if index < 0:
      raise ValueError(f"Subgraph index must be non-negative, got {index}.")
    indices.append(index)
  return indices or None


def resolve_target(backend_id: str, soc_model: str = "ALL") -> aot_types.Target:
  """Resolves `backend_id` and `soc_model` into an AOT `Target`.

  Args:
    backend_id: Backend identifier or alias (e.g. 'all', 'qualcomm',
      'fallback').
    soc_model: Target SoC model string. Defaults to 'ALL'.

  Returns:
    The resolved `aot_types.Target` instance.

  Raises:
    ValueError: If `soc_model` is specified when `backend_id` is 'all' or
      'fallback', or if `backend_id` is unrecognized.
  """
  canonical_id = normalize_backend_id(backend_id)
  raw_soc = soc_model.strip()
  cleaned_soc = "ALL" if not raw_soc or raw_soc.upper() == "ALL" else raw_soc

  if canonical_id in ("all", "fallback"):
    if cleaned_soc != "ALL":
      raise ValueError(
          f"--soc_model cannot be specified with --backend={canonical_id}"
          f" (got {soc_model!r})."
      )
    return (
        import_vendor.AllRegisteredTarget()
        if canonical_id == "all"
        else fallback_backend.FallbackTarget()
    )

  if canonical_id in _VENDOR_TARGETS:
    target_cls, soc_model_cls = _VENDOR_TARGETS[canonical_id]
    return target_cls(soc_model_cls(cleaned_soc))

  raise ValueError(f"Unsupported backend: {backend_id!r}.")


def _resolve_aliased_flag(
    primary_value: str,
    alias_value: str,
    primary_name: str,
    alias_name: str,
) -> str:
  """Returns the active flag value and rejects conflicting alias values."""
  primary = primary_value.strip()
  alias = alias_value.strip()
  if primary and alias and primary != alias:
    raise ValueError(
        f"Cannot specify conflicting values for --{primary_name}={primary!r}"
        f" and --{alias_name}={alias!r}."
    )
  return primary or alias


def compile_from_cli(
    input_model: str | pathlib.Path,
    *,
    output_model: str | pathlib.Path | None = None,
    output_dir: str | pathlib.Path | None = None,
    backend: str = "all",
    soc_model: str = "ALL",
    subgraphs_to_compile: Sequence[int] | None = None,
    keep_going: bool = False,
    backend_options: Sequence[str] = (),
) -> aot_types.CompilationResult:
  """Compiles `input_model` via `aot_compile` and writes requested outputs.

  Args:
    input_model: Path to the input `.tflite` model file to compile.
    output_model: Exact destination file path when compiling for a single
      target. If `None`, outputs are saved to `output_dir`.
    output_dir: Directory to write compiled models to. If `None` and
      `output_model` is `None`, outputs are written to a `_compiled_models`
      directory next to `input_model`.
    backend: Backend identifier or manufacturer name (defaults to 'all').
    soc_model: Target SoC model string (defaults to 'ALL').
    subgraphs_to_compile: Sequence of subgraph indices to compile, or `None` to
      compile all subgraphs.
    keep_going: Whether to continue compiling remaining targets when one fails.
    backend_options: Sequence of raw `key=value` option strings forwarded to the
      compiler backend.

  Returns:
    The `aot_types.CompilationResult` containing compiled models and reports.

  Raises:
    FileNotFoundError: If `input_model` does not exist.
    ValueError: If both `output_model` and `output_dir` are specified, or if
      `output_model` is specified when compilation produces multiple targets.
    RuntimeError: If compilation produces no output models.
  """
  if output_model and output_dir:
    raise ValueError(
        "Cannot specify both --output_model (--o) and --output_dir."
    )

  input_path = pathlib.Path(input_model)
  if not input_path.is_file():
    raise FileNotFoundError(f"Input model not found: {input_path}")

  canonical_backend = normalize_backend_id(backend)
  target = resolve_target(canonical_backend, soc_model)
  extra_kwargs = parse_backend_options(
      backend_options, backend_id=canonical_backend
  )
  output_model_path = pathlib.Path(output_model) if output_model else None
  subgraphs = (
      list(subgraphs_to_compile) if subgraphs_to_compile is not None else None
  )

  if output_model_path is not None:
    with tempfile.TemporaryDirectory(prefix="litert_aot_cli_") as tmp_dir:
      result = aot_compile_lib.aot_compile(
          str(input_path),
          output_dir=pathlib.Path(tmp_dir),
          target=target,
          keep_going=keep_going,
          subgraphs_to_compile=subgraphs,
          **extra_kwargs,
      )
      _check_compilation_has_outputs(result, input_path)
      if len(result.models_with_backend) != 1:
        raise ValueError(
            "--output_model (--o) requires a single compiled target, but"
            f" {len(result.models_with_backend)} models were produced. Specify"
            " a concrete --soc_model or use --output_dir instead."
        )
      output_model_path.parent.mkdir(parents=True, exist_ok=True)
      _, compiled_model = result.models_with_backend[0]
      compiled_model.save(output_model_path, export_only=True)
      compiled_model.set_path(output_model_path)
    return result

  effective_output_dir = pathlib.Path(output_dir) if output_dir else None
  result = aot_compile_lib.aot_compile(
      str(input_path),
      output_dir=effective_output_dir,
      target=target,
      keep_going=keep_going,
      subgraphs_to_compile=subgraphs,
      **extra_kwargs,
  )
  _check_compilation_has_outputs(result, input_path)

  dest_dir = (
      effective_output_dir
      if effective_output_dir is not None
      else input_path.parent / "_compiled_models"
  )
  dest_dir.mkdir(parents=True, exist_ok=True)
  for compiled_backend, compiled_model in result.models_with_backend:
    if compiled_model.in_memory:
      filename = (
          f"{input_path.stem}"
          f"{compiled_backend.target_id_suffix}_apply_plugin.tflite"
      )
      dest_path = dest_dir / filename
      compiled_model.save(dest_path)

  return result


def _check_compilation_has_outputs(
    result: aot_types.CompilationResult,
    input_path: pathlib.Path,
) -> None:
  """Raises `RuntimeError` if `result` produced no compiled models."""
  if result.models_with_backend:
    return
  reasons = "; ".join(
      f"{failed_backend.target_id or failed_backend.id()}: {error}"
      for failed_backend, error in result.failed_backends
  )
  detail = f": {reasons}" if reasons else "."
  raise RuntimeError(
      f"AOT compilation produced no output models for {input_path}{detail}"
  )


def main(argv: Sequence[str]) -> None:
  """Executes the LiteRT Python ahead-of-time (AOT) compilation CLI."""
  if len(argv) > 1:
    raise app.UsageError(f"Unexpected positional arguments: {argv[1:]}")

  try:
    input_model = _resolve_aliased_flag(
        _INPUT_MODEL.value, _MODEL.value, "input_model", "model"
    )
    if not input_model:
      raise app.UsageError(
          "Must specify --input_model (or --model) with the .tflite path."
      )
    output_model = _resolve_aliased_flag(
        _OUTPUT_MODEL.value, _OUTPUT_SHORT.value, "output_model", "o"
    )
    backend = _resolve_aliased_flag(
        _BACKEND.value,
        _SOC_MANUFACTURER.value,
        "backend",
        "soc_manufacturer",
    )
    if _SUBGRAPHS_TO_COMPILE.value and _SUBGRAPHS.value:
      if _SUBGRAPHS_TO_COMPILE.value != _SUBGRAPHS.value:
        raise app.UsageError(
            "Cannot specify conflicting values for --subgraphs_to_compile and"
            " --subgraphs."
        )
    raw_subgraphs = _SUBGRAPHS_TO_COMPILE.value or _SUBGRAPHS.value
    subgraphs = parse_subgraph_indices(raw_subgraphs)

    result = compile_from_cli(
        input_model,
        output_model=output_model or None,
        output_dir=_OUTPUT_DIR.value.strip() or None,
        backend=backend or "all",
        soc_model=_SOC_MODEL.value,
        subgraphs_to_compile=subgraphs,
        keep_going=_KEEP_GOING.value,
        backend_options=_BACKEND_OPTION.value,
    )
  except (FileNotFoundError, ValueError) as error:
    raise app.UsageError(str(error)) from error

  print(result.compilation_report())
  for compiled_backend, compiled_model in result.models_with_backend:
    if not compiled_model.in_memory:
      print(
          f"Output [{compiled_backend.target_id or compiled_backend.id()}]:"
          f" {compiled_model.path}"
      )


if __name__ == "__main__":
  app.run(main)
