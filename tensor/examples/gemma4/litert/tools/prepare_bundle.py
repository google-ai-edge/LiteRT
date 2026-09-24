# Copyright 2026 Google LLC.
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

"""Extract and validate the fixed published E2B bundle for the LiteRT example."""

from collections.abc import Sequence
import os
import pathlib
import subprocess
import sys

from absl import app
from absl import flags

_MODEL = flags.DEFINE_string(
    "model",
    None,
    "Path to the source LiteRT-LM model bundle.",
    required=True,
)
_SCHEMA_DIR = flags.DEFINE_string(
    "schema_dir",
    None,
    "Parent directory of generated tflite/Model.py.",
    required=True,
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "New output directory to populate with audit/ and weights/.",
    required=True,
)


def main(argv: Sequence[str]) -> None:
  """Runs the bundle inspection, export, and validation pipeline."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if not __debug__:
    raise app.UsageError(
        "Run Python without -O: export assertions validate the source."
    )
  root = pathlib.Path(_OUTPUT_DIR.value).resolve()
  root.mkdir(parents=True, exist_ok=False)
  schema = pathlib.Path(_SCHEMA_DIR.value).resolve()
  model_path = pathlib.Path(_MODEL.value).resolve()
  tools = pathlib.Path(__file__).resolve().parent
  audit = root / "audit"
  weights = root / "weights"

  existing_pythonpath = os.environ.get("PYTHONPATH", "")
  pythonpath = (
      f"{schema}{os.pathsep}{existing_pythonpath}"
      if existing_pythonpath
      else str(schema)
  )
  child_env = {**os.environ, "PYTHONPATH": pythonpath}

  def run_step(script: str, *step_flags: str | pathlib.Path) -> None:
    """Executes a tool script with the configured PYTHONPATH."""
    subprocess.run(
        [sys.executable, str(tools / script), *map(str, step_flags)],
        env=child_env,
        check=True,
    )

  inventory = audit / "published-inventory.json"
  trace = audit / "kv-graph-trace.json"
  kv = audit / "kv-placement-summary.json"
  run_step(
      "inspect_bundle.py",
      f"--model={model_path}",
      f"--schema_dir={schema}",
      f"--output_dir={audit}",
  )
  run_step(
      "trace_kv.py",
      f"--inventory={inventory}",
      f"--schema_dir={schema}",
      f"--output={trace}",
  )
  run_step("summarize_kv.py", f"--trace={trace}", f"--output={kv}")
  common = [
      f"--schema_dir={schema}",
      f"--inventory={inventory}",
      f"--trace={trace}",
      f"--kv={kv}",
  ]
  run_step("export_bundle.py", *common, f"--output_dir={weights}")
  run_step(
      "validate_export.py",
      *common,
      f"--bundle_dir={weights}",
      f"--output={audit / 'export-validation.json'}",
  )
  print(weights)


if __name__ == "__main__":
  app.run(main)
