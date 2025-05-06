# Copyright 2025 The AI Edge LiteRT Authors.
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

"""Wrapper to create sdist using setup.py."""

import argparse
import glob
import os
import shutil
import subprocess
import sys


def main():
  parser = argparse.ArgumentParser(
      description="Wrapper to create sdist using setup.py."
  )
  parser.add_argument(
      "--dir",
      required=True,
      help="Directory containing the project files.",
  )
  parser.add_argument(
      "--setup_py",
      default="setup.py",
      help="Name of the setup script (default: setup.py).",
  )
  parser.add_argument(
      "--output_sdist_path",
      required=True,
      help="Full path where the final sdist .tar.gz should be placed.",
  )

  args = parser.parse_args()

  original_cwd = os.getcwd()
  sdist_temp_output_dir = "dist_sdist_temp"

  try:
    print(f"Changing working directory to: {args.dir}")
    os.chdir(args.dir)

    if os.path.exists(sdist_temp_output_dir):
      shutil.rmtree(sdist_temp_output_dir)
    os.makedirs(sdist_temp_output_dir)

    cmd = [
        sys.executable,
        args.setup_py,
        "sdist",
        "--dist-dir",
        sdist_temp_output_dir,
    ]

    print(f"Running command: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if process.returncode != 0:
      print("Error running setup.py sdist:", file=sys.stderr)
      print(f"Stdout:\n{process.stdout}", file=sys.stderr)
      print(f"Stderr:\n{process.stderr}", file=sys.stderr)
      sys.exit(process.returncode)
    else:
      print("setup.py sdist ran successfully.")
      if process.stdout:
        print(f"Stdout:\n{process.stdout}")
      if process.stderr:
        print(f"Stderr:\n{process.stderr}")

    # Find the generated .tar.gz file
    # sdist usually creates <package_name>-<version>.tar.gz
    sdist_files = glob.glob(os.path.join(sdist_temp_output_dir, "*.tar.gz"))

    if not sdist_files:
      print(
          f"Error: No .tar.gz file found in {sdist_temp_output_dir}",
          file=sys.stderr,
      )
      print(
          f"Contents of {sdist_temp_output_dir}:"
          f" {os.listdir(sdist_temp_output_dir)}",
          file=sys.stderr,
      )
      sys.exit(1)
    if len(sdist_files) > 1:
      print(
          f"Warning: Multiple .tar.gz files found in {sdist_temp_output_dir}."
          f" Using the first one: {sdist_files[0]}",
          file=sys.stderr,
      )

    actual_sdist_file = sdist_files[0]
    print(f"Found sdist archive: {actual_sdist_file}")

    final_output_path_abs = os.path.join(original_cwd, args.output_sdist_path)
    final_output_dir_abs = os.path.dirname(final_output_path_abs)

    if not os.path.exists(final_output_dir_abs):
      os.makedirs(final_output_dir_abs)

    print(f"Moving {actual_sdist_file} to {final_output_path_abs}")
    shutil.move(actual_sdist_file, final_output_path_abs)

    print(f"Successfully created sdist: {final_output_path_abs}")

  finally:
    os.chdir(original_cwd)


if __name__ == "__main__":
  main()
