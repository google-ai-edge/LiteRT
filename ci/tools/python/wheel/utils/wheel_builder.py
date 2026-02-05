# Copyright 2025 The Tensorflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This script is used to build a python wheel from a list of source files.

It takes a list of source files, a setup.py file, and a version string as input.
It then uses a python script, wheel_builder.py, to generate the wheel file. The
wheel builder binary is responsible for preparing the build environment and
calling the setuptools command to generate the wheel file.

args:
  --setup_py: Path to the setup.py file.
  --output: Output directory for the wheel.
  --version: Version of the wheel.
  --src: List of source files for the wheel.
  --platform: Platform name to be passed to build module.

output:
  A python wheel file is created in the output directory. The name of the wheel
  file is based on various factors, including the version and platform.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from typing import Optional


def parse_args() -> argparse.Namespace:
  """Arguments parser."""
  parser = argparse.ArgumentParser(
      description="Helper for building python wheel from pyproject.toml",
      fromfile_prefix_chars="@",
  )
  parser.add_argument(
      "--project_name",
      required=True,
      help="Name of the project",
  )
  parser.add_argument("--pyproject", help="location of pyproject.toml file")
  parser.add_argument("--setup_py", help="location of setup.py file")
  parser.add_argument("--output", help="output directory")
  parser.add_argument("--version", help="version of the wheel")
  parser.add_argument(
      "--src", help="single source file for the wheel", action="append"
  )
  parser.add_argument(
      "--py_src", help="single source file for the wheel", action="append"
  )
  parser.add_argument(
      "--structured_deps",
      help="single structured dep for the wheel",
      action="append",
  )
  parser.add_argument(
      "--package_data", help="single source data for the wheel", action="append"
  )
  parser.add_argument(
      "--platform",
      required=True,
      help="Platform name to be passed to build module",
  )
  parser.add_argument(
      "--nightly_suffix",
      help=(
          "Suffix to be added to the name of the wheel for nightly builds. Does"
          " not affect the name of the module."
      ),
  )
  return parser.parse_args()


def _to_posix(path: str) -> str:
  """Converts an OS-specific path to a posix path.

  Args:
    path: An OS-specific path.

  Returns:
    A posix path with "/" separators.
  """
  return path.replace("\\", "/")


def _join_posix(root: str, posix_relpath: str) -> str:
  """Joins root with a posix relative path, returns an OS-specific path.

  Args:
    root: An OS-specific path to join with.
    posix_relpath: A relative path that uses posix style "/" separators.

  Returns:
    An OS-specific path by joining root and posix_relpath.
  """
  return os.path.join(root, *posix_relpath.split("/"))


def _strip_bazel_out_prefix(path_posix: str) -> str:
  bazel_out = "bazel-out/"
  idx = path_posix.find(bazel_out)
  if idx == -1:
    return path_posix
  bin_idx = path_posix.find("/bin/", idx + len(bazel_out))
  if bin_idx == -1:
    return path_posix
  return path_posix[bin_idx + len("/bin/") :]


def create_empty_init_files(dst_dir: str) -> None:
  """Create __init__.py files."""
  dir_list = [f for f in os.scandir(dst_dir) if f.is_dir()]
  for dir_name in dir_list:
    if not os.path.exists(os.path.join(dir_name, "__init__.py")):
      with open(os.path.join(dir_name, "__init__.py"), "w"):
        pass
    create_empty_init_files(dir_name.path)


def create_init_files(dst_dir: str, meta_dict: Optional[dict[str, str]] = None):
  create_empty_init_files(dst_dir)

  with open(os.path.join(dst_dir, "__init__.py"), "w") as f:
    if meta_dict:
      for key, value in meta_dict.items():
        f.write(f'{key} = "{value}"\n')


def construct_meta_dict(args) -> dict[str, str]:
  return {
      "__version__": args.version,
  }


def _dedupe_macos_rpaths(file_path: str) -> None:
  """Remove duplicate LC_RPATH entries from a Mach-O file on macOS."""
  try:
    output = subprocess.check_output(["otool", "-l", file_path], text=True)
  except (subprocess.CalledProcessError, FileNotFoundError):
    return

  rpaths = []
  saw_cmd = False
  for line in output.splitlines():
    stripped = line.strip()
    if stripped == "cmd LC_RPATH":
      saw_cmd = True
      continue
    if saw_cmd and stripped.startswith("path "):
      # Example: "path @loader_path (offset 12)"
      path = stripped.split(" ", 1)[1].split(" (offset", 1)[0].strip()
      rpaths.append(path)
      saw_cmd = False

  seen = set()
  for path in rpaths:
    if path in seen:
      subprocess.run(
          ["install_name_tool", "-delete_rpath", path, file_path],
          check=False,
      )
    else:
      seen.add(path)


# This is a temporary workaround to dedupe LC_RPATH entries in macos binaries.
# Otherwise we get the following error
# ImportError: dlopen(...):
#   Library not loaded: @rpath/libpywrap_litert_common.dylib
# ...
# duplicate LC_RPATH '@loader_path'
def _postprocess_macos_binaries(package_dir: str) -> None:
  if sys.platform != "darwin":
    return
  for root, _, files in os.walk(package_dir):
    for name in files:
      if name.endswith((".dylib", ".so")):
        _dedupe_macos_rpaths(os.path.join(root, name))


def prepare_build_tree(tree_path, args, project_name: str):
  """Prepares the build tree for the wheel build.

  Args:
    tree_path: Path to the build tree.
    args: Command line arguments.
    project_name: Name of the project.
  """
  src_dir = os.path.join(tree_path, project_name.replace("-", "_"))
  os.makedirs(src_dir)

  shutil.copyfile(args.setup_py, os.path.join(tree_path, "setup.py"))

  for src in args.src:
    shutil.copyfile(src, os.path.join(src_dir, os.path.basename(src)))

  for src in args.py_src or []:
    src_posix = _to_posix(src)
    if src_posix.startswith("litert/python/"):
      src_path = src_posix.removeprefix("litert/python/")
    elif src_posix.startswith("bazel-out/"):
      delimiter = None
      if "litert/python/" in src_posix:
        delimiter = "litert/python/"
      elif "ai_edge_litert/" in src_posix:
        delimiter = "ai_edge_litert/"
      else:
        raise ValueError(f"Unsupported source file: {src}")
      _, src_path = src_posix.split(delimiter, 1)
    else:
      raise ValueError(f"Unsupported source file: {src}")
    dest = _join_posix(src_dir, src_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(src, dest)

  structured_deps_pairs = []
  relpath_set = set()
  if args.structured_deps:
    for dep in args.structured_deps:
      dep_posix = _to_posix(dep)
      relative_path = _strip_bazel_out_prefix(dep_posix)
      structured_deps_pairs.append((dep, relative_path))
      relpath_set.add(relative_path.split("/")[0])

  # Create directory structure for structured deps
  dir_set = set()
  for _, relative_path in structured_deps_pairs:
    dir_set.add(os.path.dirname(relative_path))
  for dir_name in dir_set:
    os.makedirs(_join_posix(tree_path, dir_name), exist_ok=True)

  # Copy structured deps
  for original_path, relative_path in structured_deps_pairs:
    shutil.copyfile(original_path, _join_posix(tree_path, relative_path))

  # create empty init files for structured deps
  for relative_path in relpath_set:
    create_init_files(_join_posix(tree_path, relative_path))

  meta_dict = construct_meta_dict(args)

  create_init_files(src_dir, meta_dict)

  # Copy package data files to the build tree, after filling the __init__.
  if args.package_data is not None:
    for src in args.package_data:
      src_posix = _to_posix(src)

      def get_dest(file_path_posix: str):
        delimiter = "litert/"
        index = file_path_posix.find(delimiter)
        if index != -1:
          return file_path_posix[index + len(delimiter) :]
        else:
          return file_path_posix

      dest_rel = get_dest(src_posix)
      dest = _join_posix(src_dir, dest_rel)
      os.makedirs(os.path.dirname(dest), exist_ok=True)
      shutil.copyfile(src, dest)
  _postprocess_macos_binaries(src_dir)


def build_pyproject_wheel(
    buildtree_path: str, platform_name: Optional[str] = None
):
  """Builds a python wheel from a pyproject.toml file.

  Args:
    buildtree_path: Path to the build tree.
    platform_name: Platform name to be passed to build module.
  """
  env = os.environ.copy()

  command = [
      sys.executable,
      "-m",
      "build",
      "-w",
      "-o",
      os.getcwd(),
  ]

  if platform_name:
    command.append(
        # This is due to setuptools not making it possible to pass the
        # platform name as a dynamic pyproject.toml property.
        f"--config-setting=--build-option=--plat-name={platform_name}"
    )

  subprocess.run(
      command,
      check=True,
      cwd=buildtree_path,
      env=env,
  )


def build_setup_py_wheel(
    project_name: str,
    buildtree_path: str,
    output_dir: str,
    version: str,
    platform_name: Optional[str] = None,
    nightly_suffix: Optional[str] = None,
):
  """Builds a python wheel from a setup.py file.

  Args:
    project_name: Name of the project.
    buildtree_path: Path to the build tree.
    output_dir: Output directory for the wheel.
    version: Version of the wheel.
    platform_name: Platform name to be passed to build module.
    nightly_suffix: Suffix to be added to the name of the wheel for nightly
      builds. Does not affect the name of the module.
  """
  env = os.environ.copy()

  env["PROJECT_NAME"] = (
      project_name + nightly_suffix if nightly_suffix else project_name
  )
  env["PACKAGE_VERSION"] = version

  command = [
      sys.executable,
      f"{buildtree_path}/setup.py",
      "bdist_wheel",
      f"--plat-name={platform_name}",
  ]

  subprocess.run(
      command,
      check=True,
      cwd=buildtree_path,
      env=env,
  )

  for filename in glob.glob(os.path.join(buildtree_path, "dist/*.whl")):
    shutil.copy(filename, output_dir)


if __name__ == "__main__":
  build_dir = os.path.join(os.getcwd(), "wheel_build")
  arg_data = parse_args()

  prepare_build_tree(build_dir, arg_data, arg_data.project_name)
  build_setup_py_wheel(
      arg_data.project_name,
      build_dir,
      arg_data.output,
      arg_data.version,
      arg_data.platform,
      arg_data.nightly_suffix if arg_data.nightly_suffix else None,
  )
