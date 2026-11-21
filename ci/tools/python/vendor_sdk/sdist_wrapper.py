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
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile


# BEGIN_LITERT_PREBUILTS_HELPER
_LITERT_PREBUILTS_GCS_BASE = "https://storage.googleapis.com/litert/binaries"


class _HttpRangeFile(io.RawIOBase):
  """Seekable HTTP file using Range requests so zipfile reads only needed entries."""

  def __init__(self, url: str):
    self._url = url
    self._pos = 0
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req) as resp:
      if resp.status != 206:
        raise OSError(
            f"HTTP Range not supported for {url} (status {resp.status})"
        )
      content_range = resp.headers.get("Content-Range", "")
      self._size = int(content_range.rpartition("/")[-1])
    tail_len = min(self._size, 65536)
    self._tail_offset = self._size - tail_len
    self._tail = self._fetch(self._tail_offset, self._size)

  def _fetch(self, start: int, end: int) -> bytes:
    req = urllib.request.Request(
        self._url, headers={"Range": f"bytes={start}-{end - 1}"}
    )
    with urllib.request.urlopen(req) as resp:
      return resp.read()

  def seekable(self) -> bool:
    return True

  def readable(self) -> bool:
    return True

  def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
    if whence == io.SEEK_SET:
      self._pos = offset
    elif whence == io.SEEK_CUR:
      self._pos += offset
    elif whence == io.SEEK_END:
      self._pos = self._size + offset
    return self._pos

  def tell(self) -> int:
    return self._pos

  def readinto(self, b) -> int:
    if self._pos >= self._size or not b:
      return 0
    end = min(self._pos + len(b), self._size)
    if self._pos >= self._tail_offset:
      start = self._pos - self._tail_offset
      data = self._tail[start : start + (end - self._pos)]
    else:
      data = self._fetch(self._pos, end)
    n = len(data)
    b[:n] = data
    self._pos += n
    return n


def _candidate_prebuilt_urls(package_version: str) -> list[str]:
  """Returns candidate GCS URLs (or local override path) in priority order."""
  override = os.environ.get("LITERT_PREBUILTS_URL", "").strip()
  if override:
    return [override]
  base = _LITERT_PREBUILTS_GCS_BASE
  nightly_match = re.search(r"\.dev(\d{8})$", package_version)
  if nightly_match:
    date = nightly_match.group(1)
    return [
        f"{base}/nightly/archive/litert_prebuilts_{date}.zip",
        f"{base}/nightly/litert_prebuilts.zip",
    ]
  return [f"{base}/{package_version}/litert_prebuilts.zip"]


def _extract_members_from_zip(
    zf: zipfile.ZipFile, members: dict[str, str]
) -> bool:
  """Extracts `members` ({zip_entry: dest_path}) if all exist in `zf`."""
  names = set(zf.namelist())
  if not all(entry in names for entry in members):
    return False
  for entry, dest_path in members.items():
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with zf.open(entry) as src, open(dest_path, "wb") as dst:
      dst.write(src.read())
    os.chmod(dest_path, 0o755)
    print(f"Extracted {entry} -> {dest_path}")
  return True


def _download_litert_prebuilts(
    package_version: str,
    members: dict[str, str],
) -> bool:
  """Downloads and extracts LiteRT vendor prebuilts for the current host."""
  if not members:
    return False
  for url in _candidate_prebuilt_urls(package_version):
    print(f"Fetching LiteRT prebuilts from {url}...")
    try:
      if os.path.isfile(url):
        with zipfile.ZipFile(url, "r") as zf:
          if _extract_members_from_zip(zf, members):
            return True
        continue
      try:
        raw = _HttpRangeFile(url)
        with zipfile.ZipFile(
            io.BufferedReader(raw, buffer_size=262144), "r"
        ) as zf:
          if _extract_members_from_zip(zf, members):
            return True
          continue
      except urllib.error.HTTPError:
        raise
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, "litert_prebuilts.zip")
        urllib.request.urlretrieve(url, archive_path)
        with zipfile.ZipFile(archive_path, "r") as zf:
          if _extract_members_from_zip(zf, members):
            return True
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f"WARNING: Could not fetch prebuilts from {url}: {e}", file=sys.stderr
      )
  print(
      f"ERROR: Failed to download LiteRT prebuilts: {sorted(members)}",
      file=sys.stderr,
  )
  if os.environ.get("LITERT_PREBUILTS_OPTIONAL", "").strip().lower() in (
      "1",
      "true",
      "yes",
  ):
    return False
  raise SystemExit(
      "Failed to download LiteRT vendor prebuilts. Set LITERT_PREBUILTS_URL to"
      " a local litert_prebuilts zip, or LITERT_PREBUILTS_OPTIONAL=1 to"
      " install without them."
  )
# END_LITERT_PREBUILTS_HELPER

_PREBUILTS_HELPER_STUB = (
    f"def {_download_litert_prebuilts.__name__}(*unused_args):"
    "  # {{ LITERT_PREBUILTS_HELPER }}\n"
    "  raise NotImplementedError\n"
)


def _load_prebuilts_helper_source() -> str:
  with open(__file__, "rt", encoding="utf-8") as f:
    source = f.read()
  start_marker = "# BEGIN_LITERT_PREBUILTS_HELPER\n"
  end_marker = "# END_LITERT_PREBUILTS_HELPER\n"
  start = source.index(start_marker) + len(start_marker)
  end = source.index(end_marker, start)
  imports = "import io\nimport re\nimport urllib.error\n\n"
  return imports + source[start:end]


def _get_python_subprocess_env():
  """Returns the current Python runtime and import path for child processes."""
  env = os.environ.copy()
  pythonpath_entries = []

  for entry in sys.path:
    if entry and entry not in pythonpath_entries:
      pythonpath_entries.append(entry)

  existing_pythonpath = env.get("PYTHONPATH")
  if existing_pythonpath:
    for entry in existing_pythonpath.split(os.pathsep):
      if entry and entry not in pythonpath_entries:
        pythonpath_entries.append(entry)

  if pythonpath_entries:
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

  return sys.executable, env


def main():
  parser = argparse.ArgumentParser(
      description="Wrapper to create sdist using setup.py."
  )
  parser.add_argument(
      "--project_name",
      required=True,
      help="Name of the project",
  )
  parser.add_argument(
      "--dir",
      required=True,
      help="Directory containing the project files.",
  )
  parser.add_argument(
      "--setup_py",
      default="setup.template.py",
      help="Name of the setup script (default: setup.py).",
  )
  parser.add_argument(
      "--output_sdist_path",
      required=True,
      help="Full path where the final sdist .tar.gz should be placed.",
  )
  parser.add_argument(
      "--nightly_suffix",
      help=(
          "Suffix to be added to the name of the sdist for nightly builds. Does"
          " not affect the name of the module."
      ),
  )
  parser.add_argument("--version", help="version of the sdist")
  parser.add_argument(
      "--substitution",
      action="append",
      default=[],
      metavar="KEY=VALUE",
      help=(
          "Extra {{ KEY }} -> VALUE substitution to apply to setup.py's"
          " contents, in addition to the built-in {{ PACKAGE_NAME }} /"
          " {{ PACKAGE_VERSION }} substitutions. May be repeated."
      ),
  )

  args = parser.parse_args()

  substitutions = {}
  for substitution in args.substitution:
    key, sep, value = substitution.partition("=")
    if not sep:
      raise ValueError(
          f"--substitution must be of the form KEY=VALUE, got: {substitution!r}"
      )
    substitutions[key] = value

  original_cwd = os.getcwd()
  build_dir = os.path.join(os.getcwd(), "sdist_build")
  os.makedirs(build_dir)
  sdist_temp_output_dir = "dist_sdist_temp"
  project_name = args.project_name
  version = args.version

  try:
    print(f"Changing working directory to: {args.dir}")
    os.chdir(args.dir)

    with open(args.setup_py, "rt") as f:
      setup_py_content = f.read()
    setup_py_content = setup_py_content.replace(
        "{{ PACKAGE_NAME }}",
        project_name + "_nightly" if args.nightly_suffix else project_name,
    )
    setup_py_content = setup_py_content.replace(
        "{{ PACKAGE_VERSION }}",
        version,
    )
    for key, value in substitutions.items():
      setup_py_content = setup_py_content.replace("{{ " + key + " }}", value)
    if _PREBUILTS_HELPER_STUB in setup_py_content:
      setup_py_content = setup_py_content.replace(
          _PREBUILTS_HELPER_STUB, _load_prebuilts_helper_source()
      )
    tmp_setup_py_path = os.path.join(build_dir, "setup.py")
    with open(tmp_setup_py_path, "wt") as f:
      f.write(setup_py_content)

    shutil.copy("MANIFEST.in", build_dir)

    os.makedirs(os.path.join(build_dir, project_name))
    with open(os.path.join(project_name, "__init__.py"), "rt") as f:
      init_py_content = f.read()
    init_py_content = init_py_content.replace(
        "{{ PACKAGE_VERSION }}",
        version,
    )
    dest_init_py_path = os.path.join(build_dir, project_name, "__init__.py")
    with open(dest_init_py_path, "wt") as f:
      f.write(init_py_content)

    if os.path.exists("TERMS_OF_USE.md"):
      shutil.copy(
          "TERMS_OF_USE.md",
          os.path.join(build_dir, project_name, "TERMS_OF_USE.md"),
      )

    print(f"Changing working directory to: {build_dir}")
    os.chdir(build_dir)

    if os.path.exists(sdist_temp_output_dir):
      shutil.rmtree(sdist_temp_output_dir)
    os.makedirs(sdist_temp_output_dir)

    py_executable, env = _get_python_subprocess_env()
    cmd = [
        py_executable,
        tmp_setup_py_path,
        "sdist",
        "--dist-dir",
        sdist_temp_output_dir,
    ]

    print(f"Running command: {' '.join(cmd)}")
    process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

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
    try:
      shutil.rmtree(build_dir)
    except OSError as e:
      print(
          f"Warning: Could not clean up {build_dir}: {e}",
          file=sys.stderr,
      )

  finally:
    os.chdir(original_cwd)


if __name__ == "__main__":
  main()
