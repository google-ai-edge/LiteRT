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

from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import zipfile

# BEGIN_LITERT_PREBUILTS_HELPER
_LITERT_PREBUILTS_GCS_BASE = "https://storage.googleapis.com/litert/binaries"
# Seconds to wait on each network operation before giving up.
_LITERT_PREBUILTS_TIMEOUT_S = 60
# HTTP statuses meaning the archive does not exist (no full-download retry).
_LITERT_PREBUILTS_MISSING_STATUSES = (404, 410)
# Upper bounds that protect against malformed archives (zip bombs, endless
# responses). The archive is ~350 MB; each vendor library is a few MB.
_LITERT_PREBUILTS_MAX_ARCHIVE_BYTES = 4 << 30
_LITERT_PREBUILTS_MAX_ENTRY_BYTES = 512 << 20
# zipfile read size over HTTPS Range requests; most libraries fit in one read.
_LITERT_PREBUILTS_RANGE_READ_BYTES = 1 << 20
# Pinned archive, written when the sdist is built:
# {"url": <immutable archive URL>, "sha256": {<archive entry>: <hex digest>}}.
# None in an unpinned (development) sdist.
_LITERT_PREBUILTS_PIN = None


class _RangeNotSupportedError(OSError):
  """The server did not answer a Range request with 206 Partial Content."""


class _HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
  """Refuses to follow a redirect to a non-HTTPS URL before contacting it."""

  def redirect_request(self, req, fp, code, msg, headers, newurl):
    if urllib.parse.urlsplit(newurl).scheme != "https":
      raise ValueError(f"Refusing redirect to non-HTTPS URL: {newurl}")
    return super().redirect_request(req, fp, code, msg, headers, newurl)


_HTTPS_OPENER = urllib.request.build_opener(_HttpsOnlyRedirectHandler)


def _open_https(url: str, headers: dict[str, str] | None = None):
  """Opens `url` with a timeout; rejects non-HTTPS URLs and redirects."""
  if urllib.parse.urlsplit(url).scheme != "https":
    raise ValueError(f"Refusing to download from non-HTTPS URL: {url}")
  request = urllib.request.Request(url, headers=headers or {})
  return _HTTPS_OPENER.open(request, timeout=_LITERT_PREBUILTS_TIMEOUT_S)


def _copy_bounded(src, dst, limit: int, digest=None) -> None:
  """Copies `src` to `dst` (None to discard) in chunks, up to `limit` bytes."""
  copied = 0
  for chunk in iter(lambda: src.read(1 << 20), b""):
    copied += len(chunk)
    if copied > limit:
      raise ValueError(f"Refusing to write more than {limit} bytes")
    if digest is not None:
      digest.update(chunk)
    if dst is not None:
      dst.write(chunk)


class _HttpRangeFile(io.RawIOBase):
  """Seekable HTTPS file using Range requests so zipfile reads only needed entries."""

  def __init__(self, url: str):
    self._url = url
    self._pos = 0
    with _open_https(url, {"Range": "bytes=0-0"}) as resp:
      if resp.status != 206:
        raise _RangeNotSupportedError(f"{url}: HTTP {resp.status}, not 206")
      total = resp.headers.get("Content-Range", "").rpartition("/")[-1]
    if not total.isdigit() or int(total) == 0:
      raise _RangeNotSupportedError(f"{url}: bad Content-Range total {total!r}")
    self._size = int(total)
    tail_len = min(self._size, 65536)
    self._tail_offset = self._size - tail_len
    self._tail = self._fetch(self._tail_offset, self._size)

  def _fetch(self, start: int, end: int) -> bytes:
    """Returns bytes [start, end); never buffers more than end - start bytes."""
    with _open_https(self._url, {"Range": f"bytes={start}-{end - 1}"}) as resp:
      if resp.status != 206:
        raise _RangeNotSupportedError(
            f"{self._url}: HTTP {resp.status}, not 206"
        )
      data = resp.read(end - start)
    if len(data) != end - start:
      raise OSError(f"{self._url}: short read at bytes {start}-{end - 1}")
    return data

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


def _pinned_archive_url(package_version: str) -> str:
  """Returns the immutable prebuilt archive URL for an SDK version.

  Nightly `<version>.devYYYYMMDD` -> `nightly/archive/litert_prebuilts_YYYYMMDD
  .zip`; release `<version>` -> `<version>/litert_prebuilts.zip`. The mutable
  `nightly/litert_prebuilts.zip` is never used.

  Args:
    package_version: The SDK package version.
  """
  nightly_match = re.search(r"\.dev(\d{8})$", package_version)
  if nightly_match:
    date = nightly_match.group(1)
    return (
        f"{_LITERT_PREBUILTS_GCS_BASE}/nightly/archive/"
        f"litert_prebuilts_{date}.zip"
    )
  return f"{_LITERT_PREBUILTS_GCS_BASE}/{package_version}/litert_prebuilts.zip"


def _open_prebuilts_zip(url: str, stack: contextlib.ExitStack):
  """Opens the archive at `url` (HTTPS or local path) as a ZipFile.

  Reads only what zipfile needs via Range requests; downloads the whole
  archive to a temporary file only if the server does not support Range.

  Args:
    url: HTTPS URL or local zip path.
    stack: Owns the temporary directory of a full download.

  Returns:
    The opened ZipFile, registered with `stack`.
  """
  if os.path.isfile(url):
    return stack.enter_context(zipfile.ZipFile(url, "r"))
  try:
    raw = _HttpRangeFile(url)
    return stack.enter_context(
        zipfile.ZipFile(
            io.BufferedReader(
                raw, buffer_size=_LITERT_PREBUILTS_RANGE_READ_BYTES
            ),
            "r",
        )
    )
  except urllib.error.HTTPError as e:
    if e.code in _LITERT_PREBUILTS_MISSING_STATUSES:
      raise
  except _RangeNotSupportedError:
    pass
  tmpdir = stack.enter_context(tempfile.TemporaryDirectory())
  archive_path = os.path.join(tmpdir, "litert_prebuilts.zip")
  with _open_https(url) as resp, open(archive_path, "wb") as f:
    _copy_bounded(resp, f, _LITERT_PREBUILTS_MAX_ARCHIVE_BYTES)
  return stack.enter_context(zipfile.ZipFile(archive_path, "r"))


def _extract_verified(
    zf: zipfile.ZipFile, entry: str, dest_path: str, sha256: str | None
) -> None:
  """Extracts `entry` to `dest_path`, verifying its SHA-256 if given.

  The file is written to a temporary name and moved into place only after the
  digest matches; on any error the temporary file is removed, so no partial or
  unverified library is left behind (or packaged).

  Args:
    zf: The archive.
    entry: Archive entry name.
    dest_path: Destination path.
    sha256: Expected hex SHA-256, or None to skip verification.

  Raises:
    ValueError: On a SHA-256 mismatch or an oversized entry.
  """
  os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
  tmp_path = dest_path + ".tmp"
  try:
    digest = hashlib.sha256()
    with zf.open(entry) as src, open(tmp_path, "wb") as dst:
      _copy_bounded(src, dst, _LITERT_PREBUILTS_MAX_ENTRY_BYTES, digest)
    if sha256 is not None and digest.hexdigest() != sha256:
      raise ValueError(
          f"SHA-256 mismatch for {entry}: got {digest.hexdigest()}, expected"
          f" {sha256}"
      )
    os.chmod(tmp_path, 0o755)
    os.replace(tmp_path, dest_path)
  except BaseException:
    if os.path.exists(tmp_path):
      os.remove(tmp_path)
    raise
  print(f"Extracted {entry} -> {dest_path}")


def _download_litert_prebuilts(
    package_version: str,
    members: dict[str, str],
) -> bool:
  """Downloads, verifies, and extracts LiteRT vendor prebuilts for this host.

  Source, in priority order:
    1. `LITERT_PREBUILTS_URL` (HTTPS URL or local zip), if set. The user chose
       the archive, so digests are not checked (a warning is printed).
    2. The archive pinned when the sdist was built (`_LITERT_PREBUILTS_PIN`):
       one immutable URL plus the SHA-256 of every entry. Each extracted file
       must match its digest. There is no fallback to another archive.
    3. Unpinned sdist (local development build) and no override: fails.

  Only the needed entries are read, via HTTPS Range requests (whole archive
  only if Range is unsupported); remote URLs and redirects must be HTTPS.

  Failure model (archive missing, network error, entry missing, digest
  mismatch, or unpinned sdist without an override):
    * Default: raises SystemExit, so `pip install` fails instead of installing
      an SDK without (or with unverified) compiler plugin / dispatch library.
    * `LITERT_PREBUILTS_OPTIONAL=1`: logs an error and returns False; the SDK
      installs without the LiteRT vendor libraries (e.g. offline CI).
  `SKIP_SDK_DOWNLOAD` does not affect this function; it only skips the vendor
  SDK download in setup.py.

  Args:
    package_version: The SDK package version baked into setup.py.
    members: Map of archive entry (e.g. `linux_x86_64/libX.so`) to destination
      path. Empty on hosts the vendor does not support; nothing is fetched.

  Returns:
    True if the libraries were extracted, False if `members` is empty or the
    download failed with `LITERT_PREBUILTS_OPTIONAL=1`.

  Raises:
    SystemExit: On failure when `LITERT_PREBUILTS_OPTIONAL` is unset.
  """
  if not members:
    return False
  override = os.environ.get("LITERT_PREBUILTS_URL", "").strip()
  pin = _LITERT_PREBUILTS_PIN
  try:
    if override:
      print(
          f"WARNING: LITERT_PREBUILTS_URL={override} is set; SHA-256"
          " verification of the LiteRT prebuilts is skipped.",
          file=sys.stderr,
      )
      url, expected = override, {}
    elif pin:
      url, expected = pin["url"], pin["sha256"]
      unpinned = sorted(entry for entry in members if not expected.get(entry))
      if unpinned:
        raise ValueError(f"No pinned SHA-256 for {unpinned}")
    else:
      raise ValueError(
          f"This {package_version} sdist has no pinned LiteRT prebuilts (it"
          " was built without network access to the prebuilt archive). Set"
          " LITERT_PREBUILTS_URL to a litert_prebuilts zip."
      )
    print(f"Fetching LiteRT prebuilts from {url}...")
    with contextlib.ExitStack() as stack:
      zf = _open_prebuilts_zip(url, stack)
      missing = sorted(set(members) - set(zf.namelist()))
      if missing:
        raise ValueError(f"{url} does not contain {missing}")
      for entry, dest_path in members.items():
        _extract_verified(zf, entry, dest_path, expected.get(entry))
    return True
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(
        f"ERROR: Failed to install LiteRT prebuilts {sorted(members)}: {e}",
        file=sys.stderr,
    )
  if os.environ.get("LITERT_PREBUILTS_OPTIONAL", "").strip().lower() in (
      "1",
      "true",
      "yes",
  ):
    return False
  raise SystemExit(
      "Failed to install LiteRT vendor prebuilts. Set LITERT_PREBUILTS_URL to"
      " a litert_prebuilts zip, or LITERT_PREBUILTS_OPTIONAL=1 to install"
      " without them."
  )


# END_LITERT_PREBUILTS_HELPER

_PREBUILTS_HELPER_MARKER = "{{ LITERT_PREBUILTS_HELPER }}"
_PREBUILTS_HELPER_STUB = (
    f"def {_download_litert_prebuilts.__name__}(*unused_args):"
    f"  # {_PREBUILTS_HELPER_MARKER}\n"
    "  raise NotImplementedError\n"
)
_PREBUILTS_PIN_LINE = "_LITERT_PREBUILTS_PIN = None\n"
# Modules the inlined helper uses that SDK setup.py files may not import.
_PREBUILTS_HELPER_IMPORTS = (
    "import contextlib\n"
    "import hashlib\n"
    "import io\n"
    "import re\n"
    "import shutil\n"
    "import urllib.error\n"
    "import urllib.parse\n"
    "\n"
)
# Archive entries that are LiteRT vendor compiler plugins / dispatch libraries
# for the host platforms the SDK sdists support.
_PREBUILT_ENTRY_RE = re.compile(
    r"^(linux_x86_64|linux_arm64|windows_x86_64|windows_arm64)/"
    r"(lib)?LiteRt(CompilerPlugin|Dispatch)_\w+\.(so|dll)$"
)


def _validate_pin(pin: dict[str, Any]) -> None:
  """Raises ValueError unless `pin` is `{"url": https, "sha256": {e: hex64}}`."""
  if (
      not isinstance(pin, dict)
      or urllib.parse.urlsplit(str(pin.get("url", ""))).scheme != "https"
      or not isinstance(pin.get("sha256"), dict)
      or not pin["sha256"]
      or not all(
          _PREBUILT_ENTRY_RE.match(entry)
          and re.fullmatch(r"[0-9a-f]{64}", str(digest))
          for entry, digest in pin["sha256"].items()
      )
  ):
    raise ValueError(f"Invalid LiteRT prebuilts pin: {pin!r}")


def _load_prebuilts_helper_source(pin: dict[str, Any] | None = None) -> str:
  """Returns the helper block above with its imports and `pin` baked in."""
  with open(__file__, "rt", encoding="utf-8") as f:
    source = f.read()
  start_marker = "# BEGIN_LITERT_PREBUILTS_HELPER\n"
  end_marker = "# END_LITERT_PREBUILTS_HELPER\n"
  start = source.index(start_marker) + len(start_marker)
  end = source.index(end_marker, start)
  helper = source[start:end]
  if pin is not None:
    helper = helper.replace(
        _PREBUILTS_PIN_LINE, f"_LITERT_PREBUILTS_PIN = {pin!r}\n", 1
    )
  return _PREBUILTS_HELPER_IMPORTS + helper


def _inline_prebuilts_helper(
    setup_py_content: str,
    setup_py_name: str,
    pin: dict[str, Any] | None = None,
) -> str:
  """Replaces the helper stub in an SDK setup.py with the real helper.

  Args:
    setup_py_content: Contents of the SDK setup.py.
    setup_py_name: Name of the setup.py, for error messages.
    pin: Pinned archive (`{"url", "sha256"}`), or None for an unpinned sdist.
      Only digests of the archive entries that the setup.py references are baked
      in, and every referenced entry must be pinned.

  Returns:
    `setup_py_content` with the stub replaced, or unchanged if it has no
    `LITERT_PREBUILTS_HELPER` marker.

  Raises:
    ValueError: If the marker is present but the stub does not match exactly
      (which would otherwise ship `raise NotImplementedError` in the sdist),
      or if `pin` is invalid or lacks an entry the setup.py references (which
      would otherwise publish an uninstallable sdist).
  """
  if _PREBUILTS_HELPER_STUB in setup_py_content:
    if pin is not None:
      _validate_pin(pin)
      referenced = sorted({
          match.group(1)
          for match in re.finditer(
              rf"'({_PREBUILT_ENTRY_RE.pattern[1:-1]})'", setup_py_content
          )
      })
      missing = [entry for entry in referenced if entry not in pin["sha256"]]
      if not referenced or missing:
        raise ValueError(
            f"{setup_py_name}: pin {pin['url']} lacks {missing or 'entries'}"
            f" referenced by setup.py ({referenced})"
        )
      pin = {
          "url": pin["url"],
          "sha256": {entry: pin["sha256"][entry] for entry in referenced},
      }
    return setup_py_content.replace(
        _PREBUILTS_HELPER_STUB, _load_prebuilts_helper_source(pin)
    )
  if _PREBUILTS_HELPER_MARKER in setup_py_content:
    raise ValueError(
        f"{setup_py_name} has a {_PREBUILTS_HELPER_MARKER} marker that does"
        f" not match the expected stub:\n{_PREBUILTS_HELPER_STUB}"
    )
  return setup_py_content


def _read_litert_version(version_bzl: str) -> str:
  """Returns LITERT_EXPERIMENTAL_VERSION parsed from the `version_bzl` file."""
  with open(version_bzl, "rt", encoding="utf-8") as f:
    match = re.search(
        r'^LITERT_EXPERIMENTAL_VERSION = "([^"]+)"', f.read(), re.MULTILINE
    )
  if not match:
    raise ValueError(f"LITERT_EXPERIMENTAL_VERSION not found in {version_bzl}")
  return match.group(1)


def pin_litert_prebuilts(
    url: str, wait_seconds: int = 0, poll_seconds: int = 60
) -> dict[str, Any]:
  """Returns `{"url", "sha256"}` for the vendor libraries in the archive.

  Reads only the vendor library entries (via Range requests) and hashes them.

  Args:
    url: Immutable HTTPS URL of the prebuilt archive.
    wait_seconds: How long to keep polling while the archive does not exist yet
      (HTTP 404/410), e.g. when the nightly prebuilt job is late.
    poll_seconds: Delay between polls.

  Raises:
    urllib.error.HTTPError: If the archive still does not exist.
    ValueError: If the archive has no vendor libraries.
  """
  deadline = time.monotonic() + wait_seconds
  while True:
    try:
      with contextlib.ExitStack() as stack:
        zf = _open_prebuilts_zip(url, stack)
        sha256 = {}
        for entry in sorted(zf.namelist()):
          if not _PREBUILT_ENTRY_RE.match(entry):
            continue
          digest = hashlib.sha256()
          with zf.open(entry) as src:
            _copy_bounded(src, None, _LITERT_PREBUILTS_MAX_ENTRY_BYTES, digest)
          sha256[entry] = digest.hexdigest()
      break
    except urllib.error.HTTPError as e:
      if (
          e.code not in _LITERT_PREBUILTS_MISSING_STATUSES
          or time.monotonic() + poll_seconds > deadline
      ):
        raise
      print(f"{url} not available yet (HTTP {e.code}); retrying...")
      time.sleep(poll_seconds)
  if not sha256:
    raise ValueError(f"{url} contains no LiteRT vendor libraries")
  return {"url": url, "sha256": sha256}


def _pin_main(argv: list[str]) -> None:
  """`sdist_wrapper.py pin`: writes the pin file consumed by sdist_rule."""
  parser = argparse.ArgumentParser(
      prog="sdist_wrapper.py pin",
      description=(
          "Pins the LiteRT prebuilt archive (URL + per-file SHA-256) for the"
          " ai-edge-litert-sdk-* sdists."
      ),
  )
  parser.add_argument("--version_bzl", required=True)
  parser.add_argument(
      "--nightly_date", default="", help="YYYYMMDD for nightly builds."
  )
  parser.add_argument("--url", default="", help="Overrides the archive URL.")
  parser.add_argument("--wait_seconds", type=int, default=0)
  parser.add_argument("--out", required=True)
  args = parser.parse_args(argv)
  version = _read_litert_version(args.version_bzl)
  if args.nightly_date:
    version += f".dev{args.nightly_date}"
  url = args.url or _pinned_archive_url(version)
  pin = pin_litert_prebuilts(url, args.wait_seconds)
  _validate_pin(pin)
  with open(args.out, "wt", encoding="utf-8") as f:
    # Compact (no whitespace): passed as a single Bazel flag value.
    json.dump(pin, f, sort_keys=True, separators=(",", ":"))
  print(f"Pinned {len(pin['sha256'])} LiteRT prebuilts from {url}")


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
  if sys.argv[1:2] == ["pin"]:
    _pin_main(sys.argv[2:])
    return
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

  parser.add_argument(
      "--prebuilts_pin_json",
      default="",
      help=(
          "JSON from `sdist_wrapper.py pin` with the LiteRT prebuilt archive"
          " URL and per-file SHA-256 to bake into setup.py. Without it, the"
          " sdist is unpinned and needs LITERT_PREBUILTS_URL at install time."
      ),
  )

  args = parser.parse_args()

  pin = json.loads(args.prebuilts_pin_json) if args.prebuilts_pin_json else None

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
    setup_py_content = _inline_prebuilts_helper(
        setup_py_content, args.setup_py, pin
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
