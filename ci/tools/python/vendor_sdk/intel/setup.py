# Copyright 2026 The AI Edge LiteRT Authors.
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
"""Intel OpenVINO SDK for AI Edge LiteRT.

This setup.py fetches `libopenvino_intel_npu_compiler.{so,dll}` from the
OpenVINO toolkit archives at install time. The public `openvino` pip wheel
ships the NPU plugin but not this compiler library, so AOT compile for
Intel NPU fails without it. Level Zero loader, NPU firmware and UMD
(intel-level-zero-npu, intel-fw-npu, intel-driver-compiler-npu) remain a
user-installed prerequisite.
"""

import os
import platform
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

import setuptools
from setuptools.command.build_py import build_py as _build_py  # pylint: disable=g-importing-member

PACKAGE_NAME = '{{ PACKAGE_NAME }}'
PACKAGE_VERSION = '{{ PACKAGE_VERSION }}'

SKIP_SDK_DOWNLOAD = os.environ.get('SKIP_SDK_DOWNLOAD', '').strip().lower() in (
    '1',
    'true',
    'yes',
)

IS_LINUX = sys.platform == 'linux'
IS_WINDOWS = sys.platform == 'win32'
IS_X86_ARCHITECTURE = platform.machine() in ('x86_64', 'AMD64', 'i386', 'i686')

# --- Configuration for OpenVINO NPU Compiler Download ---
#
# Must match the OpenVINO build pinned in third_party/intel_openvino/
# openvino.bzl. That workspace file pins the OpenVINO SDK used to compile the
# Intel OV compiler plugin and dispatch library at build time; this file pins
# the OpenVINO nightly archive that provides the NPU compiler shared library
# fetched at pip install time, and the `openvino` PyPI wheel referenced by
# install_requires below. All three must point at the same commit. OpenVINO's
# build number (the numeric segment in `2026.2.0-21820-<commit>/`) uniquely
# identifies a build: it appears both in the toolkit archive directory and as
# the wheel filename's build tag (the `-21820-` between version and py-tag).
# ci/check_openvino_version_sync.py enforces this across openvino.bzl and
# this file.
# LINT.IfChange(wheel_openvino_sdk_version)
_OV_BUILD_NUMBER = '21820'
_OV_COMMIT = '9a25caa5a15'
_OV_PEP440_VERSION = '2026.2.0.dev20260506'
_OV_ARCHIVE_DIR = f'2026.2.0-{_OV_BUILD_NUMBER}-{_OV_COMMIT}'
_OV_BASE_URL = (
    'https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly'
    f'/{_OV_ARCHIVE_DIR}'
)
# PyPI nightly wheel index used for install_requires direct URLs; lets a plain
# `pip install ai-edge-litert-sdk-intel-nightly` pull the matching openvino
# wheel without --extra-index-url.
_OV_WHEEL_BASE_URL = (
    'https://storage.openvinotoolkit.org/wheels/nightly/openvino'
)
# LINT.ThenChange(
#   ../../../../../third_party/intel_openvino/openvino.bzl:openvino_packages,
# )

# Archive -> list of members to extract. OpenVINO 2026.2 split the NPU VCL
# adapter out of libopenvino_intel_npu_compiler.{so,dll}: the `_compiler_loader`
# sibling carries the VCL entry points (vclGetVersion etc.) that the NPU plugin
# dlopens, and `_compiler` now holds the actual compiler backend. Both files
# must land in openvino/libs/ for AOT to succeed.
_LINUX_MEMBERS = (
    {
        'member_suffix': (
            'runtime/lib/intel64/libopenvino_intel_npu_compiler.so'
        ),
        'output_name': 'libopenvino_intel_npu_compiler.so',
    },
    {
        'member_suffix': (
            'runtime/lib/intel64/libopenvino_intel_npu_compiler_loader.so'
        ),
        'output_name': 'libopenvino_intel_npu_compiler_loader.so',
    },
)
_WINDOWS_MEMBERS = (
    {
        'member_suffix': (
            'runtime/bin/intel64/Release/openvino_intel_npu_compiler.dll'
        ),
        'output_name': 'openvino_intel_npu_compiler.dll',
    },
    {
        'member_suffix': (
            'runtime/bin/intel64/Release/openvino_intel_npu_compiler_loader.dll'
        ),
        'output_name': 'openvino_intel_npu_compiler_loader.dll',
    },
)

_ARCHIVES = {
    'ubuntu22': {
        'url': (
            f'{_OV_BASE_URL}/openvino_toolkit_ubuntu22_{_OV_PEP440_VERSION}_x86_64.tgz'
        ),
        'members': _LINUX_MEMBERS,
    },
    'ubuntu24': {
        'url': (
            f'{_OV_BASE_URL}/openvino_toolkit_ubuntu24_{_OV_PEP440_VERSION}_x86_64.tgz'
        ),
        'members': _LINUX_MEMBERS,
    },
    'windows': {
        'url': (
            f'{_OV_BASE_URL}/openvino_toolkit_windows_{_OV_PEP440_VERSION}_x86_64.zip'
        ),
        'members': _WINDOWS_MEMBERS,
    },
}

# Pin the `openvino` PyPI wheel to the exact nightly build that matches
# _OV_ARCHIVE_DIR. One direct URL per supported (python, os); pip picks the
# right one via the PEP 508 environment marker.
# Wheel filename schema at _OV_WHEEL_BASE_URL:
#   openvino-<pep440>-<build>-<py>-<abi>-<plat>.whl
# Linux wheels use manylinux2014_x86_64, Windows uses win_amd64.
_OV_WHEEL_TARGETS = (
    # (py_tag, abi_tag, platform_tag, env_marker)
    (
        'cp310',
        'cp310',
        'manylinux_2_28_x86_64',
        (
            'python_version == "3.10" and sys_platform == "linux" and'
            ' platform_machine == "x86_64"'
        ),
    ),
    (
        'cp311',
        'cp311',
        'manylinux_2_28_x86_64',
        (
            'python_version == "3.11" and sys_platform == "linux" and'
            ' platform_machine == "x86_64"'
        ),
    ),
    (
        'cp312',
        'cp312',
        'manylinux_2_28_x86_64',
        (
            'python_version == "3.12" and sys_platform == "linux" and'
            ' platform_machine == "x86_64"'
        ),
    ),
    (
        'cp313',
        'cp313',
        'manylinux_2_28_x86_64',
        (
            'python_version == "3.13" and sys_platform == "linux" and'
            ' platform_machine == "x86_64"'
        ),
    ),
    (
        'cp311',
        'cp311',
        'win_amd64',
        'python_version == "3.11" and sys_platform == "win32"',
    ),
)


def _openvino_install_requires():
  reqs = []
  for py_tag, abi_tag, plat_tag, marker in _OV_WHEEL_TARGETS:
    wheel = (
        f'openvino-{_OV_PEP440_VERSION}-{_OV_BUILD_NUMBER}'
        f'-{py_tag}-{abi_tag}-{plat_tag}.whl'
    )
    reqs.append(f'openvino @ {_OV_WHEEL_BASE_URL}/{wheel} ; {marker}')
  return reqs


_TARGET_DIR = 'ai_edge_litert_sdk_intel/data'


def _parse_os_release() -> tuple[str, str]:
  """Returns (ID, VERSION_ID) from /etc/os-release, or ('', '')."""
  path = '/etc/os-release'
  if not os.path.isfile(path):
    return '', ''
  data = {}
  try:
    with open(path, 'rt', encoding='utf-8') as f:
      for line in f:
        if '=' not in line:
          continue
        key, _, value = line.strip().partition('=')
        data[key] = value.strip().strip('"').strip("'")
  except OSError:
    return '', ''
  return data.get('ID', ''), data.get('VERSION_ID', '')


def _select_archive_key() -> str | None:
  """Returns the _ARCHIVES key to use, or None to skip."""
  override = os.environ.get('LITERT_OV_OS_ID', '').strip().lower()
  if override:
    if override in _ARCHIVES:
      print(f'Using LITERT_OV_OS_ID override: {override}')
      return override
    print(
        f'WARNING: LITERT_OV_OS_ID={override!r} is not a known archive key;'
        f' ignoring. Known keys: {sorted(_ARCHIVES)}',
        file=sys.stderr,
    )

  if IS_WINDOWS and IS_X86_ARCHITECTURE:
    return 'windows'

  if not (IS_LINUX and IS_X86_ARCHITECTURE):
    print(
        'IGNORED: Intel NPU AOT is only supported on Linux x86_64 (Ubuntu'
        ' 22/24) and Windows x86_64.'
    )
    return None

  os_id, version_id = _parse_os_release()
  if os_id == 'ubuntu':
    if version_id.startswith('22.'):
      return 'ubuntu22'
    if version_id.startswith(('24.', '25.')):
      return 'ubuntu24'
    if version_id.startswith('20.'):
      print(
          'WARNING: Ubuntu 20.04 detected. Falling back to the ubuntu22'
          ' archive; the NPU compiler may fail to load if the host glibc is'
          ' too old.',
          file=sys.stderr,
      )
      return 'ubuntu22'
  if os_id == 'debian' and version_id.startswith('12'):
    print('Debian 12 detected; using ubuntu22 archive.')
    return 'ubuntu22'

  print(
      f'IGNORED: Unsupported Linux distribution (ID={os_id!r},'
      f' VERSION_ID={version_id!r}). Only Ubuntu 22/24 x86_64 is supported.'
      ' Set LITERT_OV_OS_ID=ubuntu22 or ubuntu24 to force a selection.'
  )
  return None


def _member_target_path(
    normalized_path: str,
    member_suffix: str,
    output_name: str,
    target_dir: str,
) -> str | None:
  """Returns the absolute output path for the one member we want, or None."""
  if not normalized_path.replace(os.sep, '/').endswith(member_suffix):
    return None
  candidate = os.path.normpath(os.path.join(target_dir, output_name))
  if os.path.isabs(output_name) or output_name.startswith('..'):
    return None
  return candidate


def _extract_tar(
    archive_path: str,
    members: tuple[dict[str, str], ...],
    target_dir: str,
) -> set[str]:
  """Extracts a set of files from a gzipped tar archive.

  Args:
    archive_path: The path to the .tgz archive.
    members: Tuple of {'member_suffix', 'output_name'} dicts to extract.
    target_dir: The directory where extracted files should be placed.

  Returns:
    Set of output_name values that were successfully extracted.
  """
  found = set()
  remaining = {spec['member_suffix']: spec for spec in members}
  with tarfile.open(archive_path, 'r:gz') as tar:
    for member in tar.getmembers():
      if not member.isfile() or not remaining:
        continue
      normalized_path = os.path.normpath(member.name)
      if normalized_path.startswith('..') or os.path.isabs(normalized_path):
        continue
      for suffix, spec in list(remaining.items()):
        out_path = _member_target_path(
            normalized_path, suffix, spec['output_name'], target_dir
        )
        if out_path is None:
          continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        source = tar.extractfile(member)
        if source is None:
          break
        with source, open(out_path, 'wb') as target:
          target.write(source.read())
        print(f'Extracted {member.name} -> {out_path}')
        found.add(spec['output_name'])
        remaining.pop(suffix)
        break
  return found


def _extract_zip(
    archive_path: str,
    members: tuple[dict[str, str], ...],
    target_dir: str,
) -> set[str]:
  """Extracts a set of files from a zip archive.

  Args:
    archive_path: The path to the .zip archive.
    members: Tuple of {'member_suffix', 'output_name'} dicts to extract.
    target_dir: The directory where extracted files should be placed.

  Returns:
    Set of output_name values that were successfully extracted.
  """
  found = set()
  remaining = {spec['member_suffix']: spec for spec in members}
  with zipfile.ZipFile(archive_path, 'r') as zipf:
    for info in zipf.infolist():
      if info.is_dir() or not remaining:
        continue
      normalized_path = os.path.normpath(info.filename)
      if normalized_path.startswith('..') or os.path.isabs(normalized_path):
        continue
      for suffix, spec in list(remaining.items()):
        out_path = _member_target_path(
            normalized_path, suffix, spec['output_name'], target_dir
        )
        if out_path is None:
          continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with zipf.open(info) as source, open(out_path, 'wb') as target:
          target.write(source.read())
        print(f'Extracted {info.filename} -> {out_path}')
        found.add(spec['output_name'])
        remaining.pop(suffix)
        break
  return found


def _download_and_extract_compiler(target_dir: str) -> None:
  """Downloads an OpenVINO archive and extracts the NPU compiler libs."""
  archive_key = _select_archive_key()
  if archive_key is None:
    return

  spec = _ARCHIVES[archive_key]
  url = spec['url']
  members = spec['members']
  expected = {m['output_name'] for m in members}

  os.makedirs(target_dir, exist_ok=True)

  with tempfile.TemporaryDirectory() as tmpdir:
    archive_path = os.path.join(tmpdir, os.path.basename(url))
    print(f'Downloading OpenVINO archive from {url}...')
    try:
      urllib.request.urlretrieve(url, archive_path)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'ERROR: Failed to download archive: {e}', file=sys.stderr)
      print(
          'Install will continue without the NPU compiler; Intel NPU AOT'
          ' compile will not work until it is provided.',
          file=sys.stderr,
      )
      return

    print(f'Extracting {sorted(expected)}...')
    try:
      if url.endswith('.zip'):
        found = _extract_zip(archive_path, members, target_dir)
      elif url.endswith('.tgz') or url.endswith('.tar.gz'):
        found = _extract_tar(archive_path, members, target_dir)
      else:
        print(
            f'ERROR: Unsupported archive type for URL: {url}', file=sys.stderr
        )
        return
    except (tarfile.TarError, zipfile.BadZipFile) as e:
      print(f'ERROR: Failed to extract archive: {e}', file=sys.stderr)
      return
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f'ERROR: Unexpected error during archive extraction: {e}',
          file=sys.stderr,
      )
      return

    missing = expected - found
    if missing:
      print(
          f'ERROR: Expected archive members not found: {sorted(missing)}',
          file=sys.stderr,
      )


class CustomBuildPy(_build_py):
  """Runs the NPU compiler download as part of build_py."""

  def run(self):
    print('Preparing Intel OpenVINO SDK...')
    if SKIP_SDK_DOWNLOAD:
      print('SKIP_SDK_DOWNLOAD set; skipping NPU compiler download.')
    else:
      _download_and_extract_compiler(_TARGET_DIR)
    super().run()


setuptools.setup(
    name=PACKAGE_NAME.replace('_', '-'),
    version=PACKAGE_VERSION,
    description='Intel OpenVINO SDK for AI Edge LiteRT',
    long_description='Intel OpenVINO SDK for AI Edge LiteRT.',
    long_description_content_type='text/markdown',
    url='https://www.tensorflow.org/lite/',
    author='Google AI Edge Authors',
    author_email='packages@tensorflow.org',
    license='Apache 2.0',
    include_package_data=True,
    has_ext_modules=lambda: True,
    keywords='litert tflite tensorflow tensor machine learning',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=['ai_edge_litert_sdk_intel'],
    package_dir={'ai_edge_litert_sdk_intel': 'ai_edge_litert_sdk_intel'},
    python_requires='>=3.10',
    install_requires=_openvino_install_requires(),
    cmdclass={
        'build_py': CustomBuildPy,
    },
    zip_safe=False,
)
