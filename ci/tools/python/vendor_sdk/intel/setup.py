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
_OV_BUILD = '2026.1.0.21367.63e31528c62'
_OV_BASE_URL = (
    'https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.1'
)

# Archive -> in-archive member is always <archive_prefix>/<member_suffix>, where
# archive_prefix is the archive filename minus the extension.
_ARCHIVES = {
    'ubuntu22': {
        'url': f'{_OV_BASE_URL}/linux/openvino_toolkit_ubuntu22_{_OV_BUILD}_x86_64.tgz',
        'member_suffix': 'runtime/lib/intel64/libopenvino_intel_npu_compiler.so',
        'output_name': 'libopenvino_intel_npu_compiler.so',
    },
    'ubuntu24': {
        'url': f'{_OV_BASE_URL}/linux/openvino_toolkit_ubuntu24_{_OV_BUILD}_x86_64.tgz',
        'member_suffix': 'runtime/lib/intel64/libopenvino_intel_npu_compiler.so',
        'output_name': 'libopenvino_intel_npu_compiler.so',
    },
    'windows': {
        'url': f'{_OV_BASE_URL}/windows/openvino_toolkit_windows_{_OV_BUILD}_x86_64.zip',
        'member_suffix': 'runtime/bin/intel64/Release/openvino_intel_npu_compiler.dll',
        'output_name': 'openvino_intel_npu_compiler.dll',
    },
}

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
    member_suffix: str,
    output_name: str,
    target_dir: str,
) -> bool:
  with tarfile.open(archive_path, 'r:gz') as tar:
    for member in tar.getmembers():
      if not member.isfile():
        continue
      normalized_path = os.path.normpath(member.name)
      if normalized_path.startswith('..') or os.path.isabs(normalized_path):
        continue
      out_path = _member_target_path(
          normalized_path, member_suffix, output_name, target_dir
      )
      if out_path is None:
        continue
      os.makedirs(os.path.dirname(out_path), exist_ok=True)
      source = tar.extractfile(member)
      if source is None:
        continue
      with source, open(out_path, 'wb') as target:
        target.write(source.read())
      print(f'Extracted {member.name} -> {out_path}')
      return True
  return False


def _extract_zip(
    archive_path: str,
    member_suffix: str,
    output_name: str,
    target_dir: str,
) -> bool:
  with zipfile.ZipFile(archive_path, 'r') as zipf:
    for info in zipf.infolist():
      if info.is_dir():
        continue
      normalized_path = os.path.normpath(info.filename)
      if normalized_path.startswith('..') or os.path.isabs(normalized_path):
        continue
      out_path = _member_target_path(
          normalized_path, member_suffix, output_name, target_dir
      )
      if out_path is None:
        continue
      os.makedirs(os.path.dirname(out_path), exist_ok=True)
      with zipf.open(info) as source, open(out_path, 'wb') as target:
        target.write(source.read())
      print(f'Extracted {info.filename} -> {out_path}')
      return True
  return False


def _download_and_extract_compiler(target_dir: str) -> None:
  """Downloads an OpenVINO archive and extracts just the NPU compiler lib."""
  archive_key = _select_archive_key()
  if archive_key is None:
    return

  spec = _ARCHIVES[archive_key]
  url = spec['url']
  member_suffix = spec['member_suffix']
  output_name = spec['output_name']

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

    print(f'Extracting {output_name}...')
    try:
      if url.endswith('.zip'):
        found = _extract_zip(
            archive_path, member_suffix, output_name, target_dir
        )
      elif url.endswith('.tgz') or url.endswith('.tar.gz'):
        found = _extract_tar(
            archive_path, member_suffix, output_name, target_dir
        )
      else:
        print(f'ERROR: Unsupported archive type for URL: {url}', file=sys.stderr)
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

    if not found:
      print(
          f'ERROR: Member ending in {member_suffix!r} not found in archive.',
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
    install_requires=[
        'openvino==2026.1.0',
    ],
    cmdclass={
        'build_py': CustomBuildPy,
    },
    zip_safe=False,
)
