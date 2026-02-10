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
"""LiteRT is for mobile and embedded devices.

LiteRT is the official solution for running machine learning models on mobile
and embedded devices. It enables on-device machine learning inference with low
latency and a small binary size on Android, iOS, and other operating systems.
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

SKIP_SDK_DOWNLOAD = os.environ.get('SKIP_SDK_DOWNLOAD', False)


# Platform information
IS_LINUX = sys.platform == 'linux'
IS_X86_ARCHITECTURE = platform.machine() in ('x86_64', 'i386', 'i686')


# --- Configuration for MediaTek NeuroPilot SDK Download ---
# NeuroPilot version doesn't not necessarily match the SDK version though.
NEUROPILOT_URL = 'https://s3.ap-southeast-1.amazonaws.com/mediatek.neuropilot.com/66f2c33a-2005-4f0b-afef-2053c8654e4f.gz'  # pylint: disable=line-too-long
NEUROPILOT_CONTENT_DIR = 'neuro_pilot'
NEUROPILOT_TARGET_DIR = 'ai_edge_litert_sdk_mediatek/data'
# ---


def _download_and_extract(
    tarball_url: str, prefix_to_strip: str, target_dir: str
):
  """Download archive, extracts and copy."""
  if not (IS_LINUX and IS_X86_ARCHITECTURE):
    print(
        'IGNORED: Currently LiteRT NPU AOT for MediaTek is only supported on'
        ' Linux x86 architecture.'
    )
    return

  os.makedirs(target_dir, exist_ok=True)

  with tempfile.TemporaryDirectory() as tmpdir:
    archive_name_local = os.path.join(tmpdir, os.path.basename(tarball_url))

    print(f'Downloading SDK from {tarball_url}...')
    try:
      urllib.request.urlretrieve(tarball_url, archive_name_local)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'ERROR: Failed to download SDK: {e}', file=sys.stderr)
      print(
          'Please ensure you have an active internet connection.',
          file=sys.stderr,
      )
      return

    print('Extracting SDK files...')
    try:
      if tarball_url.endswith('.zip'):
        archive_type = 'zip'
      elif (
          tarball_url.endswith('.tar.gz')
          or tarball_url.endswith('.tgz')
          or tarball_url.endswith('.gz')
      ):
        archive_type = 'tar.gz'
      else:
        print(
            f'ERROR: Unsupported archive type for URL: {tarball_url}',
            file=sys.stderr,
        )
        return

      if archive_type == 'zip':
        with zipfile.ZipFile(archive_name_local, 'r') as zipf:
          for member_info in zipf.infolist():
            original_name = member_info.filename
            # Ensure paths are normalized and secure (prevent path traversal)
            if (
                original_name.startswith(prefix_to_strip + '/')
                and original_name != prefix_to_strip + '/'
            ):
              path_inside_subdir = original_name[len(prefix_to_strip) + 1 :]

              if not path_inside_subdir:
                continue

              normalized_path = os.path.normpath(path_inside_subdir)
              if normalized_path.startswith('..') or os.path.isabs(
                  normalized_path
              ):
                print(
                    'WARNING: Skipping potentially malicious path:'
                    f' {original_name}',
                    file=sys.stderr,
                )
                continue

              target_path_for_member = os.path.join(target_dir, normalized_path)

              if member_info.is_dir():
                os.makedirs(target_path_for_member, exist_ok=True)
              else:  # It's a file
                os.makedirs(
                    os.path.dirname(target_path_for_member), exist_ok=True
                )
                with zipf.open(member_info) as source, open(
                    target_path_for_member, 'wb'
                ) as target:
                  target.write(source.read())

      elif archive_type == 'tar.gz':
        with tarfile.open(archive_name_local, 'r:gz') as tar:
          members_to_extract = []
          for member in tar.getmembers():
            if (
                member.name.startswith(prefix_to_strip + '/')
                and member.name != prefix_to_strip + '/'
            ):
              path_inside_subdir = member.name[len(prefix_to_strip) + 1 :]

              if not path_inside_subdir:  # Skip if it's empty after stripping
                continue

              normalized_path = os.path.normpath(path_inside_subdir)
              if normalized_path.startswith('..') or os.path.isabs(
                  normalized_path
              ):
                print(
                    'WARNING: Skipping potentially malicious path:'
                    f' {member.name}',
                    file=sys.stderr,
                )
                continue

              member_copy = tar.getmember(member.name)
              member_copy.name = normalized_path
              members_to_extract.append(member_copy)

          if members_to_extract:
            tar.extractall(path=target_dir, members=members_to_extract)
            print(
                f"SDK files from '{prefix_to_strip}/' extracted to {target_dir}"
            )
          else:
            print(f"No files found under '{prefix_to_strip}/' in the tarball.")

    except (
        zipfile.BadZipFile,
        tarfile.TarError,
    ) as e:  # Catch both zipfile and tarfile specific errors
      print(f'ERROR: Failed to extract archive: {e}', file=sys.stderr)
      return
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f'ERROR: An unexpected error occurred during SDK extraction: {e}',
          file=sys.stderr,
      )
      raise SystemExit('Install SDK failed. Aborting installation.') from e


class CustomBuildPy(_build_py):
  """Command to replace import statements."""

  def run(self):

    print('Preparing SDK...')
    if SKIP_SDK_DOWNLOAD:
      print('Skipping SDK download...')
    else:
      _download_and_extract(
          NEUROPILOT_URL, NEUROPILOT_CONTENT_DIR, NEUROPILOT_TARGET_DIR
      )

    super().run()


setuptools.setup(
    name=PACKAGE_NAME.replace('_', '-'),
    version=PACKAGE_VERSION,
    description='MediaTek NeuroPilot SDK for AI Edge LiteRT',
    long_description='MediaTek NeuroPilot SDK for AI Edge LiteRT.',
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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=['ai_edge_litert_sdk_mediatek'],
    package_dir={'ai_edge_litert_sdk_mediatek': 'ai_edge_litert_sdk_mediatek'},
    # Use the custom command for the build_py step
    cmdclass={
        'build_py': CustomBuildPy,
    },
    zip_safe=False,
)
