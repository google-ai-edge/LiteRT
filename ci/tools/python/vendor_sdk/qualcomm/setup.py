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
import sys
import tarfile
import tempfile
import urllib.request

import setuptools
from setuptools.command.build_py import build_py as _build_py  # pylint: disable=g-importing-member

PACKAGE_NAME = 'ai_edge_litert_sdk_qualcomm'
PACKAGE_VERSION = '0.0.0'


# --- Configuration for Qualcomm SDK Download ---
# TODO(lukeboyer): Update the URL to contain the Qairt SDK version.
# Qairt version doesn't not necessarily match the SDK version though.
QAIRT_URL = (
    'https://storage.googleapis.com/litert/litert_qualcomm_sdk_release.tar.gz'
)
QAIRT_CONTENT_DIR = 'latest'
QAIRT_TARGET_DIR = 'ai_edge_litert_sdk_qualcomm/data'
# ---


def _download_tarball_and_extract(
    tarball_url: str, prefix_to_strip: str, target_dir: str
):
  """Download tarball archieve, extracts and copy."""
  os.makedirs(target_dir, exist_ok=True)

  # Use a temporary directory for downloading and intermediate extraction
  with tempfile.TemporaryDirectory() as tmpdir:
    tarball_name_local = os.path.join(tmpdir, os.path.basename(tarball_url))

    print(f'Downloading SDK from {tarball_url}...')
    try:
      urllib.request.urlretrieve(tarball_url, tarball_name_local)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'ERROR: Failed to download SDK: {e}', file=sys.stderr)
      print(
          'Please ensure you have an active internet connection and the URL is'
          ' correct.',
          file=sys.stderr,
      )
      return

    print('Extracting SDK files...')
    try:
      with tarfile.open(tarball_name_local, 'r:gz') as tar:
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

    except tarfile.TarError as e:
      print(f'ERROR: Failed to extract tarball: {e}', file=sys.stderr)
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
    _download_tarball_and_extract(
        QAIRT_URL, QAIRT_CONTENT_DIR, QAIRT_TARGET_DIR
    )

    super().run()


setuptools.setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    description='Qualcomm SDK for AI Edge LiteRT',
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
    packages=['ai_edge_litert_sdk_qualcomm'],
    package_dir={'ai_edge_litert_sdk_qualcomm': 'ai_edge_litert_sdk_qualcomm'},
    # Use the custom command for the build_py step
    cmdclass={
        'build_py': CustomBuildPy,
    },
    zip_safe=False,
)
