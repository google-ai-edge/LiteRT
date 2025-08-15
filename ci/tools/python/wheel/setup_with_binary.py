# Copyright 2024 The AI Edge LiteRT Authors.
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
import stat

import setuptools
from setuptools.command.build_py import build_py as _build_py  # pylint: disable=g-importing-member

PACKAGE_NAME = os.environ['PROJECT_NAME']
PACKAGE_VERSION = os.environ['PACKAGE_VERSION']
DOCLINES = __doc__.split('\n')

EXECUTABLE_FILENAMES = [
    'tools/apply_plugin_main',
]


# This extension does the following:
# 1. Update import statements in .py files to replace 'from litert.python' with
# 'from ai_edge_litert'.
# 2. Set execute permission for executable files.
class CustomBuildPy(_build_py):
  """Command to replace import statements."""

  def run(self):
    super().run()

    if self.dry_run:
      self.announce(
          'Dry run, skipping import replacement.', level=1  # INFO
      )
      return

    target_dir = self.build_lib
    self.announce(
        f"Scanning for .py files in {target_dir} to replace 'from"
        " litert.python' with 'from ai_edge_litert'.",
        level=1,
    )

    modified_count = 0
    for root, _, files in os.walk(target_dir):
      for filename in files:
        if filename.endswith('.py'):
          filepath = os.path.join(root, filename)
          try:
            with open(filepath, 'r', encoding='utf-8') as f:
              content = f.read()

            if 'from litert.python' in content:
              new_content = content.replace(
                  'from litert.python', 'from ai_edge_litert'
              )
              if content != new_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                  f.write(new_content)
                self.announce(f'Replaced imports in: {filepath}', level=1)
                modified_count += 1

          except Exception as e:  # pylint: disable=broad-exception-caught
            self.warn(
                f'Could not process {filepath} for import replacement: {str(e)}'
            )

    if modified_count > 0:
      self.announce(
          f'Finished replacing imports in {modified_count} files.', level=1
      )
    else:
      self.announce(
          "No files were modified as 'from litert.python' was not found or"
          ' files were already compliant.',
          level=1,
      )

    self.announce(
        f'Scanning for files in {target_dir} to add execute permission.',
        level=1,
    )

    files_updated = 0
    for _, _, build_dir, data_filenames_list in self.data_files:
      for data_filename in data_filenames_list:
        if data_filename in EXECUTABLE_FILENAMES:
          filepath_in_build = os.path.join(build_dir, data_filename)
          if os.path.exists(filepath_in_build):
            self.announce(
                f'Setting execute permission for: {filepath_in_build}', level=1
            )
            try:
              current_mode = os.stat(filepath_in_build).st_mode
              new_mode = (
                  current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
              )

              if current_mode == new_mode:
                self.announce(
                    'Execute permission already effectively set for'
                    f' {filepath_in_build}',
                    level=1,
                )
                files_updated += 1
              else:
                os.chmod(filepath_in_build, new_mode)
                self.announce(
                    'Execute permission successfully set for'
                    f' {filepath_in_build}',
                    level=1,
                )
                files_updated += 1
            except Exception as e:  # pylint: disable=broad-exception-caught
              self.warn(
                  'Failed to set execute permission for'
                  f' {filepath_in_build}: {e}'
              )
          else:
            self.warn(
                f'File {filepath_in_build} (from data_files) not found in build'
                ' directory.'
            )
    if files_updated == 0:
      self.warn('No files were updated.')


setuptools.setup(
    name=PACKAGE_NAME.replace('_', '-'),
    version=PACKAGE_VERSION,
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
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
    packages=setuptools.find_packages(exclude=[]),
    package_dir={'': '.'},
    package_data={'': [
        '*.so', '*.pyd', '**/*_main', '**/*.so',
    ]},
    install_requires=[
        'backports.strenum',
        'flatbuffers',
        'numpy >= 1.23.2',  # Better to keep sync with both TF ci_build
        # and OpenCV-Python requirement.
        'tqdm',
        'typing-extensions',
    ],
    extras_require={
        'npu-sdk': ['ai-edge-litert-sdk-qualcomm~=0.1.0',
                    'ai-edge-litert-sdk-mediatek~=0.1.0'],
    },
    # Use the custom command for the build_py step
    cmdclass={
        'build_py': CustomBuildPy,
    },
    zip_safe=False,
)
