# Copyright 2017 Google Inc.
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

"""Utility methods for managing Apple bundles."""

import glob
import os
import subprocess
import tempfile

from xctestrunner.shared import ios_errors
from xctestrunner.shared import plist_util


def ExtractApp(compressed_app_path, working_dir):
  """Creates a temp directory and extracts compressed file of the app there.

  Args:
    compressed_app_path: string, full path of compressed file. The file
      extension name must end with .ipa.
    working_dir: string, the working directory where the extracted bundle
      places.

  Returns:
    string, the path of extracted bundle, which is
      {working_dir}/*/Payload/*.app

  Raises:
    BundleError: when bundle is not found in extracted IPA or multiple files are
      found in the extracted directory.
  """
  if not compressed_app_path.endswith('.ipa'):
    ios_errors.BundleError(
        'The extension of the compressed file should be .ipa.')
  unzip_target_dir = tempfile.mkdtemp(dir=working_dir)
  _UnzipWithShell(compressed_app_path, unzip_target_dir)
  return _ExtractBundleFile('%s/Payload' % unzip_target_dir, 'app')


def ExtractTestBundle(compressed_test_path, working_dir):
  """Creates a temp directory and extracts compressed file of the test bundle.

  Args:
    compressed_test_path: string, full path of compressed file. The file
      extension name must end with .ipa/.zip.
    working_dir: string, the working directory where the extracted bundle
      places.

  Returns:
    string, the path of extracted bundle, which is
      {working_dir}/*/Payload/*.xctest or {working_dir}/*/*.xctest

  Raises:
    BundleError: when bundle is not found in extracted IPA or multiple files are
      found in the extracted directory.
  """
  if not (compressed_test_path.endswith('.ipa') or
          compressed_test_path.endswith('.zip')):
    ios_errors.BundleError(
        'The extension of the compressed file should be .ipa/zip.')
  unzip_target_dir = tempfile.mkdtemp(dir=working_dir)
  _UnzipWithShell(compressed_test_path, unzip_target_dir)
  try:
    return _ExtractBundleFile(unzip_target_dir, 'xctest')
  except ios_errors.BundleError:
    return _ExtractBundleFile('%s/Payload' % unzip_target_dir, 'xctest')


def GetMinimumOSVersion(bundle_path):
  """Gets the minimum OS version of the bundle deployment.

  Args:
    bundle_path: string, full path of bundle folder.

  Returns:
    string, the minimum OS version of the provided bundle.

  Raises:
    ios_errors.PlistError: the MinimumOSVersion does not exist in the bundle's
      Info.plist.
  """
  info_plist = os.path.join(bundle_path, 'Info.plist')
  return plist_util.Plist(info_plist).GetPlistField('MinimumOSVersion')


def GetBundleId(bundle_path):
  """Gets the bundle ID of the bundle.

  Args:
    bundle_path: string, full path of bundle folder.

  Returns:
    string, the bundle ID of the provided bundle.

  Raises:
    ios_errors.PlistError: the CFBundleIdentifier does not exist in the bundle's
      Info.plist.
  """
  info_plist = os.path.join(bundle_path, 'Info.plist')
  return plist_util.Plist(info_plist).GetPlistField('CFBundleIdentifier')


def GetCodesignIdentity(bundle_path):
  """Gets the codesign identity which signs the bundle with.

  Args:
    bundle_path: string, full path of bundle folder.

  Returns:
    string, the codesign identity which signs the bundle with.

  Raises:
    ios_errors.BundleError: when failed to get the signing identity from the
      bundle.
  """
  command = ('codesign', '-dvv', bundle_path)
  process = subprocess.Popen(command, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
  output = process.communicate()[0].decode('utf-8')
  for line in output.split('\n'):
    if line.startswith('Authority='):
      return line[len('Authority='):]

  raise ios_errors.BundleError('Failed to extract signing identity from %s' %
                               output)


def GetDevelopmentTeam(bundle_path):
  """Gets the development team of the bundle.

  Args:
    bundle_path: string, full path of bundle folder.

  Returns:
    string, the development team of the provided bundle.

  Raises:
    ios_errors.BundleError: when failed to get the development team from the
      bundle.
  """
  command = ('codesign', '-dvv', bundle_path)
  process = subprocess.Popen(command, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
  output = process.communicate()[0].decode('utf-8')
  for line in output.split('\n'):
    if line.startswith('TeamIdentifier='):
      return line[len('TeamIdentifier='):]

  raise ios_errors.BundleError('Failed to extract development team from %s' %
                               output)


def CodesignBundle(bundle_path,
                   entitlements_plist_path=None,
                   identity=None):
  """Codesigns the bundle.

  Args:
    bundle_path: string, full path of bundle folder.
    entitlements_plist_path: string, the path of the Entitlement to sign bundle.
    identity: string, the identity to sign bundle.

  Raises:
    ios_errors.BundleError: when failed to codesign the bundle.
  """
  if identity is None:
    identity = GetCodesignIdentity(bundle_path)
  try:
    if entitlements_plist_path is None:
      subprocess.check_call(
          [
              'codesign', '-f', '--preserve-metadata=identifier,entitlements',
              '--timestamp=none', '-s', identity, bundle_path
          ],
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE)
    else:
      subprocess.check_call(
          [
              'codesign', '-f', '--entitlements', entitlements_plist_path,
              '--timestamp=none', '-s', identity, bundle_path
          ],
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE)
  except subprocess.CalledProcessError as e:
    raise ios_errors.BundleError(
        'Failed to codesign the bundle %s with %s: %s' %
        (bundle_path, identity, e.output))


def EnableUIFileSharing(bundle_path, resigning=True):
  """Enable the UIFileSharingEnabled field in the bundle's Info.plist.

  Args:
    bundle_path: string, full path of bundle folder.
    resigning: bool, whether resigning the bundle after enable
               UIFileSharingEnabled.

  Raises:
    ios_errors.BundleError: when failed to codesign the bundle.
  """
  info_plist = plist_util.Plist(os.path.join(bundle_path, 'Info.plist'))
  info_plist.SetPlistField('UIFileSharingEnabled', True)
  if resigning:
    CodesignBundle(bundle_path)


def GetFileArchTypes(file_path):
  """Gets the architecture types of the file."""
  output = subprocess.check_output(['/usr/bin/lipo', file_path,
                                    '-archs']).decode('utf-8').strip()
  return output.split(' ')


def RemoveArchType(file_path, arch_type):
  """Remove the given architecture types for the file."""
  subprocess.check_call(
      ['/usr/bin/lipo', file_path, '-remove', arch_type, '-output', file_path])


def _ExtractBundleFile(target_dir, bundle_extension):
  """Extract single bundle file with given extension.

  Args:
    target_dir: string, the direcotry to be fetched bundle file.
    bundle_extension: string, the extension of the extracted bundle.

  Returns:
    string, the path of extracted bundle which is with given extension.

  Raises:
    BundleError: when bundle is not found or multiple bundles are found in the
      directory.
  """
  extracted_bundles = glob.glob('%s/*.%s' % (target_dir, bundle_extension))
  if not extracted_bundles:
    raise ios_errors.BundleError(
        'No file with extension %s is found under %s.'
        % (bundle_extension, target_dir))
  if len(extracted_bundles) > 1:
    raise ios_errors.BundleError(
        'Multiple files with extension %s are found under %s. Can not '
        'determine which is the target bundle. The files are %s.'
        % (bundle_extension, target_dir, extracted_bundles))
  return extracted_bundles[0]


def _UnzipWithShell(src_file_path, des_file_path):
  """Unzips the file in shell.

  Args:
    src_file_path: string, full path of the file to be unzipped.
    des_file_path: string, full path of the extracted file.
  """
  # Python zipfile extractall method silently removes file permission bits.
  # See https://bugs.python.org/issue15795. As workaround for now, use shell
  # command unzip.
  subprocess.check_call(['unzip', '-q', '-o',
                         src_file_path, '-d', des_file_path])
