# Copyright 2020 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
"""A tool to generate dossiers for code signing.

Provides functionality to generate codesigning dossiers from bundles.
"""

import argparse
import json
import os
import os.path
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple, Union
import uuid

from tools.dossier_codesigningtool import dossier_codesigning_reader as dossier_reader

# A type to access the leaf JSON values from a manifest.
_ManifestJsonValue = Union[str, List[str], Dict[str, str]]


class DossierDirectory(object):
  """Class to manage dossier directories.

  Must used as a context manager.

  Attributes:
    path: The string path to the directory.
    unzipped: A boolean indicating if the dossier was unzipped or already was a
      directory.
  """

  def __init__(self, path, unzipped):
    self.path = path
    self.unzipped = unzipped

  def __enter__(self):
    return self

  def __exit__(self, exception_type, exception_value, traceback):
    if self.unzipped:
      shutil.rmtree(self.path)


# Directories within a bundle that embedded bundles may be present in.
_EMBEDDED_BUNDLE_DIRECTORY_NAMES = [
    'AppClips', 'Extensions', 'PlugIns', 'Frameworks', 'Watch'
]


def generate_arg_parser():
  """Generates an argument parser for this tool."""
  parser = argparse.ArgumentParser(
      description='Tool for generating dossiers for signing iOS bundles.',
      fromfile_prefix_chars='@')
  subparsers = parser.add_subparsers(help='Sub-commands')

  generate_parser = subparsers.add_parser(
      'generate', help='Generate a dossier from a signed bundle.')
  generate_parser.add_argument(
      '--output',
      required=True,
      help='Path to output manifest dossier location.')
  generate_parser.add_argument(
      '--zip',
      action='store_true',
      help='Zip the final dossier into a file at specified location.')
  generate_parser.add_argument(
      '--codesign', required=True, type=str, help='Path to codesign binary')
  generate_parser.add_argument('bundle', help='Path to the bundle')
  generate_parser.set_defaults(func=_generate_manifest_dossier)

  create_parser = subparsers.add_parser('create', help='Create a dossier.')
  create_parser.add_argument(
      '--output',
      required=True,
      help='Path to output manifest dossier location.')
  create_parser.add_argument(
      '--zip',
      action='store_true',
      help='Zip the final dossier into a file at specified location.')
  identity_group = create_parser.add_mutually_exclusive_group(required=True)
  identity_group.add_argument(
      '--codesign_identity', type=str, help='Codesigning identity to be used.')
  identity_group.add_argument(
      '--infer_identity',
      action='store_true',
      help=(
          'Infer the codesigning identity based on provisioning profile at'
          ' signing time. If this option is passed, the provisioning profile is'
          ' mandatory.'
      ),
  )
  create_parser.add_argument(
      '--provisioning_profile',
      type=str,
      help='Optional provisioning profile to be used.')
  create_parser.add_argument(
      '--entitlements_file',
      type=str,
      help='Optional path to optional entitlements')
  create_parser.add_argument(
      '--embedded_dossier',
      action='append',
      default=[],
      nargs=2,
      help=(
          'Specifies an embedded bundle dossier to be included in created'
          ' dossier. Should be in form [relative path of artifact dossier'
          ' signs] [path to dossier]'
      ),
  )
  create_parser.set_defaults(func=_create_dossier)

  embed_parser = subparsers.add_parser(
      'embed',
      help=(
          'Embeds a dossier into an existing dossier. Only supports embedding'
          ' at the top level of the existing dossier.'
      ),
  )
  embed_parser.add_argument(
      '--dossier', required=True, help='Path to dossier location to edit.')
  embed_parser.add_argument(
      '--embedded_relative_artifact_path',
      required=True,
      type=str,
      help='Relative path of artifact the dossier to be embedded signs')
  embed_parser.add_argument(
      '--embedded_dossier_path',
      required=True,
      type=str,
      help='Path to dossier to be embedded')
  embed_parser.set_defaults(func=_embed_dossier)

  return parser


def _extract_codesign_data(
    bundle_path: str,
    output_directory: str,
    unique_id: str,
    codesign_path: str) -> Tuple[Optional[str], Optional[str]]:
  """Extracts codesigning data from the provided bundle to the output directory.

   Given a bundle_path will extract the entitlements file to the provided
   output_directory as well as extract the codesigning identity.

  Args:
    bundle_path: The absolute path to the bundle to extract entitlements from.
    output_directory: The absolute path to the output directory the entitlements
      should be placed in, it must already exist.
    unique_id: Unique identifier to use for filename of extracted entitlements.
    codesign_path: Path to the codesign tool as a string.

  Returns:
    A tuple of the output file name for the entitlements in the output_directory
    and the codesigning identity. If either of these is not available, they will
    be set to None in the tuple.

  Raises:
    OSError: If unable to extract codesign identity.
  """
  command = (codesign_path, '-dvv', '--entitlements', ':-', bundle_path)
  process = subprocess.Popen(
      command,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      encoding='utf8',
      errors='replace')
  output, stderr = process.communicate()
  if process.poll() != 0:
    raise OSError('Fail to extract entitlements from bundle: %s' % stderr)

  if not output:
    return None, None

  signing_info = re.search(r'^Authority=(.*)$', str(stderr), re.MULTILINE)
  if signing_info:
    cert_authority = signing_info.group(1)
  else:
    cert_authority = None

  plist = plistlib.loads(output.encode('utf-8'))
  if not plist:
    return None, cert_authority

  output_file_name = unique_id + '.entitlements'
  output_file_path = os.path.join(output_directory, output_file_name)
  output_file = open(output_file_path, 'w')
  output_file.write(output)
  output_file.close()

  return output_file_name, cert_authority


def _copy_entitlements_file(
    original_entitlements_file_path: str,
    output_directory: str,
    unique_id: str) -> Optional[str]:
  """Copies an entitlements file from an original path to an output directory.

  Args:
    original_entitlements_file_path: The absolute path to the original
      entitlements file.
    output_directory: The absolute path to the output directory the entitlements
      should be placed in, it must already exist.
    unique_id: Unique identifier to use for filename of extracted entitlements.

  Returns:
    The filename relative to output_directory the entitlements were copied to,
    or if the original path does not exist it does nothing and will return
    `None`.
  """
  if os.path.exists(original_entitlements_file_path):
    dest_entitlements_filename = unique_id + '.entitlements'
    dest_entitlements_path = os.path.join(output_directory,
                                          dest_entitlements_filename)
    shutil.copy(original_entitlements_file_path, dest_entitlements_path)
    return dest_entitlements_filename
  else:
    return None


def _copy_provisioning_profile(
    original_provisioning_profile_path: str,
    output_directory: str,
    unique_id: str) -> str:
  """Copies a provisioning profile file from its path to an output directory.

  Args:
    original_provisioning_profile_path: The absolute path to the original
      provisioning profile file. File must exist.
    output_directory: The absolute path to the output directory the profile
      should be placed in, it must already exist.
    unique_id: Unique identifier to use for filename of extracted entitlements.

  Returns:
    The filename relative to output_directory the profile was copied to.
  """
  _, profile_extension = os.path.splitext(original_provisioning_profile_path)
  dest_provisioning_profile_filename = unique_id + profile_extension
  dest_provision_profile_path = os.path.join(output_directory,
                                             dest_provisioning_profile_filename)
  shutil.copy(original_provisioning_profile_path, dest_provision_profile_path)
  return dest_provisioning_profile_filename


def _extract_provisioning_profile(
    bundle_path: str,
    output_directory: str,
    unique_id: str) -> Optional[str]:
  """Extracts the profile for the provided bundle to a destination file name.

  Given a bundle_path will extract the profile file to the provided
  output_directory, and return the filename relative to the output_directory
  that the profile has been placed in, or None if no profile exists.

  Args:
    bundle_path: The absolute path to the bundle to extract profile from.
    output_directory: The absolute path to the output directory the profile
      should be placed in, it must already exist.
    unique_id: Unique identifier to use for filename of extracted profile.

  Returns:
    The filename relative to output_directory the profile was placed in,
    or None if there was no profile found.
  """
  embedded_mobileprovision_path = os.path.join(bundle_path,
                                               'embedded.mobileprovision')
  embedded_provisioning_profile_path = os.path.join(
      bundle_path, 'Contents', 'embedded.provisionprofile')
  if os.path.exists(embedded_mobileprovision_path):
    original_provisioning_profile_path = embedded_mobileprovision_path
  elif os.path.exists(embedded_provisioning_profile_path):
    original_provisioning_profile_path = embedded_provisioning_profile_path
  else:
    return None
  return _copy_provisioning_profile(original_provisioning_profile_path,
                                    output_directory, unique_id)


def _generate_manifest(
    codesign_identity: Optional[str],
    entitlement_file: Optional[str],
    provisioning_profile_file: Optional[str],
    embedded_bundle_manifests: Optional[_ManifestJsonValue],
) -> Dict[str, _ManifestJsonValue]:
  """Generates the manifest based on provided parameters.

  Given a set of code signing parameters, generates a manifest representation
  suitable for inclusion in a codesigning dossier.

  Args:
    codesign_identity: The string representing the codesigning identity to be
      used for signing this bundle. If None is specified, the identity will be
      inferred from the provisioning profile based on the available identities
      when the `sign` command is given. If None is passed, the provisioning
      profile becomes mandatory.
    entitlement_file: The absolute path to the entitlements file to use for
      signing this bundle, or None if no entitlements need to be included.
    provisioning_profile_file: The absolute path to the provisioning profile to
      embed within the signed bundle, or None if none needs to be embedded.
    embedded_bundle_manifests: Manifests for embedded bundles that should be
      included in this manifest, or None if there are none.

  Returns:
    The manifest contents.
  """
  manifest = {}
  if codesign_identity:
    manifest[dossier_reader.CODESIGN_IDENTITY_KEY] = codesign_identity
  if entitlement_file is not None:
    manifest[dossier_reader.ENTITLEMENTS_KEY] = entitlement_file
  if provisioning_profile_file is not None:
    manifest[dossier_reader
             .PROVISIONING_PROFILE_KEY] = provisioning_profile_file
  if embedded_bundle_manifests is not None:
    manifest[dossier_reader
             .EMBEDDED_BUNDLE_MANIFESTS_KEY] = embedded_bundle_manifests
  return manifest


def _embedded_manifests_for_path(
    bundle_path: str,
    dossier_directory: str,
    target_directory: str,
    codesign_path: str,
) -> List[Dict[str, _ManifestJsonValue]]:
  """Generates embedded manifests for a bundle in a sub-directory.

  Provided a bundle, output directory, and a target directory, traverses the
  target directory to find any bundles that are signed, and generate manifests.
  Copies any referenced assets to the output directory.

  Args:
    bundle_path: The absolute path to the bundle that will be searched.
    dossier_directory: The absolute path to the output dossier directory that
      manifest referenced assets will be copied to.
    target_directory: The target directory name, relative to the bundle_path, to
      be traversed.
    codesign_path: Path to the codesign tool as a string.

  Returns:
    A list of manifest contents with the contents they reference copied into
    dossier_directory, or an empty list if no bundles are codesigned.
  """
  if target_directory not in _EMBEDDED_BUNDLE_DIRECTORY_NAMES:
    raise ValueError(
        'Invalid bundle directory for dossier manifest: %s' % target_directory)

  embedded_manifests = []
  target_directory_path = os.path.join(bundle_path, target_directory)
  if os.path.exists(target_directory_path):
    target_directory_contents = os.listdir(target_directory_path)
    target_directory_contents.sort()
    for filename in target_directory_contents:
      absolute_embedded_bundle_path = os.path.join(target_directory_path,
                                                   filename)
      embedded_manifest = _manifest_with_dossier_for_bundle(
          absolute_embedded_bundle_path, dossier_directory, codesign_path)
      if embedded_manifest is not None:
        embedded_manifest[
            dossier_reader.EMBEDDED_RELATIVE_PATH_KEY] = os.path.join(
                target_directory, filename)
        embedded_manifests.append(embedded_manifest)
  return embedded_manifests


def _manifest_with_dossier_for_bundle(
    bundle_path: str,
    dossier_directory: str,
    codesign_path: str) -> Optional[Dict[str, _ManifestJsonValue]]:
  """Generates a manifest and assets for a provided bundle.

  Provided a bundle and output directory, prepares a code signing dossier by
  generating the manifest contents for the bundle referenced and copying any
  assets referenced by the manifest into the dossier folder.

  Args:
    bundle_path: The absolute path to the bundle that a manifest will be
      generated for.
    dossier_directory: The absolute path to the output dossier directory that
      manifest referenced assets will be copied to.
    codesign_path: Path to the codesign tool as a string.

  Returns:
    The manifest contents with files they reference copied into
    dossier_directory.
  """
  unique_id = str(uuid.uuid4())
  entitlements_file, codesign_identity = _extract_codesign_data(
      bundle_path, dossier_directory, unique_id, codesign_path)
  if not codesign_identity:
    return None
  provisioning_profile = _extract_provisioning_profile(bundle_path,
                                                       dossier_directory,
                                                       unique_id)
  embedded_manifests = []
  for embedded_bundle_directory in _EMBEDDED_BUNDLE_DIRECTORY_NAMES:
    embedded_manifests.extend(
        _embedded_manifests_for_path(bundle_path, dossier_directory,
                                     embedded_bundle_directory, codesign_path))
  if not embedded_manifests:
    embedded_manifests = None
  return _generate_manifest(codesign_identity, entitlements_file,
                            provisioning_profile, embedded_manifests)


def _generate_manifest_dossier(parsed_args: argparse.Namespace):
  """Generates a manifest dossier for provided args."""
  bundle_path = parsed_args.bundle
  dossier_directory = parsed_args.output

  packaging_required = False
  if parsed_args.zip:
    dossier_directory = tempfile.mkdtemp()
    packaging_required = True

  if not os.path.exists(dossier_directory):
    os.makedirs(dossier_directory)

  manifest = _manifest_with_dossier_for_bundle(
      os.path.abspath(bundle_path), dossier_directory, parsed_args.codesign)

  _write_manifest(manifest, dossier_directory)
  if packaging_required:
    _zip_dossier(dossier_directory, parsed_args.output)
    shutil.rmtree(dossier_directory)


def _write_manifest(
    manifest: Dict[str, _ManifestJsonValue],
    dossier_directory: str) -> None:
  """Writes a dossier manifest.json file at dossier_directory.

  Args:
    manifest: The dossier manifest to write.
    dossier_directory: Target directory to write the manifest.json file.
  """
  manifest_file = os.path.join(
      dossier_directory, dossier_reader.MANIFEST_FILENAME)
  with open(manifest_file, 'w') as fp:
    json.dump(manifest, fp, sort_keys=True)


def _zip_dossier(dossier_path: str, destination_path: str) -> None:
  """Zips a dossier into a file.

  Args:
    dossier_path: The path to the unzipped dossier.
    destination_path: The file path to place the zipped dossier.

  Raises:
    OSError: If unable to execute packaging command
  """
  command = (
      '/usr/bin/zip',
      '--recurse-paths',
      '--junk-paths',
      '--quiet',
      '--strip-extra',
      '--compression-method', 'store',
      destination_path,
      dossier_path,
  )
  process = subprocess.Popen(
      command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  _, stderr = process.communicate()
  if process.poll() != 0:
    raise OSError('Fail to zip dossier: %s' % stderr)


def _merge_dossier_contents(
    source_dossier_path: str,
    destination_dossier_path: str) -> None:
  """Merges all files except the actual manifest from one dossier to another.

  Args:
    source_dossier_path: The path to the source dossier.
    destination_dossier_path: The path to the destination dossier.
  """
  dossier_files = os.listdir(source_dossier_path)
  for filename in dossier_files:
    if filename == dossier_reader.MANIFEST_FILENAME:
      continue
    shutil.copy(
        os.path.join(source_dossier_path, filename),
        os.path.join(destination_dossier_path, filename))


def _create_dossier(parsed_args: argparse.Namespace):
  """Creates a signing dossier.

  Provided a set of args from generate sub-command, creates a new dossier.

  Args:
    parsed_args: A struct of arguments required for dossier creation that were
      generated from an instance of argparse.ArgumentParser(...).

  Raises:
    SystemExit: If the identity can only be inferred and a provisioning profile
      was not provided.
  """
  dossier_directory = parsed_args.output
  packaging_required = False
  if parsed_args.zip:
    dossier_directory = tempfile.mkdtemp()
    packaging_required = True
  if not os.path.exists(dossier_directory):
    os.makedirs(dossier_directory)
  unique_id = str(uuid.uuid4())

  entitlements_filename = None
  entitlements_file = getattr(parsed_args, 'entitlements_file', None)
  if entitlements_file:
    entitlements_filename = _copy_entitlements_file(entitlements_file,
                                                    dossier_directory,
                                                    unique_id)

  provisioning_profile_filename = None
  provisioning_profile = getattr(parsed_args, 'provisioning_profile', None)
  if provisioning_profile:
    provisioning_profile_filename = _copy_provisioning_profile(
        parsed_args.provisioning_profile, dossier_directory, unique_id)
  if parsed_args.infer_identity and provisioning_profile_filename is None:
    raise SystemExit(
        'A provisioning profile must be provided to infer the signing identity')

  embedded_manifests = []
  for embedded_dossier in getattr(parsed_args, 'embedded_dossier', []):
    embedded_dossier_bundle_relative_path = embedded_dossier[0]
    with dossier_reader.extract_zipped_dossier_if_required(
        embedded_dossier[1]) as embedded_dossier_directory:
      embedded_dossier_path = embedded_dossier_directory.path
      _merge_dossier_contents(embedded_dossier_path, dossier_directory)
      embedded_manifest = dossier_reader.read_manifest_from_dossier(
          embedded_dossier_path)
      embedded_manifest[
          dossier_reader
          .EMBEDDED_RELATIVE_PATH_KEY] = embedded_dossier_bundle_relative_path
      embedded_manifests.append(embedded_manifest)

  codesign_identity = getattr(parsed_args, 'codesign_identity', None)
  manifest = _generate_manifest(codesign_identity, entitlements_filename,
                                provisioning_profile_filename,
                                embedded_manifests)

  _write_manifest(manifest, dossier_directory)
  if packaging_required:
    _zip_dossier(dossier_directory, parsed_args.output)
    shutil.rmtree(dossier_directory)


def _embed_dossier(parsed_args):
  """Embeds an existing dossier into the specified dossier.

  Provided a set of args from generate sub-command, embeds a dossier in a
  dossier.

  Args:
    parsed_args: A struct of arguments required for generating a dossier from a
      signed bundle that were generated from an instance of
      argparse.ArgumentParser(...).

  Raises:
    OSError: If any of specified dossiers are not found.
  """
  with (dossier_reader.extract_zipped_dossier_if_required(
            parsed_args.dossier) as dossier_dir,
        dossier_reader.extract_zipped_dossier_if_required(
            parsed_args.embedded_dossier_path) as embedded_dossier_dir):

    manifest = dossier_reader.read_manifest_from_dossier(dossier_dir.path)
    embedded_manifest = dossier_reader.read_manifest_from_dossier(
        embedded_dossier_dir.path)

    _merge_dossier_contents(embedded_dossier_dir.path, dossier_dir.path)
    embedded_manifest[dossier_reader.EMBEDDED_RELATIVE_PATH_KEY] = (
        parsed_args.embedded_relative_artifact_path)
    manifest[dossier_reader.EMBEDDED_BUNDLE_MANIFESTS_KEY].append(
        embedded_manifest)

    _write_manifest(manifest, dossier_dir.path)
    if dossier_dir.unzipped:
      _zip_dossier(dossier_dir.path, parsed_args.dossier)


if __name__ == '__main__':
  args = generate_arg_parser().parse_args()
  sys.exit(args.func(args))
