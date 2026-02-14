# coding=utf-8
# Copyright 2021 The Bazel Authors. All rights reserved.
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
# limitations under the License.
"""Tests for dossier_codesigningtool_reader."""

import concurrent.futures
import contextlib
import io
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from tools.dossier_codesigningtool import dossier_codesigning_reader


_FAKE_MANIFEST = {
    'codesign_identity': '-',
    'embedded_bundle_manifests': [
        {
            'codesign_identity': '-',
            'embedded_bundle_manifests': [],
            'embedded_relative_path': 'Extensions/AppIntentsExtension.appex',
            'entitlements': 'fake.entitlements',
            'provisioning_profile': 'fake.mobileprovision'
        },
        {
            'codesign_identity': '-',
            'embedded_bundle_manifests': [],
            'embedded_relative_path': 'PlugIns/IntentsExtension.appex',
            'entitlements': 'fake.entitlements',
            'provisioning_profile': 'fake.mobileprovision'
        },
        {
            'codesign_identity': '-',
            'embedded_bundle_manifests': [],
            'embedded_relative_path': 'PlugIns/IntentsUIExtension.appex',
            'entitlements': 'fake.entitlements',
            'provisioning_profile': 'fake.mobileprovision'
        },
        {
            'codesign_identity': '-',
            'embedded_bundle_manifests': [{
                'codesign_identity': '-',
                'embedded_bundle_manifests': [],
                'embedded_relative_path': 'PlugIns/WatchExtension.appex',
                'entitlements': 'fake.entitlements',
                'provisioning_profile': 'fake.mobileprovision',
            }],
            'embedded_relative_path': 'Watch/WatchApp.app',
            'entitlements': 'fake.entitlements',
            'provisioning_profile': 'fake.mobileprovision',
        },
    ],
    'entitlements': 'fake.entitlements',
    'provisioning_profile': 'fake.mobileprovision',
}

_FAKE_SIMPLE_MANIFEST_NO_PROFILE = {
    'codesign_identity': '-',
    'embedded_bundle_manifests': [],
}

_IPA_WORKSPACE_PATH = 'test/starlark_tests/targets_under_test/ios/app.ipa'
_IPA_W_WATCHOS_WORKSPACE_PATH = 'test/starlark_tests/targets_under_test/watchos/app_companion.ipa'
_COMBINED_ZIP_W_WATCHOS_WORKSPACE_PATH = 'test/starlark_tests/targets_under_test/watchos/app_companion_dossier_with_bundle.zip'

_ADDITIONAL_SIGNING_KEYCHAIN = '/tmp/Library/Keychains/ios-dev-signing.keychain'


class DossierCodesigningReaderTest(unittest.TestCase):

  @mock.patch.object(dossier_codesigning_reader, '_invoke_codesign')
  def test_sign_bundle_with_manifest_codesign_invocations(self, mock_codesign):
    mock.patch('shutil.copy').start()
    dossier_codesigning_reader._sign_bundle_with_manifest(
        root_bundle_path='/tmp/fake.app/',
        manifest=_FAKE_MANIFEST,
        dossier_directory_path='/tmp/dossier/',
        codesign_path='/usr/bin/fake_codesign',
        signing_keychain=_ADDITIONAL_SIGNING_KEYCHAIN,
        certificates_directory_path='/tmp/certs/',
        override_codesign_identity='-',
        allowed_entitlements=None)

    self.assertEqual(mock_codesign.call_count, 6)
    actual_paths = [
        mock_codesign.call_args_list[0][1]['full_path_to_sign'],
        mock_codesign.call_args_list[1][1]['full_path_to_sign'],
        mock_codesign.call_args_list[2][1]['full_path_to_sign'],
        mock_codesign.call_args_list[3][1]['full_path_to_sign'],
        mock_codesign.call_args_list[4][1]['full_path_to_sign'],
        mock_codesign.call_args_list[5][1]['full_path_to_sign'],
    ]
    expected_paths = [
        '/tmp/fake.app/Extensions/AppIntentsExtension.appex',
        '/tmp/fake.app/PlugIns/IntentsExtension.appex',
        '/tmp/fake.app/PlugIns/IntentsUIExtension.appex',
        '/tmp/fake.app/Watch/WatchApp.app/PlugIns/WatchExtension.appex',
        '/tmp/fake.app/Watch/WatchApp.app',
        '/tmp/fake.app/',
    ]
    self.assertSetEqual(set(actual_paths), set(expected_paths))

    # assert codesign threads block correctly (executed bottom-up)
    self.assertLess(
        actual_paths.index(
            '/tmp/fake.app/Watch/WatchApp.app/PlugIns/WatchExtension.appex'
        ),
        actual_paths.index('/tmp/fake.app/Watch/WatchApp.app'),
    )
    self.assertLess(
        actual_paths.index(
            '/tmp/fake.app/Watch/WatchApp.app/PlugIns/WatchExtension.appex'
        ),
        actual_paths.index('/tmp/fake.app/'),
    )
    self.assertLess(
        actual_paths.index('/tmp/fake.app/Watch/WatchApp.app'),
        actual_paths.index('/tmp/fake.app/'),
    )
    self.assertLess(
        actual_paths.index(
            '/tmp/fake.app/Extensions/AppIntentsExtension.appex'
        ),
        actual_paths.index('/tmp/fake.app/'),
    )
    self.assertLess(
        actual_paths.index('/tmp/fake.app/PlugIns/IntentsExtension.appex'),
        actual_paths.index('/tmp/fake.app/'),
    )
    self.assertLess(
        actual_paths.index('/tmp/fake.app/PlugIns/IntentsUIExtension.appex'),
        actual_paths.index('/tmp/fake.app/'),
    )

  @mock.patch.object(dossier_codesigning_reader, '_invoke_codesign')
  def test_sign_adhoc_simple_bundle_with_manifest_codesign_invocations(
      self, mock_codesign
  ):
    mock.patch('shutil.copy').start()
    dossier_codesigning_reader._sign_bundle_with_manifest(
        root_bundle_path='/tmp/fake.app/',
        manifest=_FAKE_SIMPLE_MANIFEST_NO_PROFILE,
        dossier_directory_path='/tmp/dossier/',
        codesign_path='/usr/bin/fake_codesign',
        signing_keychain=_ADDITIONAL_SIGNING_KEYCHAIN,
        certificates_directory_path='/tmp/certs/',
        override_codesign_identity='-',
        allowed_entitlements=None)
    self.assertEqual(mock_codesign.call_count, 1)
    actual_paths = [
        mock_codesign.call_args_list[0][1]['full_path_to_sign'],
    ]
    expected_paths = [
        '/tmp/fake.app/',
    ]
    self.assertSetEqual(set(actual_paths), set(expected_paths))

  @mock.patch.object(
      dossier_codesigning_reader, '_generate_entitlements_for_signing'
  )
  @mock.patch.object(dossier_codesigning_reader, '_invoke_codesign')
  def test_sign_bundle_with_allowed_entitlements(
      self, mock_codesign, mock_gen_entitlements
  ):
    mock.patch('shutil.copy').start()
    mock_gen_entitlements.return_value = None
    dossier_codesigning_reader._sign_bundle_with_manifest(
        root_bundle_path='/tmp/fake.app/',
        manifest=_FAKE_MANIFEST,
        dossier_directory_path='/tmp/dossier/',
        codesign_path='/usr/bin/fake_codesign',
        signing_keychain=_ADDITIONAL_SIGNING_KEYCHAIN,
        certificates_directory_path='/tmp/certs/',
        override_codesign_identity='-',
        allowed_entitlements=['test-an-entitlement'])

    self.assertEqual(mock_codesign.call_count, 6)
    self.assertEqual(mock_gen_entitlements.call_count, 6)
    actual_src_paths = [
        mock_gen_entitlements.call_args_list[0][1]['src'],
        mock_gen_entitlements.call_args_list[1][1]['src'],
        mock_gen_entitlements.call_args_list[2][1]['src'],
        mock_gen_entitlements.call_args_list[3][1]['src'],
        mock_gen_entitlements.call_args_list[4][1]['src'],
        mock_gen_entitlements.call_args_list[5][1]['src'],
    ]
    expected_src_paths = [
        '/tmp/dossier/fake.entitlements',
        '/tmp/dossier/fake.entitlements',
        '/tmp/dossier/fake.entitlements',
        '/tmp/dossier/fake.entitlements',
        '/tmp/dossier/fake.entitlements',
        '/tmp/dossier/fake.entitlements',
    ]
    self.assertSetEqual(set(actual_src_paths), set(expected_src_paths))

    actual_allowed_entitlements = [
        mock_gen_entitlements.call_args_list[0][1]['allowed_entitlements'],
        mock_gen_entitlements.call_args_list[1][1]['allowed_entitlements'],
        mock_gen_entitlements.call_args_list[2][1]['allowed_entitlements'],
        mock_gen_entitlements.call_args_list[3][1]['allowed_entitlements'],
        mock_gen_entitlements.call_args_list[4][1]['allowed_entitlements'],
        mock_gen_entitlements.call_args_list[5][1]['allowed_entitlements'],
    ]
    expected_allowed_entitlements = [
        ['test-an-entitlement'],
        ['test-an-entitlement'],
        ['test-an-entitlement'],
        ['test-an-entitlement'],
        ['test-an-entitlement'],
        ['test-an-entitlement'],
    ]
    self.assertListEqual(
        actual_allowed_entitlements, expected_allowed_entitlements)

    # Make sure that the generated entitlements are passed to codesigning.
    actual_dest_paths = [
        mock_gen_entitlements.call_args_list[0][1]['dest'],
        mock_gen_entitlements.call_args_list[1][1]['dest'],
        mock_gen_entitlements.call_args_list[2][1]['dest'],
        mock_gen_entitlements.call_args_list[3][1]['dest'],
        mock_gen_entitlements.call_args_list[4][1]['dest'],
        mock_gen_entitlements.call_args_list[5][1]['dest'],
    ]
    actual_codesign_entitlements_paths = [
        mock_codesign.call_args_list[0][1]['entitlements_path'],
        mock_codesign.call_args_list[1][1]['entitlements_path'],
        mock_codesign.call_args_list[2][1]['entitlements_path'],
        mock_codesign.call_args_list[3][1]['entitlements_path'],
        mock_codesign.call_args_list[4][1]['entitlements_path'],
        mock_codesign.call_args_list[5][1]['entitlements_path'],
    ]
    self.assertSetEqual(
        set(actual_dest_paths), set(actual_codesign_entitlements_paths))

  @mock.patch.object(
      dossier_codesigning_reader, '_fetch_preferred_signing_identity')
  def test_sign_bundle_with_manifest_raises_identity_infer_error(
      self, mock_fetch_preferred_signing_identity):
    fake_manifest = {'provisioning_profile': 'fake.mobileprovision'}
    mock_fetch_preferred_signing_identity.return_value = None

    with self.assertRaisesRegex(SystemExit, 'unable to infer identity'):
      dossier_codesigning_reader._sign_bundle_with_manifest(
          root_bundle_path='/tmp/fake.app/',
          manifest=fake_manifest,
          dossier_directory_path='/tmp/dossier/',
          signing_keychain=_ADDITIONAL_SIGNING_KEYCHAIN,
          certificates_directory_path='/tmp/certs/',
          codesign_path='/usr/bin/fake_codesign',
          allowed_entitlements=None)

  @mock.patch.object(dossier_codesigning_reader, '_sign_bundle_with_manifest')
  def test_sign_embedded_bundles_with_manifest(self, mock_sign_bundle):
    mock_sign_bundle.return_value = concurrent.futures.Future()
    executor = concurrent.futures.ThreadPoolExecutor()
    futures = dossier_codesigning_reader._sign_embedded_bundles_with_manifest(
        manifest=_FAKE_MANIFEST,
        root_bundle_path='/tmp/fake.app/',
        dossier_directory_path='/tmp/dossier/',
        codesign_path='/usr/bin/fake_codesign',
        allowed_entitlements=None,
        signing_keychain=_ADDITIONAL_SIGNING_KEYCHAIN,
        certificates_directory_path='/tmp/certs/',
        codesign_identity='-',
        executor=executor)
    dossier_codesigning_reader._wait_signing_futures(futures)
    self.assertEqual(len(futures), 4)
    self.assertEqual(mock_sign_bundle.call_count, 4)
    default_args = (
        '/tmp/dossier/',
        '/usr/bin/fake_codesign',
        None,
        _ADDITIONAL_SIGNING_KEYCHAIN,
        '/tmp/certs/',
        '-',
        executor,
    )
    mock_sign_bundle.assert_has_calls([
        mock.call(
            '/tmp/fake.app/Extensions/AppIntentsExtension.appex',
            {
                'codesign_identity': '-',
                'embedded_bundle_manifests': [],
                'embedded_relative_path': (
                    'Extensions/AppIntentsExtension.appex'
                ),
                'entitlements': 'fake.entitlements',
                'provisioning_profile': 'fake.mobileprovision',
            },
            *default_args),
        mock.call(
            '/tmp/fake.app/PlugIns/IntentsExtension.appex',
            {
                'codesign_identity': '-',
                'embedded_bundle_manifests': [],
                'embedded_relative_path': 'PlugIns/IntentsExtension.appex',
                'entitlements': 'fake.entitlements',
                'provisioning_profile': 'fake.mobileprovision'
            },
            *default_args),
        mock.call(
            '/tmp/fake.app/PlugIns/IntentsUIExtension.appex',
            {
                'codesign_identity': '-',
                'embedded_bundle_manifests': [],
                'embedded_relative_path': 'PlugIns/IntentsUIExtension.appex',
                'entitlements': 'fake.entitlements',
                'provisioning_profile': 'fake.mobileprovision'
            },
            *default_args),
        mock.call(
            '/tmp/fake.app/Watch/WatchApp.app',
            {
                'codesign_identity': '-',
                'embedded_bundle_manifests': [{
                    'codesign_identity': '-',
                    'embedded_bundle_manifests': [],
                    'embedded_relative_path': 'PlugIns/WatchExtension.appex',
                    'entitlements': 'fake.entitlements',
                    'provisioning_profile': 'fake.mobileprovision'
                }],
                'embedded_relative_path': 'Watch/WatchApp.app',
                'entitlements': 'fake.entitlements',
                'provisioning_profile': 'fake.mobileprovision'
            },
            *default_args),
    ])

  @mock.patch('shutil.copy')
  @mock.patch('os.path.exists')
  def test_copy_embedded_provisioning_profile(self, mock_exists, mock_copy):
    mock_exists.return_value = False
    dossier_codesigning_reader._copy_embedded_provisioning_profile(
        provisioning_profile_file_path='/tmp/fake.mobileprovision',
        root_bundle_path='/tmp/fake.app/')
    mock_copy.assert_called_with(
        '/tmp/fake.mobileprovision', '/tmp/fake.app/embedded.mobileprovision')

    dossier_codesigning_reader._copy_embedded_provisioning_profile(
        provisioning_profile_file_path='/tmp/fake.mobile',
        root_bundle_path='/tmp/fake.app/')
    mock_copy.assert_called_with(
        '/tmp/fake.mobile', '/tmp/fake.app/Contents/embedded.mobile')

  def test_wait_signing_futures_reraises_exception(self):
    future_with_exception = concurrent.futures.Future()
    future_with_exception.set_exception(RuntimeError)

    future_with_no_exception = concurrent.futures.Future()
    future_with_no_exception.set_result(None)
    futures = [
        dossier_codesigning_reader.SigningFuture(future_with_exception, 'n1'),
        dossier_codesigning_reader.SigningFuture(future_with_no_exception, 'n2')
    ]

    stdout = io.StringIO()
    with self.assertRaises(RuntimeError), contextlib.redirect_stdout(stdout):
      dossier_codesigning_reader._wait_signing_futures(futures)
    self.assertNotRegex(stdout.getvalue(), 'Multiple codesign tasks failed:')

  def test_wait_signing_futures_does_not_raises_exception(self):
    futures = []
    for _ in range(3):
      future = concurrent.futures.Future()
      future.set_result(None)
      futures.append(dossier_codesigning_reader.SigningFuture(future, 'note'))
    dossier_codesigning_reader._wait_signing_futures(futures)

  @mock.patch('concurrent.futures.wait')
  def test_wait_signing_futures_cancel_futures(self, mock_wait):
    mock_future_done = mock.Mock()
    mock_future_exception = mock.Mock()
    mock_future_not_done = mock.Mock()

    mock_future_exception.exception.return_value = EOFError()
    mock_wait.return_value = (
        [mock_future_exception, mock_future_done], [mock_future_not_done])

    futures = [
        dossier_codesigning_reader.SigningFuture(mock_future_exception, 'n1'),
        dossier_codesigning_reader.SigningFuture(mock_future_done, 'n2'),
        dossier_codesigning_reader.SigningFuture(mock_future_not_done, 'n3')
    ]

    stdout = io.StringIO()
    with self.assertRaises(EOFError), contextlib.redirect_stdout(stdout):
      dossier_codesigning_reader._wait_signing_futures(futures)
    self.assertRegex(stdout.getvalue(), r'Codesign task\(s\) failed\:')

    mock_future_not_done.cancel.assert_called()
    mock_future_exception.cancel.assert_not_called()
    mock_future_done.cancel.assert_not_called()

  @mock.patch.object(dossier_codesigning_reader, '_sign_bundle_with_manifest')
  @mock.patch.object(dossier_codesigning_reader, 'read_manifest_from_dossier')
  @mock.patch.object(dossier_codesigning_reader,
                     'extract_zipped_dossier_if_required')
  def test_app_bundle_inputs(
      self,
      mock_extract_dossier,
      mock_read_manifest,
      mock_sign_bundle):
    with tempfile.NamedTemporaryFile() as tmp_fake_codesign, \
        tempfile.NamedTemporaryFile(suffix='.zip') as tmp_dossier_zip, \
            tempfile.TemporaryDirectory(suffix='.app') as tmp_app_bundle:

      dossier_dir = (
          dossier_codesigning_reader.DossierDirectory('/tmp/dossier/', False)
      )

      mock_read_manifest.return_value = _FAKE_MANIFEST
      mock_extract_dossier.return_value = dossier_dir

      required_args = [
          'sign',
          '--codesign', tmp_fake_codesign.name,
          '--dossier', tmp_dossier_zip.name,
          '--keychain', _ADDITIONAL_SIGNING_KEYCHAIN,
          tmp_app_bundle,
      ]

      args = dossier_codesigning_reader.generate_arg_parser().parse_args(
          required_args
      )
      args.func(args)

      # Fully expand this path as it's treated as a user-provided path in this
      # particular case.
      tmp_app_bundle_fullpath = os.path.realpath(
          os.path.expanduser(tmp_app_bundle))

      mock_sign_bundle.assert_called_with(
          tmp_app_bundle_fullpath,
          _FAKE_MANIFEST,
          dossier_dir.path,
          tmp_fake_codesign.name,
          None,
          _ADDITIONAL_SIGNING_KEYCHAIN,
          None,
      )

  @mock.patch.object(dossier_codesigning_reader, '_package_ipa')
  @mock.patch.object(dossier_codesigning_reader, '_sign_bundle_with_manifest')
  @mock.patch.object(dossier_codesigning_reader, '_extract_archive')
  @mock.patch.object(dossier_codesigning_reader, 'read_manifest_from_dossier')
  @mock.patch.object(dossier_codesigning_reader,
                     'extract_zipped_dossier_if_required')
  def test_ipa_inputs(
      self,
      mock_extract_dossier,
      mock_read_manifest,
      mock_extract_archive,
      mock_sign_bundle,
      mock_package):
    with tempfile.NamedTemporaryFile() as tmp_fake_codesign, \
        tempfile.NamedTemporaryFile(suffix='.zip') as tmp_dossier_zip, \
            tempfile.NamedTemporaryFile(suffix='.ipa') as tmp_ipa_archive, \
                tempfile.TemporaryDirectory(suffix='.app') as tmp_app_bundle:

      dossier_dir = (
          dossier_codesigning_reader.DossierDirectory('/tmp/dossier/', False)
      )

      mock_read_manifest.return_value = _FAKE_MANIFEST
      mock_extract_dossier.return_value = dossier_dir
      mock_extract_archive.return_value = tmp_app_bundle

      required_args = [
          'sign',
          '--codesign', tmp_fake_codesign.name,
          '--dossier', tmp_dossier_zip.name,
          '--output_artifact', '/tmp/dossier_output/output.ipa',
          '--keychain', _ADDITIONAL_SIGNING_KEYCHAIN,
          tmp_ipa_archive.name,
      ]

      args = dossier_codesigning_reader.generate_arg_parser().parse_args(
          required_args
      )
      args.func(args)

      mock_sign_bundle.assert_called_with(
          tmp_app_bundle,
          _FAKE_MANIFEST,
          dossier_dir.path,
          tmp_fake_codesign.name,
          None,
          _ADDITIONAL_SIGNING_KEYCHAIN,
          None,
      )
      mock_package.assert_called_once()

  @mock.patch.object(dossier_codesigning_reader, '_package_ipa')
  @mock.patch.object(dossier_codesigning_reader, '_sign_bundle_with_manifest')
  @mock.patch.object(dossier_codesigning_reader, '_extract_archive')
  @mock.patch.object(dossier_codesigning_reader, 'read_manifest_from_dossier')
  @mock.patch.object(tempfile, 'TemporaryDirectory')
  def test_combined_zip_inputs(
      self,
      mock_temp_dir,
      mock_read_manifest,
      mock_extract_archive,
      mock_sign_bundle,
      mock_package):
    with tempfile.NamedTemporaryFile() as tmp_fake_codesign, \
        tempfile.NamedTemporaryFile(suffix='.zip') as tmp_combined_zip, \
            tempfile.TemporaryDirectory(suffix='.app') as tmp_app_bundle:

      # Overriding TemporaryDirectory with a mocked instance that returns a path
      # from the actual method TemporaryDirectory(), so that we can validate its
      # path is being properly passed around, particularly for dossier handling.
      temp_path = tempfile.TemporaryDirectory()

      mock_temp_dir.return_value.__enter__.return_value = temp_path
      mock_read_manifest.return_value = _FAKE_MANIFEST
      mock_extract_archive.return_value = tmp_app_bundle

      required_args = [
          'sign',
          '--codesign', tmp_fake_codesign.name,
          '--output_artifact', '/tmp/dossier_output/output.ipa',
          '--keychain', _ADDITIONAL_SIGNING_KEYCHAIN,
          tmp_combined_zip.name,
      ]

      args = dossier_codesigning_reader.generate_arg_parser().parse_args(
          required_args
      )
      args.func(args)

      mock_sign_bundle.assert_called_with(
          tmp_app_bundle,
          _FAKE_MANIFEST,
          os.path.join(temp_path, 'dossier'),
          tmp_fake_codesign.name,
          None,
          _ADDITIONAL_SIGNING_KEYCHAIN,
          None,
      )
      mock_package.assert_called_once()


  def _test_extract_and_package_flow(
      self, unsigned_archive_path, app_name, app_bundle_subdir, expected_folders
  ):
    working_dir = tempfile.mkdtemp()
    try:
      extracted_input_path = dossier_codesigning_reader._extract_archive(
          working_dir=working_dir,
          app_bundle_subdir=app_bundle_subdir,
          unsigned_archive_path=unsigned_archive_path,
      )
      self.assertEqual(
          os.path.join(working_dir, app_bundle_subdir, 'Payload', app_name),
          extracted_input_path
      )
      output_dir = tempfile.mkdtemp()
      try:
        output_ipa_path = os.path.join(output_dir, 'output.ipa')
        dossier_codesigning_reader._package_ipa(
            working_dir=working_dir,
            app_bundle_subdir=app_bundle_subdir,
            output_ipa=output_ipa_path,
        )
        if not os.path.isfile(output_ipa_path):
          self.fail('output ipa was not created!')
        extracted_output_path = os.path.join(output_dir, 'extracted')
        subprocess.check_call(
            ['ditto', '-x', '-k', output_ipa_path, extracted_output_path])
        for folder in expected_folders:
          if not os.path.exists(os.path.join(extracted_output_path, folder)):
            self.fail(f'"{folder}" not found in output IPA!')
      finally:
        shutil.rmtree(output_dir)
    finally:
      shutil.rmtree(working_dir)


  def test_extract_and_package_flow1(self):
    self._test_extract_and_package_flow(
        _IPA_WORKSPACE_PATH, 'app.app', '', ['Payload'])


  def test_extract_and_package_flow2(self):
    self._test_extract_and_package_flow(
        _IPA_W_WATCHOS_WORKSPACE_PATH, 'app_companion.app',
        '', ['Payload', 'WatchKitSupport2'])


  def test_extract_and_package_flow3(self):
    self._test_extract_and_package_flow(
        _COMBINED_ZIP_W_WATCHOS_WORKSPACE_PATH, 'app_companion.app',
        'bundle', ['Payload', 'WatchKitSupport2'])

if __name__ == '__main__':
  unittest.main()
