# coding=utf-8
# Copyright 2017 The Bazel Authors. All rights reserved.
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

"""Tests for PlistTool."""

import datetime
import io
import json
import os
import re
import tempfile
import unittest

from tools.plisttool import plisttool

# Used as the target name for all tests.
_testing_target = '//plisttool:tests'


def _xml_plist(content):
  """Returns a BytesIO for a plist with the given content.

  This helper function wraps plist XML (key/value pairs) in the necessary XML
  boilerplate for a plist with a root dictionary.

  Args:
    content: The XML content of the plist, which will be inserted into a
        dictionary underneath the root |plist| element.
  Returns:
    A BytesIO object containing the full XML text of the plist.
  """
  xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
         '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
         '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
         '<plist version="1.0">\n'
         '<dict>\n' +
         content + '\n' +
         '</dict>\n'
         '</plist>\n')
  xml_bytes = xml.encode('utf8')
  return io.BytesIO(xml_bytes)


def _plisttool_result(control):
  """Helper function that runs PlistTool with the given control struct.

  This function inserts a BytesIO object as the control's "output" key and
  returns the dictionary containing the result of the tool after parsing it
  from that BytesIO.

  Args:
    control: The control struct to pass to PlistTool. See the module doc for
        the plisttool module for a description of this format.
  Returns:
    The dictionary containing the result of the tool after parsing it from
    the in-memory string file.
  """
  output = io.BytesIO()
  control['output'] = output
  control['target'] = _testing_target

  tool = plisttool.PlistTool(control)
  tool.run()

  return plisttool.plist_from_bytes(output.getvalue())


class PlistToolMainTest(unittest.TestCase):

  def test_main_invocation(self):
    plist_fp = tempfile.NamedTemporaryFile(delete=False)
    self.addCleanup(lambda: os.unlink(plist_fp.name))
    with plist_fp:
      plist = _xml_plist('<key>Foo</key><string>abc</string>')
      plist_fp.write(plist.getvalue())

    json_fp = tempfile.NamedTemporaryFile(mode='wt', delete=False)
    self.addCleanup(lambda: os.unlink(json_fp.name))
    with json_fp:
      outfile = tempfile.NamedTemporaryFile(delete=False)
      self.addCleanup(lambda: os.unlink(outfile.name))
      outfile.close()
      control = {'plists': [plist_fp.name],
                 'target': '//test:target',
                 'output': outfile.name}
      json.dump(control, json_fp)

    # A None/zero return code means success.
    self.assertFalse(plisttool._main(json_fp.name),
                     'plisttool did not successfully run')

    # TODO(b/111687215): Test that the written output is correct.
    with open(outfile.name, 'rb') as fp:
      self.assertIn(b'<?xml', fp.read())


class PlistToolVariableReferenceTest(unittest.TestCase):

  def _assert_result(self, s, expected):
    """Asserts string is the expected variable reference."""
    m = plisttool.VARIABLE_REFERENCE_RE.match(s)
    # Testing that based on the whole string.
    self.assertEqual(m.group(0), s)
    self.assertEqual(plisttool.extract_variable_from_match(m), expected)

  def _assert_invalid(self, s):
    """Asserts string is not a valid variable reference."""
    self._assert_result(s, None)

  def test_valid_parens(self):
    self._assert_result('$(foo)', 'foo')
    self._assert_result('$(FOO12)', 'FOO12')
    self._assert_result('$(PRODUCT_NAME:rfc1034identifier)',
                        'PRODUCT_NAME:rfc1034identifier')

  def test_valid_braces(self):
    self._assert_result('${foo}', 'foo')
    self._assert_result('${FOO12}', 'FOO12')
    self._assert_result('${PRODUCT_NAME:rfc1034identifier}',
                        'PRODUCT_NAME:rfc1034identifier')

  def test_empty_reference(self):
    self._assert_invalid('$()')
    self._assert_invalid('${}')

  def test_mismatched_bracing(self):
    self._assert_invalid('${foo)')
    self._assert_invalid('$(foo}')

  def test_invalid_names(self):
    self._assert_invalid('${no space}')
    self._assert_invalid('${no-hypens}')

  def test_unknown_qualifier(self):
    self._assert_invalid('${foo:mumble}')
    self._assert_invalid('${foo:rfc666dentifier}')

  def test_missing_closer(self):
    # Valid, just missing the closer...
    self._assert_invalid('$(foo')
    self._assert_invalid('$(FOO12')
    self._assert_invalid('$(PRODUCT_NAME:rfc1034identifier')
    self._assert_invalid('${foo')
    self._assert_invalid('${FOO12')
    self._assert_invalid('${PRODUCT_NAME:rfc1034identifier')
    # Invalid and missing the closer...
    self._assert_invalid('${no space')
    self._assert_invalid('${no-hypens')
    self._assert_invalid('${foo:mumble')
    self._assert_invalid('${foo:rfc666dentifier')


class PlistToolVersionStringTest(unittest.TestCase):

  def _assert_valid(self, s):
    self.assertEqual(plisttool.is_valid_version_string(s), True)

  def _assert_invalid(self, s):
    self.assertEqual(plisttool.is_valid_version_string(s), False)

  def test_all_good(self):
    self._assert_valid('1')
    self._assert_valid('1.2')
    self._assert_valid('1.2.3')
    self._assert_valid('1.2.3.4')
    self._assert_valid('1.0')
    self._assert_valid('1.0.0')
    self._assert_valid('1.0.0.0')
    self._assert_valid('1.0.3')
    self._assert_valid('10.11.12')
    self._assert_valid('10.11.12.13')

  def test_non_numbers(self):
    self._assert_invalid('abc')
    self._assert_invalid('abc1')
    self._assert_invalid('1abc')
    self._assert_invalid('1abc.1')
    self._assert_invalid('1.abc')
    self._assert_invalid('1.1abc')
    self._assert_invalid('1.abc.1')
    self._assert_invalid('1.1abc.1')
    self._assert_invalid('abc.1')
    self._assert_invalid('1.abc.2')

  def test_to_many_segments(self):
    self._assert_invalid('1.2.3.4.5')
    self._assert_invalid('1.2.3.4.0')
    self._assert_invalid('1.2.3.4.5.6')

  def test_to_badly_formed(self):
    self._assert_invalid('1.')
    self._assert_invalid('1.2.')
    self._assert_invalid('1.2.3.')
    self._assert_invalid('1.2.3.4.')
    self._assert_invalid('.1')
    self._assert_invalid('.1.2')
    self._assert_invalid('.1.2.3')
    self._assert_invalid('.1.2.3.4')
    self._assert_invalid('1..3')
    self._assert_invalid('1.2..4')

  def test_to_other_punct(self):
    self._assert_invalid('1,2')
    self._assert_invalid('1$2')
    self._assert_invalid('1:2')

  def test_to_long(self):
    self._assert_invalid('123456789.123456789')
    self._assert_invalid('1234.6789.123456789')
    self._assert_invalid('1234.6789.1234.6789')

  def test_all_good_padded(self):
    self._assert_valid('01')
    self._assert_valid('01.1')
    self._assert_valid('01.01')
    self._assert_valid('01.0.1')
    self._assert_valid('01.0.01')
    self._assert_valid('1.00')
    self._assert_valid('1.0.00')
    self._assert_valid('1.001')
    self._assert_valid('1.0.001')

  def test_all_good_with_tracks(self):
    self._assert_valid('1a1')
    self._assert_valid('1.2d12')
    self._assert_valid('1.2.3b7')
    self._assert_valid('1.0fc100')
    self._assert_valid('1.0.0b7')
    self._assert_valid('1.0.3fc1')
    self._assert_valid('10.11.12d123')
    self._assert_valid('1abc1')
    self._assert_valid('1.1abc1')
    self._assert_valid('1.2.3foo4')
    self._assert_valid('1.2.3bar1')
    self._assert_valid('1.2.3baz255')

  def test_invalid_tracks(self):
    self._assert_invalid('1a0')
    self._assert_invalid('1a')
    self._assert_invalid('1.2d')
    self._assert_invalid('1.2d01')
    self._assert_invalid('1.2.3b')
    self._assert_invalid('1.2.3b1234')
    self._assert_invalid('1.0fc')
    self._assert_invalid('1.0fc256')
    self._assert_invalid('1.2.3thisistoolong4')
    self._assert_invalid('1.2.3stilltoolong45')
    self._assert_invalid('1.2.3alsotoolong128')


class PlistToolShortVersionStringTest(unittest.TestCase):

  def _assert_valid(self, s):
    self.assertEqual(plisttool.is_valid_short_version_string(s), True)

  def _assert_invalid(self, s):
    self.assertEqual(plisttool.is_valid_short_version_string(s), False)

  def test_all_good(self):
    self._assert_valid('1')
    self._assert_valid('1.2')
    self._assert_valid('1.2.3')
    self._assert_valid('1.2.3.4')
    self._assert_valid('1.0')
    self._assert_valid('1.0.0')
    self._assert_valid('1.0.0.0')
    self._assert_valid('1.0.3')
    self._assert_valid('10.11.12')
    self._assert_valid('10.11.12.13')

  def test_non_numbers(self):
    self._assert_invalid('abc')
    self._assert_invalid('abc1')
    self._assert_invalid('1abc')
    self._assert_invalid('1abc1')
    self._assert_invalid('1.abc')
    self._assert_invalid('1.1abc')
    self._assert_invalid('1.abc1')
    self._assert_invalid('1.1abc1')
    self._assert_invalid('abc.1')
    self._assert_invalid('1.abc.2')

  def test_to_many_segments(self):
    self._assert_invalid('1.2.3.4.5')
    self._assert_invalid('1.2.3.4.0')
    self._assert_invalid('1.2.3.4.5.6')

  def test_to_badly_formed(self):
    self._assert_invalid('1.')
    self._assert_invalid('1.2.')
    self._assert_invalid('1.2.3.')
    self._assert_invalid('1.2.3.4.')
    self._assert_invalid('.1')
    self._assert_invalid('.1.2')
    self._assert_invalid('.1.2.3')
    self._assert_invalid('.1.2.3.4')
    self._assert_invalid('1..3')
    self._assert_invalid('1.2..4')

  def test_to_other_punct(self):
    self._assert_invalid('1,2')
    self._assert_invalid('1$2')
    self._assert_invalid('1:2')

  def test_to_long(self):
    self._assert_invalid('123456789.123456789')
    self._assert_invalid('1234.6789.123456789')
    self._assert_invalid('1234.6789.1234.6789')

  def test_all_good_padded(self):
    self._assert_valid('01')
    self._assert_valid('01.1')
    self._assert_valid('01.01')
    self._assert_valid('01.0.1')
    self._assert_valid('01.0.01')
    self._assert_valid('1.00')
    self._assert_valid('1.0.00')
    self._assert_valid('1.001')
    self._assert_valid('1.0.001')

  def test_all_good_with_tracks_are_bad(self):
    self._assert_invalid('1a1')
    self._assert_invalid('1.2d12')
    self._assert_invalid('1.2.3b7')
    self._assert_invalid('1.0fc100')
    self._assert_invalid('1.0.0b7')
    self._assert_invalid('1.0.3fc1')
    self._assert_invalid('10.11.12d123')
    self._assert_invalid('1.2.3abcstaging1')


class PlistToolGetWithKeyPath(unittest.TestCase):

  def test_one_level(self):
    d = {'a': 'A', 'b': 2, 3: 'c', 'list': ['x', 'y'], 'dict': {1: 2, 3: 4}}
    self.assertEqual(plisttool.get_with_key_path(d, ['a']), 'A')
    self.assertEqual(plisttool.get_with_key_path(d, ['b']), 2)
    self.assertEqual(plisttool.get_with_key_path(d, [3]), 'c')
    self.assertEqual(plisttool.get_with_key_path(d, ['list']), ['x', 'y'])
    self.assertEqual(plisttool.get_with_key_path(d, ['dict']), {1: 2, 3: 4})

  def test_two_level(self):
    d = {'list': ['x', 'y'], 'dict': {1: 2, 3: 4}}
    self.assertEqual(plisttool.get_with_key_path(d, ['list', 1]), 'y')
    self.assertEqual(plisttool.get_with_key_path(d, ['dict', 3]), 4)

  def test_deep(self):
    d = {1: {'a': ['c', [4, 'e']]}}
    self.assertEqual(plisttool.get_with_key_path(d, [1, 'a', 1, 1]), 'e')

  def test_misses(self):
    d = {'list': ['x', 'y'], 'dict': {1: 2, 3: 4}}
    self.assertIsNone(plisttool.get_with_key_path(d, ['not_found']))
    self.assertIsNone(plisttool.get_with_key_path(d, [99]))
    self.assertIsNone(plisttool.get_with_key_path(d, ['list', 99]))
    self.assertIsNone(plisttool.get_with_key_path(d, ['dict', 'not_found']))
    self.assertIsNone(plisttool.get_with_key_path(d, ['dict', 99]))

  def test_invalids(self):
    d = {'list': ['x', 'y'], 'str': 'foo', 'int': 42}
    self.assertIsNone(plisttool.get_with_key_path(d, ['list', 'not_int']))
    self.assertIsNone(plisttool.get_with_key_path(d, ['str', 'nope']))
    self.assertIsNone(plisttool.get_with_key_path(d, ['str', 99]))
    self.assertIsNone(plisttool.get_with_key_path(d, ['int', 'nope']))
    self.assertIsNone(plisttool.get_with_key_path(d, ['int', 99]))


class PlistToolTest(unittest.TestCase):

  def _assert_plisttool_result(self, control, expected):
    """Asserts that PlistTool's result equals the expected dictionary.

    Args:
      control: The control struct to pass to PlistTool. See the module doc for
          the plisttool module for a description of this format.
      expected: The dictionary that represents the expected result from running
          PlistTool.
    """
    outdict = _plisttool_result(control)
    self.assertEqual(expected, outdict)

  def _assert_pkginfo(self, plist, expected):
    """Asserts that PlistTool generates the expected PkgInfo file contents.

    Args:
      plist: The plist file from which to obtain the PkgInfo values.
      expected: The expected 8-byte string written to the PkgInfo file.
    """
    pkginfo = io.BytesIO()
    control = {
        'plists': [plist],
        'output': io.BytesIO(),
        'target': _testing_target,
        'info_plist_options': {'pkginfo': pkginfo},
    }
    tool = plisttool.PlistTool(control)
    tool.run()
    self.assertEqual(expected, pkginfo.getvalue())

  def test_merge_of_one_file(self):
    plist1 = _xml_plist('<key>Foo</key><string>abc</string>')
    self._assert_plisttool_result({'plists': [plist1]}, {'Foo': 'abc'})

  def test_merge_of_one_dict(self):
    plist1 = {'Foo': 'abc'}
    self._assert_plisttool_result({'plists': [plist1]}, {'Foo': 'abc'})

  def test_merge_of_one_empty_file(self):
    plist1 = _xml_plist('')
    self._assert_plisttool_result({'plists': [plist1]}, {})

  def test_merge_of_one_empty_dict(self):
    plist1 = {}
    self._assert_plisttool_result({'plists': [plist1]}, {})

  def test_merge_of_two_files(self):
    plist1 = _xml_plist('<key>Foo</key><string>abc</string>')
    plist2 = _xml_plist('<key>Bar</key><string>def</string>')
    self._assert_plisttool_result({'plists': [plist1, plist2]}, {
        'Foo': 'abc',
        'Bar': 'def',
    })

  def test_merge_of_file_and_dict(self):
    plist1 = _xml_plist('<key>Foo</key><string>abc</string>')
    plist2 = {'Bar': 'def'}
    self._assert_plisttool_result({'plists': [plist1, plist2]}, {
        'Foo': 'abc',
        'Bar': 'def',
    })

  def test_merge_of_two_dicts(self):
    plist1 = {'Foo': 'abc'}
    plist2 = {'Bar': 'def'}
    self._assert_plisttool_result({'plists': [plist1, plist2]}, {
        'Foo': 'abc',
        'Bar': 'def',
    })

  def test_merge_where_one_file_is_empty(self):
    plist1 = _xml_plist('<key>Foo</key><string>abc</string>')
    plist2 = _xml_plist('')
    self._assert_plisttool_result({'plists': [plist1, plist2]}, {'Foo': 'abc'})

  def test_merge_where_one_dict_is_empty(self):
    plist1 = {'Foo': 'abc'}
    plist2 = {}
    self._assert_plisttool_result({'plists': [plist1, plist2]}, {'Foo': 'abc'})

  def test_merge_where_both_files_are_empty(self):
    plist1 = _xml_plist('')
    plist2 = _xml_plist('')
    self._assert_plisttool_result({'plists': [plist1, plist2]}, {})

  def test_merge_where_both_dicts_are_empty(self):
    plist1 = {}
    plist2 = {}
    self._assert_plisttool_result({'plists': [plist1, plist2]}, {})

  def test_more_complicated_merge(self):
    plist1 = _xml_plist(
        '<key>String1</key><string>abc</string>'
        '<key>Integer1</key><integer>123</integer>'
        '<key>Array1</key><array><string>a</string><string>b</string></array>'
    )
    plist2 = _xml_plist(
        '<key>String2</key><string>def</string>'
        '<key>Integer2</key><integer>987</integer>'
        '<key>Dictionary2</key><dict>'
        '<key>k1</key><string>a</string>'
        '<key>k2</key><string>b</string>'
        '</dict>'
    )
    plist3 = _xml_plist(
        '<key>String3</key><string>ghi</string>'
        '<key>Integer3</key><integer>465</integer>'
        '<key>Bundle</key><string>this.is.${BUNDLE_NAME}.bundle</string>'
    )
    self._assert_plisttool_result({
        'plists': [plist1, plist2, plist3],
        'variable_substitutions': {
            'BUNDLE_NAME': 'my'
        },
    }, {
        'String1': 'abc',
        'Integer1': 123,
        'Array1': ['a', 'b'],
        'String2': 'def',
        'Integer2': 987,
        'Dictionary2': {'k1': 'a', 'k2': 'b'},
        'String3': 'ghi',
        'Integer3': 465,
        'Bundle': 'this.is.my.bundle',
    })

  def test_merge_with_forced_plist_overrides_on_collisions(self):
    plist1 = {'Foo': 'bar'}
    plist2 = {'Foo': 'baz'}
    self._assert_plisttool_result({
        'plists': [plist1],
        'forced_plists': [plist2],
    }, {'Foo': 'baz'})

  def test_merge_with_forced_plists_with_same_key_keeps_last_one(self):
    plist1 = {'Foo': 'bar'}
    plist2 = {'Foo': 'baz'}
    plist3 = {'Foo': 'quux'}
    self._assert_plisttool_result({
        'plists': [plist1],
        'forced_plists': [plist2, plist3],
    }, {'Foo': 'quux'})

  def test_invalid_variable_substitution_name_space(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.INVALID_SUBSTITUTION_VARIABLE_NAME % (
            _testing_target, 'foo bar'))):
      _plisttool_result({
          'plists': [{}],
          'variable_substitutions': {
              'foo bar': 'bad name',
          },
      })

  def test_invalid_variable_substitution_name_hyphen(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.INVALID_SUBSTITUTION_VARIABLE_NAME % (
            _testing_target, 'foo-bar'))):
      _plisttool_result({
          'plists': [{}],
          'variable_substitutions': {
              'foo-bar': 'bad name',
          },
      })

  def test_invalid_variable_substitution_name_qualifier(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.INVALID_SUBSTITUTION_VARIABLE_NAME % (
            _testing_target, 'foo:bar'))):
      _plisttool_result({
          'plists': [{}],
          'variable_substitutions': {
              'foo:bar': 'bad name',
          },
      })

  def test_invalid_variable_substitution_name_rfc_qualifier(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.SUBSTITUTION_VARIABLE_CANT_HAVE_QUALIFIER % (
            _testing_target, 'foo:rfc1034identifier'))):
      _plisttool_result({
          'plists': [{}],
          'variable_substitutions': {
              'foo:rfc1034identifier': 'bad name',
          },
      })

  def test_both_types_variable_substitutions(self):
    plist1 = _xml_plist(
        '<key>FooBraces</key><string>${TARGET_NAME}</string>'
        '<key>BarBraces</key><string>${PRODUCT_NAME}</string>'
        '<key>FooParens</key><string>$(TARGET_NAME)</string>'
        '<key>BarParens</key><string>$(PRODUCT_NAME)</string>'
    )
    outdict = _plisttool_result({
        'plists': [plist1],
        'variable_substitutions': {
            'PRODUCT_NAME': 'MyApp',
            'TARGET_NAME': 'MyApp',
        },
    })
    self.assertEqual('MyApp', outdict.get('FooBraces'))
    self.assertEqual('MyApp', outdict.get('BarBraces'))
    self.assertEqual('MyApp', outdict.get('FooParens'))
    self.assertEqual('MyApp', outdict.get('BarParens'))

  def test_rfc1034_conversion(self):
    plist1 = _xml_plist(
        '<key>Foo</key><string>${PRODUCT_NAME:rfc1034identifier}</string>'
    )
    outdict = _plisttool_result({
        'plists': [plist1],
        'variable_substitutions': {
            'PRODUCT_NAME': 'foo_bar?baz'
        },
    })
    self.assertEqual('foo-bar-baz', outdict.get('Foo'))

  def test_raw_substitutions(self):
    plist1 = _xml_plist(
        '<key>One</key><string>RAW1</string>'
        '<key>Two</key><string>RAW2</string>'
        '<key>SpaceOneSpaceTwoSpace</key><string> RAW1 RAW2 </string>'
        '<key>OneTwoOneTwo</key><string>RAW1RAW2RAW1RAW2</string>'
        '<key>XOneX</key><string>XRAW1X</string>'
        '<key>XTwoX</key><string>XRAW2X</string>'
    )
    outdict = _plisttool_result({
        'plists': [plist1],
        'raw_substitutions': {
            'RAW1': 'a',
            'RAW2': 'b',
        },
    })
    self.assertEqual('a', outdict.get('One'))
    self.assertEqual('b', outdict.get('Two'))
    self.assertEqual(' a b ', outdict.get('SpaceOneSpaceTwoSpace'))
    self.assertEqual('abab', outdict.get('OneTwoOneTwo'))
    self.assertEqual('XaX', outdict.get('XOneX'))
    self.assertEqual('XbX', outdict.get('XTwoX'))

  def test_raw_substitutions_overlap_raw(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.OVERLAP_IN_SUBSTITUTION_KEYS % (
            _testing_target, 'mum', 'mumble'))):
      _plisttool_result({
          'plists': [{}],
          'raw_substitutions': {
              'mumble': 'value1',
              'mum': 'value2',
          },
      })

  def test_raw_substitutions_overlap_variable(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.OVERLAP_IN_SUBSTITUTION_KEYS % (
            _testing_target, '$(mumble)', 'mum'))):
      _plisttool_result({
          'plists': [{}],
          'variable_substitutions': {
              'mumble': 'value1',
          },
          'raw_substitutions': {
              'mum': 'value2',
          },
      })

  def test_raw_substitutions_key_in_value(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.RAW_SUBSTITUTION_KEY_IN_VALUE % (
            _testing_target, 'value', '1value2', 'mumble'))):
      _plisttool_result({
          'plists': [{}],
          'raw_substitutions': {
              'mumble': '1value2',
              'value': 'spam',
          },
      })

  def test_nonexistant_variable_substitution(self):
    plist1 = {
        'FooBraces': 'A-${NOT_A_VARIABLE}-B'
    }
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.UNKNOWN_SUBSTITUTATION_REFERENCE_MSG % (
            _testing_target, '${NOT_A_VARIABLE}', 'FooBraces',
            'A-${NOT_A_VARIABLE}-B'))):
      _plisttool_result({'plists': [plist1]})

    plist2 = {
        'FooParens': '$(NOT_A_VARIABLE)'
    }
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.UNKNOWN_SUBSTITUTATION_REFERENCE_MSG % (
            _testing_target, '$(NOT_A_VARIABLE)', 'FooParens',
            '$(NOT_A_VARIABLE)'))):
      _plisttool_result({'plists': [plist2]})

    # Nested dict, will include the keypath.
    plist3 = {
        'Key1': {
            'Key2': 'foo.bar.$(PRODUCT_NAME:rfc1034identifier)'
        }
    }
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.UNKNOWN_SUBSTITUTATION_REFERENCE_MSG % (
            _testing_target, '$(PRODUCT_NAME:rfc1034identifier)',
            'Key1:Key2', 'foo.bar.$(PRODUCT_NAME:rfc1034identifier)'))):
      _plisttool_result({'plists': [plist3]})

    # Array, will include the keypath.
    plist3 = {
        'Key': [
            'this one is ok',
            'foo.bar.$(PRODUCT_NAME:rfc1034identifier)'
        ]
    }
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.UNKNOWN_SUBSTITUTATION_REFERENCE_MSG % (
            _testing_target, '$(PRODUCT_NAME:rfc1034identifier)',
            'Key[1]', 'foo.bar.$(PRODUCT_NAME:rfc1034identifier)'))):
      _plisttool_result({'plists': [plist3]})

  def test_variable_substitution_in_key(self):
    plist1 = {
        'Foo${Braces}': 'Bar'
    }
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.UNSUPPORTED_SUBSTITUTATION_REFERENCE_IN_KEY_MSG % (
            _testing_target, '${Braces}', 'Foo${Braces}'))):
      _plisttool_result({'plists': [plist1]})

    plist2 = {
        'Foo$(Parens)': 'Bar'
    }
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.UNSUPPORTED_SUBSTITUTATION_REFERENCE_IN_KEY_MSG % (
            _testing_target, '$(Parens)', 'Foo$(Parens)'))):
      _plisttool_result({'plists': [plist2]})

    # Nested dict, will include the keypath.
    plist3 = {
        'Key1': {
            'Key${2}': 'value'
        }
    }
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.UNSUPPORTED_SUBSTITUTATION_REFERENCE_IN_KEY_MSG % (
            _testing_target, '${2}', 'Key1:Key${2}'))):
      _plisttool_result({'plists': [plist3]})

    # Array (of dict), will include the keypath.
    plist3 = {
        'Key1': [
            {'Foo': 'Bar'},
            {'Key${2}': 'value'},
        ]
    }
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.UNSUPPORTED_SUBSTITUTATION_REFERENCE_IN_KEY_MSG % (
            _testing_target, '${2}', 'Key1[1]:Key${2}'))):
      _plisttool_result({'plists': [plist3]})

  def test_invalid_variable_substitution(self):
    plist1 = {
        'Foo': 'foo.${INVALID_REFERENCE).bar'
    }
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.INVALID_SUBSTITUTATION_REFERENCE_MSG % (
            _testing_target, '${INVALID_REFERENCE)', 'Foo',
            'foo.${INVALID_REFERENCE).bar'))):
      _plisttool_result({'plists': [plist1]})

  def test_multiple_substitutions(self):
    plist1 = _xml_plist(
        '<key>Foo</key>'
        '<string>${PRODUCT_NAME}--A_RAW_SUB--${EXECUTABLE_NAME}</string>'
    )
    outdict = _plisttool_result({
        'plists': [plist1],
        'variable_substitutions': {
            'EXECUTABLE_NAME': 'MyExe',
            'PRODUCT_NAME': 'MyApp',
        },
        'raw_substitutions': {
            'A_RAW_SUB': 'MyBundle',
        },
    })
    self.assertEqual('MyApp--MyBundle--MyExe', outdict.get('Foo'))

  def test_recursive_substitutions(self):
    plist1 = _xml_plist(
        '<key>Foo</key>'
        '<dict>'
        '  <key>Foo1</key>'
        '  <string>${BUNDLE_NAME}</string>'
        '  <key>Foo2</key>'
        '  <string>RAW_NAME</string>'
        '  <key>Foo3</key>'
        '  <array>'
        '    <string>${BUNDLE_NAME}</string>'
        '    <string>RAW_NAME</string>'
        '  </array>'
        '</dict>'
        '<key>Bar</key>'
        '<array>'
        '  <string>${BUNDLE_NAME}</string>'
        '  <string>RAW_NAME</string>'
        '  <dict>'
        '    <key>Baz</key>'
        '    <string>${BUNDLE_NAME}</string>'
        '    <key>Baz2</key>'
        '    <string>RAW_NAME</string>'
        '  </dict>'
        '</array>'
    )
    outdict = _plisttool_result({
        'plists': [plist1],
        'variable_substitutions': {
            'BUNDLE_NAME': 'MyBundle',
        },
        'raw_substitutions': {
            'RAW_NAME': 'MyValue',
        },
    })
    self.assertEqual('MyBundle', outdict.get('Foo').get('Foo1'))
    self.assertEqual('MyValue', outdict.get('Foo').get('Foo2'))
    self.assertEqual('MyBundle', outdict.get('Foo').get('Foo3')[0])
    self.assertEqual('MyValue', outdict.get('Foo').get('Foo3')[1])
    self.assertEqual('MyBundle', outdict.get('Bar')[0])
    self.assertEqual('MyValue', outdict.get('Bar')[1])
    self.assertEqual('MyBundle', outdict.get('Bar')[2].get('Baz'))
    self.assertEqual('MyValue', outdict.get('Bar')[2].get('Baz2'))

  def test_keys_with_same_values_do_not_raise_error(self):
    plist1 = _xml_plist('<key>Foo</key><string>Bar</string>')
    plist2 = _xml_plist('<key>Foo</key><string>Bar</string>')
    self._assert_plisttool_result({'plists': [plist1, plist2]}, {'Foo': 'Bar'})

  def test_conflicting_keys_raises_error(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.CONFLICTING_KEYS_MSG % (
            _testing_target, 'Foo', 'Baz', 'Bar'))):
      plist1 = _xml_plist('<key>Foo</key><string>Bar</string>')
      plist2 = _xml_plist('<key>Foo</key><string>Baz</string>')
      _plisttool_result({'plists': [plist1, plist2]})

  def test_pkginfo_with_valid_values(self):
    self._assert_pkginfo({
        'CFBundlePackageType': 'APPL',
        'CFBundleSignature': '1234',
    }, b'APPL1234')

  def test_pkginfo_with_missing_package_type(self):
    self._assert_pkginfo({
        'CFBundleSignature': '1234',
    }, b'????1234')

  def test_pkginfo_with_missing_signature(self):
    self._assert_pkginfo({
        'CFBundlePackageType': 'APPL',
    }, b'APPL????')

  def test_pkginfo_with_missing_package_type_and_signature(self):
    self._assert_pkginfo({}, b'????????')

  def test_pkginfo_with_values_too_long(self):
    self._assert_pkginfo({
        'CFBundlePackageType': 'APPLE',
        'CFBundleSignature': '1234',
    }, b'????1234')

  def test_pkginfo_with_valid_values_too_short(self):
    self._assert_pkginfo({
        'CFBundlePackageType': 'APPL',
        'CFBundleSignature': '123',
    }, b'APPL????')

  def test_pkginfo_with_values_encodable_in_mac_roman(self):
    self._assert_pkginfo({
        'CFBundlePackageType': u'ÄPPL',
        'CFBundleSignature': '1234',
    }, b'\x80PPL1234')

  def test_pkginfo_with_values_not_encodable_in_mac_roman(self):
    self._assert_pkginfo({
        'CFBundlePackageType': u'😎',
        'CFBundleSignature': '1234',
    }, b'????1234')

  def test_child_plist_that_matches_parent_does_not_raise(self):
    parent = _xml_plist(
        '<key>CFBundleIdentifier</key><string>foo.bar</string>')
    child = _xml_plist(
        '<key>CFBundleIdentifier</key><string>foo.bar.baz</string>')
    children = {'//fake:label': child}
    _plisttool_result({
        'plists': [parent],
        'info_plist_options': {
            'child_plists': children,
        },
    })

  def test_child_plist_with_incorrect_bundle_id_raises(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.CHILD_BUNDLE_ID_MISMATCH_MSG % (
            _testing_target, '//fake:label', 'foo.bar.', 'foo.baz'))):
      parent = _xml_plist(
          '<key>CFBundleIdentifier</key><string>foo.bar</string>')
      child = _xml_plist(
          '<key>CFBundleIdentifier</key><string>foo.baz</string>')
      children = {'//fake:label': child}
      _plisttool_result({
          'plists': [parent],
          'info_plist_options': {
              'child_plists': children,
          },
      })

  def test_child_plist_with_incorrect_bundle_version_raises(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.CHILD_BUNDLE_VERSION_MISMATCH_MSG % (
            _testing_target, 'CFBundleVersion', '//fake:label',
            '1.2.3', '1.2.4'))):
      parent = _xml_plist(
          '<key>CFBundleIdentifier</key><string>foo.bar</string>'
          '<key>CFBundleVersion</key><string>1.2.3</string>')
      child = _xml_plist(
          '<key>CFBundleIdentifier</key><string>foo.bar.baz</string>'
          '<key>CFBundleVersion</key><string>1.2.4</string>')
      children = {'//fake:label': child}
      _plisttool_result({
          'plists': [parent],
          'info_plist_options': {
              'child_plists': children,
          },
      })

  def test_child_plist_with_incorrect_bundle_short_version_raises(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.CHILD_BUNDLE_VERSION_MISMATCH_MSG % (
            _testing_target, 'CFBundleShortVersionString', '//fake:label',
            '1.2.3', '1.2.4'))):
      parent = _xml_plist(
          '<key>CFBundleIdentifier</key><string>foo.bar</string>'
          '<key>CFBundleShortVersionString</key><string>1.2.3</string>')
      child = _xml_plist(
          '<key>CFBundleIdentifier</key><string>foo.bar.baz</string>'
          '<key>CFBundleShortVersionString</key><string>1.2.4</string>')
      children = {'//fake:label': child}
      _plisttool_result({
          'plists': [parent],
          'info_plist_options': {
              'child_plists': children,
          },
      })

  def test_child_plist_missing_required_child(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.REQUIRED_CHILD_MISSING_MSG % (
            _testing_target, '//unknown:label'))):
      parent = _xml_plist(
          '<key>CFBundleIdentifier</key><string>foo.bar</string>')
      child = _xml_plist(
          '<key>CFBundleIdentifier</key><string>foo.bar.baz</string>')
      children = {'//fake:label': child}
      _plisttool_result({
          'plists': [parent],
          'info_plist_options': {
              'child_plists': children,
              'child_plist_required_values': {
                  '//unknown:label': [['foo', 'bar']],
              }
          },
      })

  def test_child_plist_required_invalid_format_not_list(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.REQUIRED_CHILD_NOT_PAIR % (
            _testing_target, '//fake:label', 'not_right'))):
      parent = _xml_plist(
          '<key>CFBundleIdentifier</key><string>foo.bar</string>')
      child = _xml_plist(
          '<key>CFBundleIdentifier</key><string>foo.bar.baz</string>')
      children = {'//fake:label': child}
      _plisttool_result({
          'plists': [parent],
          'info_plist_options': {
              'child_plists': children,
              'child_plist_required_values': {
                  '//fake:label': ['not_right'],
              }
          },
      })

  def test_child_plist_required_invalid_format_not_pair(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.REQUIRED_CHILD_NOT_PAIR % (
            _testing_target, '//fake:label', ['not_right']))):
      parent = _xml_plist(
          '<key>CFBundleIdentifier</key><string>foo.bar</string>')
      child = _xml_plist(
          '<key>CFBundleIdentifier</key><string>foo.bar.baz</string>')
      children = {'//fake:label': child}
      _plisttool_result({
          'plists': [parent],
          'info_plist_options': {
              'child_plists': children,
              'child_plist_required_values': {
                  '//fake:label': [['not_right']],
              }
          },
      })

  def test_child_plist_required_missing_keypath(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.REQUIRED_CHILD_KEYPATH_NOT_FOUND % (
            _testing_target, '//fake:label', 'not-there', 'blah'))):
      parent = _xml_plist(
          '<key>CFBundleIdentifier</key><string>foo.bar</string>')
      child = _xml_plist(
          '<key>CFBundleIdentifier</key><string>foo.bar.baz</string>')
      children = {'//fake:label': child}
      _plisttool_result({
          'plists': [parent],
          'info_plist_options': {
              'child_plists': children,
              'child_plist_required_values': {
                  '//fake:label': [
                      # This will be found and pass.
                      [['CFBundleIdentifier'], 'foo.bar.baz'],
                      # This will raise.
                      [['not-there'], 'blah'],
                  ],
              }
          },
      })

  def test_child_plist_required_not_matching(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.REQUIRED_CHILD_KEYPATH_NOT_MATCHING % (
            _testing_target, '//fake:label', 'CFBundleIdentifier',
            'foo.bar.baz.not', 'foo.bar.baz'))):
      parent = _xml_plist(
          '<key>CFBundleIdentifier</key><string>foo.bar</string>')
      child = _xml_plist(
          '<key>CFBundleIdentifier</key><string>foo.bar.baz</string>')
      children = {'//fake:label': child}
      _plisttool_result({
          'plists': [parent],
          'info_plist_options': {
              'child_plists': children,
              'child_plist_required_values': {
                  '//fake:label': [[['CFBundleIdentifier'], 'foo.bar.baz.not']],
              }
          },
      })

  def test_unknown_control_keys_raise(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.UNKNOWN_CONTROL_KEYS_MSG % (
            _testing_target, 'unknown'))):
      plist = {'Foo': 'bar'}
      _plisttool_result({
          'plists': [plist],
          'unknown': True,
      })

  def test_unknown_info_plist_options_keys_raise(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.UNKNOWN_TASK_OPTIONS_KEYS_MSG % (
            _testing_target, 'info_plist_options', 'mumble'))):
      plist = {'Foo': 'bar'}
      children = {'//fake:label': {'foo': 'bar'}}
      _plisttool_result({
          'plists': [plist],
          'info_plist_options': {
              'child_plists': children,
              'mumble': 'something',
          },
      })

  def test_missing_version(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.MISSING_KEY_MSG % (
            _testing_target, 'CFBundleVersion'))):
      plist = {'CFBundleShortVersionString': '1.0'}
      _plisttool_result({
          'plists': [plist],
          'info_plist_options': {
              'version_keys_required': True,
          },
      })

  def test_extensionkit_attributes(self):
    plist = {
        'EXAppExtensionAttributes': {
            'EXExtensionPointIdentifier': 'com.apple.generic-extension',
        }
    }
    self._assert_plisttool_result({
        'plists': [plist],
        'info_plist_options': {
            'extensionkit_keys_required': True,
        },
    }, plist)

  def test_unexpected_nsextension_attributes(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.UNEXPECTED_KEY_MSG % (
            'NSExtension', _testing_target))):
      plist = {
          'NSExtension': {},
          'EXAppExtensionAttributes': {
              'EXExtensionPointIdentifier': 'com.apple.generic-extension',
          },
      }
      _plisttool_result({
          'plists': [plist],
          'info_plist_options': {
              'extensionkit_keys_required': True,
          },
      })

  def test_missing_app_extension_attributes(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.MISSING_KEY_MSG % (
            _testing_target, 'EXAppExtensionAttributes'))):
      plist = {}
      _plisttool_result({
          'plists': [plist],
          'info_plist_options': {
              'extensionkit_keys_required': True,
          },
      })

  def test_missing_app_extension_point_identifier(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.MISSING_KEY_MSG % (
            _testing_target, 'EXExtensionPointIdentifier'))):
      plist = {
          'EXAppExtensionAttributes': {},
      }
      _plisttool_result({
          'plists': [plist],
          'info_plist_options': {
              'extensionkit_keys_required': True,
          },
      })

  def test_missing_short_version(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.MISSING_KEY_MSG % (
            _testing_target, 'CFBundleShortVersionString'))):
      plist = {'CFBundleVersion': '1.0'}
      _plisttool_result({
          'plists': [plist],
          'info_plist_options': {
              'version_keys_required': True,
          },
      })

  def test_empty_version(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.MISSING_KEY_MSG % (
            _testing_target, 'CFBundleVersion'))):
      plist = {
          'CFBundleShortVersionString': '1.0',
          'CFBundleVersion': '',
      }
      _plisttool_result({
          'plists': [plist],
          'info_plist_options': {
              'version_keys_required': True,
          },
      })

  def test_empty_short_version(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.MISSING_KEY_MSG % (
            _testing_target, 'CFBundleShortVersionString'))):
      plist = {
          'CFBundleShortVersionString': '',
          'CFBundleVersion': '1.0',
      }
      _plisttool_result({
          'plists': [plist],
          'info_plist_options': {
              'version_keys_required': True,
          },
      })

  def test_invalid_version(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.INVALID_VERSION_KEY_VALUE_MSG % (
            _testing_target, 'CFBundleVersion', '1foo'))):
      plist = {
          'CFBundleVersion': '1foo',
      }
      _plisttool_result({
          'plists': [plist],
          'info_plist_options': {}  # presence triggers checking
      })

  def test_invalid_short_version(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.INVALID_VERSION_KEY_VALUE_MSG % (
            _testing_target, 'CFBundleShortVersionString', '1foo'))):
      plist = {
          'CFBundleShortVersionString': '1foo',
      }
      _plisttool_result({
          'plists': [plist],
          'info_plist_options': {}  # presence triggers checking
      })

  def test_versions_not_checked_without_options(self):
    plist = {
        'CFBundleShortVersionString': '1foo',
        'CFBundleVersion': '1foo',
    }
    # Even though they numbers are invalid, the plist comes back fine because
    # there was no info_plist_options to trigger validation.
    self._assert_plisttool_result(
        {'plists': [plist]},
        plist
    )

  def test_entitlements_options_var_subs(self):
    plist1 = {'Foo': '$(AppIdentifierPrefix)'}
    self._assert_plisttool_result({
        'plists': [plist1],
        'entitlements_options': {
            'bundle_id': 'my.bundle.id',
            'profile_metadata_file': {
                'ApplicationIdentifierPrefix': ['abc123'],
                'Version': 1,
            },
        },
    }, {'Foo': 'abc123.'})

  def test_entitlements_options_raw_subs(self):
    plist1 = {'Bar': 'abc123.*'}
    self._assert_plisttool_result({
        'plists': [plist1],
        'entitlements_options': {
            'bundle_id': 'my.bundle.id',
            'profile_metadata_file': {
                'ApplicationIdentifierPrefix': ['abc123'],
                'Version': 1,
            },
        },
    }, {'Bar': 'abc123.my.bundle.id'})

  def test_entitlements_no_profile_for_app_id_prefix(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(
            ' '.join([
                plisttool.UNKNOWN_SUBSTITUTATION_REFERENCE_MSG % (
                    _testing_target,
                    '${AppIdentifierPrefix}',
                    'Foo',
                    '${AppIdentifierPrefix}.my.bundle.id'),
                plisttool.UNKNOWN_SUBSTITUTION_ADDITION_APPIDENTIFIERPREFIX_MSG
            ]))):
      _plisttool_result({
          'plists': [{'Foo': '${AppIdentifierPrefix}.my.bundle.id'}],
          'entitlements_options': {
              'bundle_id': 'my.bundle.id',
          },
      })

  def test_entitlements_no_profile_for_app_id_prefix_rfc_reference(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(
            ' '.join([
                plisttool.UNKNOWN_SUBSTITUTATION_REFERENCE_MSG % (
                    _testing_target,
                    '$(AppIdentifierPrefix:rfc1034identifier)',
                    'Foo',
                    '$(AppIdentifierPrefix:rfc1034identifier).my.bundle.id'),
                plisttool.UNKNOWN_SUBSTITUTION_ADDITION_APPIDENTIFIERPREFIX_MSG
            ]))):
      _plisttool_result({
          'plists': [{
              'Foo': '$(AppIdentifierPrefix:rfc1034identifier).my.bundle.id'
          }],
          'entitlements_options': {
              'bundle_id': 'my.bundle.id',
          },
      })

  def test_entitlements_bundle_id_match(self):
    # This is really looking for the lack of an error being raised.
    plist1 = {'application-identifier': 'QWERTY.my.bundle.id'}
    self._assert_plisttool_result({
        'plists': [plist1],
        'entitlements_options': {
            'bundle_id': 'my.bundle.id',
        },
    }, plist1)

  def test_entitlements_bundle_id_wildcard_match(self):
    # This is really looking for the lack of an error being raised.
    plist1 = {'application-identifier': 'QWERTY.my.*'}
    self._assert_plisttool_result({
        'plists': [plist1],
        'entitlements_options': {
            'bundle_id': 'my.bundle.id',
        },
    }, plist1)

  def test_entitlements_bundle_id_mismatch(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.ENTITLEMENTS_BUNDLE_ID_MISMATCH % (
            _testing_target, 'my.bundle.id', 'other.bundle.id'))):
      _plisttool_result({
          'plists': [{'application-identifier': 'QWERTY.other.bundle.id'}],
          'entitlements_options': {
              'bundle_id': 'my.bundle.id',
          },
      })

  def test_entitlements_bundle_id_wildcard_mismatch(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.ENTITLEMENTS_BUNDLE_ID_MISMATCH % (
            _testing_target, 'my.bundle.id', 'other.*'))):
      _plisttool_result({
          'plists': [{'application-identifier': 'QWERTY.other.*'}],
          'entitlements_options': {
              'bundle_id': 'my.bundle.id',
          },
      })

  def test_entitlements_profile_not_expired(self):
    # This is really looking for the lack of an error being raised.
    plist1 = {'foo': 'bar'}
    self._assert_plisttool_result({
        'plists': [plist1],
        'entitlements_options': {
            'profile_metadata_file': {
                'ExpirationDate': datetime.datetime.max,
                'Version': 1,
            },
        },
    }, plist1)

  def test_entitlements_profile_expired(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.ENTITLEMENTS_PROFILE_HAS_EXPIRED % (
            _testing_target, '0001-01-01T00:00:00'))):
      _plisttool_result({
          'plists': [{
              'foo': 'bar'
          }],
          'entitlements_options': {
              'profile_metadata_file': {
                  'ExpirationDate': datetime.datetime.min,
                  'Version': 1,
              },
          },
      })

  def test_entitlements_profile_team_mismatch(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.ENTITLEMENTS_TEAM_ID_PROFILE_MISMATCH % (
            _testing_target, 'QWERTY', 'TeamIdentifier', "['ASDFGH']"))):
      _plisttool_result({
          'plists': [{'com.apple.developer.team-identifier': 'QWERTY'}],
          'entitlements_options': {
              'profile_metadata_file': {
                  'ApplicationIdentifierPrefix': ['QWERTY'],
                  'TeamIdentifier': ['ASDFGH'],
                  'Version': 1,
              },
          },
      })

  def test_entitlements_profile_teams_match(self):
    # This is really looking for the lack of an error being raised.
    plist1 = {'com.apple.developer.team-identifier': 'QWERTY'}
    self._assert_plisttool_result({
        'plists': [plist1],
        'entitlements_options': {
            'bundle_id': 'my.bundle.id',
            'profile_metadata_file': {
                'ApplicationIdentifierPrefix': ['ASDFGH', 'QWERTY'],
                'TeamIdentifier': ['ASDFGH', 'QWERTY'],
                'Version': 1,
            },
        },
    }, plist1)

  def test_entitlements_app_id_match(self):
    # This is really looking for the lack of an error being raised.
    plist1 = {'application-identifier': 'QWERTY.my.bundle.id'}
    self._assert_plisttool_result({
        'plists': [plist1],
        'entitlements_options': {
            'bundle_id': 'my.bundle.id',
            'profile_metadata_file': {
                'Entitlements': {
                    'application-identifier': 'QWERTY.my.bundle.id',
                },
                'Version': 1,
            },
        },
    }, plist1)

  def test_entitlements_app_id_wildcard_match_from_profile_metadata(self):
    # This is really looking for the lack of an error being raised.
    plist1 = {'application-identifier': 'QWERTY.my.bundle.id'}
    self._assert_plisttool_result({
        'plists': [plist1],
        'entitlements_options': {
            'bundle_id': 'my.bundle.id',
            'profile_metadata_file': {
                'Entitlements': {
                    'application-identifier': 'QWERTY.*',
                },
                'Version': 1,
            },
        },
    }, plist1)

  def test_entitlements_app_id_mismatch(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.ENTITLEMENTS_APP_ID_PROFILE_MISMATCH % (
            _testing_target, 'QWERTY.my.bundle.id', 'ASDFGH.my.bundle.id'))):
      _plisttool_result({
          'plists': [{'application-identifier': 'QWERTY.my.bundle.id'}],
          'entitlements_options': {
              'bundle_id': 'my.bundle.id',
              'profile_metadata_file': {
                  'Entitlements': {
                      'application-identifier': 'ASDFGH.my.bundle.id'
                  },
                  'Version': 1,
              },
          },
      })

  def test_entitlements_app_id_mismatch_wildcard(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.ENTITLEMENTS_APP_ID_PROFILE_MISMATCH % (
            _testing_target, 'QWERTY.my.bundle.id', 'ASDFGH.*'))):
      _plisttool_result({
          'plists': [{'application-identifier': 'QWERTY.my.bundle.id'}],
          'entitlements_options': {
              'bundle_id': 'my.bundle.id',
              'profile_metadata_file': {
                  'Entitlements': {
                      'application-identifier': 'ASDFGH.*',
                  },
                  'Version': 1,
              },
          },
      })

  # The edge case in EntitlementsTask._does_id_match()
  def test_entitlements_app_id_wildcard_match_from_plist(self):
    # This is really looking for the lack of an error being raised.
    plist1 = {'application-identifier': 'QWERTY.*'}
    self._assert_plisttool_result({
        'plists': [plist1],
        'entitlements_options': {
            'bundle_id': 'my.bundle.id',
            'profile_metadata_file': {
                'Entitlements': {
                    'application-identifier': 'QWERTY.my.bundle.id',
                },
                'Version': 1,
            },
        },
    }, plist1)

  # The edge case in EntitlementsTask._does_id_match()
  def test_entitlements_app_id_wildcard_match_wildcard(self):
    # This is really looking for the lack of an error being raised.
    plist1 = {'application-identifier': 'QWERTY.my.*'}
    self._assert_plisttool_result({
        'plists': [plist1],
        'entitlements_options': {
            'bundle_id': 'my.bundle.id',
            'profile_metadata_file': {
                'Entitlements': {
                    'application-identifier': 'QWERTY.*',
                },
                'Version': 1,
            },
        },
    }, plist1)

  # The edge case in EntitlementsTask._does_id_match()
  def test_entitlements_app_id_wildcard_mismatch(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.ENTITLEMENTS_APP_ID_PROFILE_MISMATCH % (
            _testing_target, 'QWERTY.*', 'ASDFGH.*'))):
      _plisttool_result({
          'plists': [{'application-identifier': 'QWERTY.*'}],
          'entitlements_options': {
              'bundle_id': 'my.bundle.id',
              'profile_metadata_file': {
                  'Entitlements': {
                      'application-identifier': 'ASDFGH.*',
                  },
                  'Version': 1,
              },
          },
      })

  # The edge case in EntitlementsTask._does_id_match()
  def test_entitlements_app_id_wildcard_mismatch_wildcard(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.ENTITLEMENTS_APP_ID_PROFILE_MISMATCH % (
            _testing_target, 'QWERTY.*', 'ASDFGH.my.bundle.id'))):
      _plisttool_result({
          'plists': [{'application-identifier': 'QWERTY.*'}],
          'entitlements_options': {
              'bundle_id': 'my.bundle.id',
              'profile_metadata_file': {
                  'Entitlements': {
                      'application-identifier': 'ASDFGH.my.bundle.id',
                  },
                  'Version': 1,
              },
          },
      })

  def test_entitlements_keychain_match(self):
    # This is really looking for the lack of an error being raised.
    plist1 = {'keychain-access-groups': ['QWERTY.my.bundle.id']}
    self._assert_plisttool_result({
        'plists': [plist1],
        'entitlements_options': {
            'bundle_id': 'my.bundle.id',
            'profile_metadata_file': {
                'Entitlements': {
                    'keychain-access-groups': ['QWERTY.my.bundle.id'],
                },
                'Version': 1,
            },
        },
    }, plist1)

  def test_entitlements_keychain_match_wildcard(self):
    # This is really looking for the lack of an error being raised.
    plist1 = {'keychain-access-groups': ['QWERTY.my.bundle.id']}
    self._assert_plisttool_result({
        'plists': [plist1],
        'entitlements_options': {
            'bundle_id': 'my.bundle.id',
            'profile_metadata_file': {
                'Entitlements': {
                    'keychain-access-groups': ['QWERTY.*'],
                },
                'Version': 1,
            },
        },
    }, plist1)

  def test_entitlements_no_keychain_requested(self):
    # This is really looking for the lack of an error being raised.
    plist1 = {'application-identifier': 'QWERTY.my.bundle.id'}
    self._assert_plisttool_result({
        'plists': [plist1],
        'entitlements_options': {
            'bundle_id': 'my.bundle.id',
            'profile_metadata_file': {
                'Entitlements': {
                    'application-identifier': 'QWERTY.*',
                    'keychain-access-groups': ['ASDFGH.*'],
                },
                'Version': 1,
            },
        },
    }, plist1)

  def test_entitlements_keychain_not_allowed(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.ENTITLEMENTS_HAS_GROUP_PROFILE_DOES_NOT % (
            _testing_target, 'keychain-access-groups'))):
      _plisttool_result({
          'plists': [{'keychain-access-groups': ['QWERTY.my.bundle.id']}],
          'entitlements_options': {
              'bundle_id': 'my.bundle.id',
              'profile_metadata_file': {
                  'Entitlements': {
                      'application-identifier': 'QWERTY.*',
                      # No 'keychain-access-groups'
                  },
                  'Version': 1,
              },
          },
      })

  def test_entitlements_keychain_entitlements_wildcard_not_allowed(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.ENTITLEMENTS_VALUE_HAS_WILDCARD % (
            _testing_target, 'keychain-access-groups', 'QWERTY.*'))):
      _plisttool_result({
          'plists': [{'keychain-access-groups': ['QWERTY.*']}],
          'entitlements_options': {
              'bundle_id': 'my.bundle.id',
              'profile_metadata_file': {
                  'Entitlements': {
                      'keychain-access-groups': ['QWERTY.*'],
                  },
                  'Version': 1,
              },
          },
      })

  def test_entitlements_keychain_mismatch(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.ENTITLEMENTS_HAS_GROUP_ENTRY_PROFILE_DOES_NOT % (
            _testing_target, 'keychain-access-groups', 'QWERTY.my.bundle.id',
            'ASDFGH.*", "QWERTY.my.bundle.id.also'))):
      _plisttool_result({
          'plists': [{'keychain-access-groups': ['QWERTY.my.bundle.id']}],
          'entitlements_options': {
              'bundle_id': 'my.bundle.id',
              'profile_metadata_file': {
                  'Entitlements': {
                      'keychain-access-groups': [
                          'ASDFGH.*',
                          'QWERTY.my.bundle.id.also',
                      ],
                  },
                  'Version': 1,
              },
          },
      })

  def test_entitlements_app_groups_match(self):
    # This is really looking for the lack of an error being raised.
    plist1 = {
        'com.apple.security.application-groups': ['QWERTY.my.bundle.id'],
    }
    self._assert_plisttool_result({
        'plists': [plist1],
        'entitlements_options': {
            'bundle_id': 'my.bundle.id',
            'profile_metadata_file': {
                'Entitlements': {
                    'com.apple.security.application-groups': [
                        'QWERTY.my.bundle.id',
                    ],
                },
                'Version': 1,
            },
        },
    }, plist1)

  def test_entitlements_app_groups_wildcard_no_match(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.ENTITLEMENTS_HAS_GROUP_ENTRY_PROFILE_DOES_NOT % (
            _testing_target, 'com.apple.security.application-groups',
            'QWERTY.my.bundle.id', 'QWERTY.*'))):
      _plisttool_result({
          'plists': [{
              'com.apple.security.application-groups': ['QWERTY.my.bundle.id'],
          }],
          'entitlements_options': {
              'bundle_id': 'my.bundle.id',
              'profile_metadata_file': {
                  'Entitlements': {
                      'com.apple.security.application-groups': ['QWERTY.*'],
                  },
                  'Version': 1,
              },
          },
      })

  def test_entitlements_no_app_groups_requested(self):
    # This is really looking for the lack of an error being raised.
    plist1 = {'application-identifier': 'QWERTY.my.bundle.id'}
    self._assert_plisttool_result({
        'plists': [plist1],
        'entitlements_options': {
            'bundle_id': 'my.bundle.id',
            'profile_metadata_file': {
                'Entitlements': {
                    'application-identifier': 'QWERTY.*',
                    'com.apple.security.application-groups': ['ASDFGH.*'],
                },
                'Version': 1,
            },
        },
    }, plist1)

  def test_entitlements_app_groups_not_allowed(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.ENTITLEMENTS_HAS_GROUP_PROFILE_DOES_NOT % (
            _testing_target, 'com.apple.security.application-groups'))):
      _plisttool_result({
          'plists': [{
              'com.apple.security.application-groups': ['QWERTY.my.bundle.id'],
          }],
          'entitlements_options': {
              'bundle_id': 'my.bundle.id',
              'profile_metadata_file': {
                  'Entitlements': {
                      'application-identifier': 'QWERTY.*',
                      # No 'com.apple.security.application-groups'
                  },
                  'Version': 1,
              },
          },
      })

  def test_entitlements_app_groups_mismatch(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.ENTITLEMENTS_HAS_GROUP_ENTRY_PROFILE_DOES_NOT % (
            _testing_target, 'com.apple.security.application-groups',
            'QWERTY.my.bundle.id', 'ASDFGH.*", "QWERTY.my.bundle.id.also'))):
      _plisttool_result({
          'plists': [{
              'com.apple.security.application-groups': ['QWERTY.my.bundle.id'],
          }],
          'entitlements_options': {
              'bundle_id': 'my.bundle.id',
              'profile_metadata_file': {
                  'Entitlements': {
                      'com.apple.security.application-groups': [
                          'ASDFGH.*',
                          'QWERTY.my.bundle.id.also',
                      ],
                  },
                  'Version': 1,
              },
          },
      })

  def test_nfc_matching(self):
    # This is really looking for the lack of an error being raised.
    plist1 = {
        'com.apple.developer.nfc.readersession.formats': ['NDEF'],
    }
    self._assert_plisttool_result({
        'plists': [plist1],
        'entitlements_options': {
            'bundle_id': 'my.bundle.id',
            'profile_metadata_file': {
                'Entitlements': {
                    'com.apple.developer.nfc.readersession.formats': [
                        'NDEF',
                        'TAG',
                    ],
                },
                'Version': 1,
            },
        },
    }, plist1)

  def test_entitlements_aps_environment_matches(self):
    plist = {'aps-environment': 'production'}
    self._assert_plisttool_result({
        'plists': [plist],
        'entitlements_options': {
            'profile_metadata_file': {
                'Entitlements': {
                    'aps-environment': 'production'
                },
                'Version': 1,
            },
        },
    }, plist)

  def test_entitlements_aps_environment_missing_profile(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(
            plisttool.ENTITLEMENTS_MISSING % (
                _testing_target, 'aps-environment'))):
      plist = {'aps-environment': 'production'}
      self._assert_plisttool_result({
          'plists': [plist],
          'entitlements_options': {
              'profile_metadata_file': {
                  'Entitlements': {
                      'application-identifier': 'QWERTY.*',
                  },
                  'Version': 1,
              },
          },
      }, plist)

  def test_entitlements_aps_environment_mismatch(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.ENTITLEMENTS_VALUE_MISMATCH % (
            _testing_target, 'aps-environment', 'production', 'development'))):
      plist = {'aps-environment': 'production'}
      self._assert_plisttool_result({
          'plists': [plist],
          'entitlements_options': {
              'profile_metadata_file': {
                  'Entitlements': {
                      'aps-environment': 'development',
                  },
                  'Version': 1,
              },
          },
      }, plist)

  def test_attest_valid(self):
    plist = {
      'com.apple.developer.devicecheck.appattest-environment': 'development'}
    self._assert_plisttool_result({
        'plists': [plist],
        'entitlements_options': {
            'profile_metadata_file': {
                'Entitlements': {
                    'com.apple.developer.devicecheck.appattest-environment': ['development', 'production'],
                },
                'Version': 1,
            },
        },
    }, plist)

  def test_attest_mismatch(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.ENTITLEMENTS_VALUE_NOT_IN_LIST % (
            _testing_target,
          'com.apple.developer.devicecheck.appattest-environment',
          'foo', ['development']))):
      plist = {
        'com.apple.developer.devicecheck.appattest-environment': 'foo'}
      self._assert_plisttool_result({
          'plists': [plist],
          'entitlements_options': {
              'profile_metadata_file': {
                  'Entitlements': {
                      'com.apple.developer.devicecheck.appattest-environment': ['development'],
                  },
                  'Version': 1,
              },
          },
      }, plist)

  def test_entitlements_missing_beta_reports_active(self):
    plist = {}
    self._assert_plisttool_result({
        'plists': [plist],
        'entitlements_options': {
            'profile_metadata_file': {
                'Entitlements': {
                    'beta-reports-active': True,
                },
                'Version': 1,
            },
        },
    }, plist)

  def test_entitlements_beta_reports_active_mismatch(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.ENTITLEMENTS_BETA_REPORTS_ACTIVE_MISMATCH % (
            _testing_target, 'False', 'True'))):
      plist = {'beta-reports-active': False}
      self._assert_plisttool_result({
          'plists': [plist],
          'entitlements_options': {
              'profile_metadata_file': {
                  'Entitlements': {
                      'beta-reports-active': True,
                  },
                  'Version': 1,
              },
          },
      }, plist)

  def test_entitlements_profile_missing_beta_reports_active(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(
            plisttool.ENTITLEMENTS_BETA_REPORTS_ACTIVE_MISSING_PROFILE % (
                _testing_target, 'True'))):
      plist = {'beta-reports-active': True}
      self._assert_plisttool_result({
          'plists': [plist],
          'entitlements_options': {
              'profile_metadata_file': {
                  'Version': 1,
              },
          },
      }, plist)

  def test_entitlements_missing_wifi_info_active(self):
    plist = {}
    self._assert_plisttool_result({
        'plists': [plist],
        'entitlements_options': {
            'profile_metadata_file': {
                'Entitlements': {
                    'com.apple.developer.networking.wifi-info': True,
                },
                'Version': 1,
            },
        },
    }, plist)

  def test_entitlements_wifi_info_active_mismatch(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.ENTITLEMENTS_VALUE_MISMATCH % (
            _testing_target, 'com.apple.developer.networking.wifi-info',
            'False', 'True'))):
      plist = {'com.apple.developer.networking.wifi-info': False}
      self._assert_plisttool_result({
          'plists': [plist],
          'entitlements_options': {
              'profile_metadata_file': {
                  'Entitlements': {
                      'com.apple.developer.networking.wifi-info': True,
                  },
                  'Version': 1,
              },
          },
      }, plist)

  def test_entitlements_profile_missing_wifi_info_active(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(
            plisttool.ENTITLEMENTS_MISSING %
            (_testing_target, 'com.apple.developer.networking.wifi-info'))):
      plist = {'com.apple.developer.networking.wifi-info': True}
      self._assert_plisttool_result({
          'plists': [plist],
          'entitlements_options': {
              'profile_metadata_file': {
                  'Entitlements': {
                      'application-identifier': 'QWERTY.*',
                      # No wifi-info
                  },
                  'Version': 1,
              },
          },
      }, plist)

  def test_entitlements_associated_domains_match(self):
    # This is really looking for the lack of an error being raised.
    plist1 = {
        'com.apple.developer.associated-domains': ['bundle.my'],
    }
    self._assert_plisttool_result({
        'plists': [plist1],
        'entitlements_options': {
            'bundle_id': 'my.bundle.id',
            'profile_metadata_file': {
                'Entitlements': {
                    'com.apple.developer.associated-domains': [
                        'bundle.my',
                    ],
                },
                'Version': 1,
            },
        },
    }, plist1)

  def test_entitlements_associated_domains_match_wildcard(self):
    # This is really looking for the lack of an error being raised.
    plist1 = {
        'com.apple.developer.associated-domains': ['bundle.my'],
    }
    self._assert_plisttool_result({
        'plists': [plist1],
        'entitlements_options': {
            'bundle_id': 'my.bundle.id',
            'profile_metadata_file': {
                'Entitlements': {
                    'com.apple.developer.associated-domains': [
                        '*',
                    ],
                },
                'Version': 1,
            },
        },
    }, plist1)

  # pylint: disable=line-too-long
  def test_entitlements_associated_domains_match_wildcard_requesting_wildcard(self):
    # This is really looking for the lack of an error being raised.
    plist1 = {
        'com.apple.developer.associated-domains': ['my.co', '*.my.co'],
    }
    self._assert_plisttool_result({
        'plists': [plist1],
        'entitlements_options': {
            'bundle_id': 'my.bundle.id',
            'profile_metadata_file': {
                'Entitlements': {
                    'com.apple.developer.associated-domains': [
                        '*',
                    ],
                },
                'Version': 1,
            },
        },
    }, plist1)

  def test_entitlements_associated_domains_mismatch(self):
    with self.assertRaisesRegex(
        plisttool.PlistToolError,
        re.escape(plisttool.ENTITLEMENTS_HAS_GROUP_ENTRY_PROFILE_DOES_NOT % (
            _testing_target, 'com.apple.developer.associated-domains',
            'bundle.my', 'bundle.your'))):
      _plisttool_result({
          'plists': [{
              'com.apple.developer.associated-domains': ['bundle.my'],
          }],
          'entitlements_options': {
              'bundle_id': 'my.bundle.id',
              'profile_metadata_file': {
                  'Entitlements': {
                      'com.apple.developer.associated-domains': [
                          'bundle.your',
                      ],
                  },
                  'Version': 1,
              },
          },
      })


class PlistEntitlementsMerge(PlistToolTest):

  def test_entitlements_merge_from_profile_metadata(self):
    control = {
        'plists': [{
            'com.apple.developer.healthkit': True
        }],
        'entitlements_options': {
            'bundle_id': 'my.bundle.id',
            'profile_metadata_file': {
                'Entitlements': {
                    'application-identifier': 'QWERTY.my.bundle.id',
                    'get-task-allow': True
                },
                'Version': 1,
            },
        },
    }
    expected = {
        'application-identifier': 'QWERTY.my.bundle.id',
        'com.apple.developer.healthkit': True,
        'get-task-allow': True,
    }
    self._assert_plisttool_result(control, expected)

  def test_entitlement_task_update_plist(self):
    testcases = [
        {
            'testcase_name': 'adds get-task-allow from profile metadata',
            'options': {
                'profile_metadata_file': {
                    'Entitlements': {
                        'get-task-allow': True
                    },
                    'Version': 1,
                },
            },
            'out_plist': {},
            'expected': {
                'get-task-allow': True,
            }
        },
        {
            'testcase_name':
                'adds application-identifier from profile metadata',
            'options': {
                'profile_metadata_file': {
                    'Entitlements': {
                        'application-identifier': 'QWERTY.my.bundle.id',
                    },
                    'Version': 1,
                },
            },
            'out_plist': {},
            'expected': {
                'application-identifier': 'QWERTY.my.bundle.id',
            }
        },
        {
            'testcase_name':
                'adds both get-task-allow and application-identifier',
            'options': {
                'profile_metadata_file': {
                    'Entitlements': {
                        'get-task-allow': True,
                        'application-identifier': 'QWERTY.my.bundle.id',
                    },
                    'Version': 1,
                },
            },
            'out_plist': {},
            'expected': {
                'get-task-allow': True,
                'application-identifier': 'QWERTY.my.bundle.id',
            }
        },
        {
            'testcase_name':
                'updates plist with get-task-allow and application-identifier',
            'options': {
                'profile_metadata_file': {
                    'Entitlements': {
                        'get-task-allow': True,
                        'application-identifier': 'QWERTY.my.bundle.id',
                    },
                    'Version': 1,
                },
            },
            'out_plist': {
                'com.apple.developer.healthkit': True
            },
            'expected': {
                'application-identifier': 'QWERTY.my.bundle.id',
                'com.apple.developer.healthkit': True,
                'get-task-allow': True,
            }
        },
        {
            'testcase_name':
                'does not update since no Entitlements on profile metadata',
            'options': {
                'profile_metadata_file': {
                    'Version': 1,
                },
            },
            'out_plist': {
                'com.apple.developer.healthkit': True
            },
            'expected': {
                'com.apple.developer.healthkit': True
            }
        },
        {
            'testcase_name':
                'does not update key already defined in entitlements',
            'options': {
                'profile_metadata_file': {
                    'Entitlements': {
                        'get-task-allow': True,
                        'application-identifier': 'QWERTY.my.bundle.id',
                    },
                    'Version': 1,
                },
            },
            'out_plist': {
                'get-task-allow': False
            },
            'expected': {
                'get-task-allow': False,
                'application-identifier': 'QWERTY.my.bundle.id',
            }
        },
    ]

    for testcase in testcases:
      with self.subTest(testcase.get('testcase_name')):
        task = plisttool.EntitlementsTask(
            _testing_target, testcase.get('options'))
        task.update_plist(testcase.get('out_plist'), None)
        self.assertEqual(testcase.get('out_plist'), testcase.get('expected'))


if __name__ == '__main__':
  unittest.main()
