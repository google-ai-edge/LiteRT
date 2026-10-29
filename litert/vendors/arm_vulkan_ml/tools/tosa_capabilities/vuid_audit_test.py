# Copyright 2026 The LiteRT Authors. All Rights Reserved.
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

import unittest
import vuid_audit

class ReviewTest(unittest.TestCase):
    def setUp(self):
        self.inventory = dict(source_sha256='pin', source_files={'dispatch.cc': 'hash'},
                              requirements={'VUID-a': {}})
        self.manifest = dict(source_sha256='pin', source_files={'dispatch.cc': 'hash'},
                             groups=[dict(vuids=['VUID-a'], status='enforced',
                                          reason='Fixed construction', evidence=['dispatch.cc'])])

    def test_complete_requires_evidence(self):
        self.assertEqual(vuid_audit.review(self.inventory, self.manifest), [])
        self.manifest['groups'][0]['evidence'] = []
        self.assertTrue(vuid_audit.review(self.inventory, self.manifest))

    def test_new_rules_and_changed_sources_fail(self):
        self.inventory['requirements']['VUID-b'] = {}
        self.inventory['source_files']['dispatch.cc'] = 'changed'
        errors = vuid_audit.review(self.inventory, self.manifest)
        self.assertTrue(any('unreviewed' in e for e in errors))
        self.assertTrue(any('inventory changed' in e for e in errors))

    def test_tested_requires_target_and_duplicates_fail(self):
        self.manifest['groups'][0]['status'] = 'tested'
        self.manifest['groups'].append(self.manifest['groups'][0])
        errors = vuid_audit.review(self.inventory, self.manifest)
        self.assertTrue(any('test target' in e for e in errors))
        self.assertTrue(any('duplicate' in e for e in errors))

if __name__ == '__main__':
    unittest.main()
