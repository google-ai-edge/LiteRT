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

"""Regression checks for indirect operator references missed by the old audit."""
from pathlib import Path
import tempfile
import unittest

import audit_lowering


class AuditTest(unittest.TestCase):
    def test_macros_aliases_and_test_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compiler = root / 'compiler'
            compiler.mkdir()
            (compiler / 'ops.cc').write_text(
                '#define UNARY(name, helper) g.Add##helper##Operator(x,t)\n'
                'UNARY(Abs, Abs)\nusing O = tfsc::tosa::OperatorEnum;\n'
                'g.AddOperator(O::Maximum, v, {t});\n')
            (compiler / 'ops_test.cc').write_text('g.AddCosOperator(x,t);')
            (compiler / 'tosa_requirements.cc').write_text('case O::Cos: return false;')
            headers = root / 'include/tosa'
            headers.mkdir(parents=True)
            for name in ['OperatorEnum.hpp', 'GraphOperatorHelpers.hpp']:
                (headers / name).write_text('OperatorEnum::Abs; OperatorEnum::Maximum;')
            database = {'instruction_set': 'TOSA.001000.1', 'operators': [
                {'name': name, 'variants': []} for name in ['ABS', 'MAXIMUM', 'COS']]}
            result = audit_lowering.audit(database, compiler, root)
            states = {op['name']: op['lowering'] for op in result['operators']}
            self.assertEqual(states, {'ABS': 'referenced', 'MAXIMUM': 'referenced',
                                      'COS': 'not_referenced'})
            self.assertIn('compiler/tosa_requirements.cc', result['source_sha256'])


if __name__ == '__main__':
    unittest.main()
