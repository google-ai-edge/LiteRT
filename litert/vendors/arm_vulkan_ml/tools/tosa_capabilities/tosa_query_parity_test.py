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
"""Compare C++ CLI results with the Python reference on the generated database.

Run on the host; Android only needs the C++ executable and database JSON.
"""
import argparse
import json
from pathlib import Path
import subprocess

import capability_query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, required=True)
    parser.add_argument('--database', type=Path, default=capability_query.DATABASE)
    args = parser.parse_args()
    db = json.loads(args.database.read_text())
    checked = 0

    def check(op, types=None, capabilities=None, enums=None):
        nonlocal checked
        command = [str(args.tool.resolve()), 'tosa-query', '--database',
                   str(args.database), '--op', op, '--json']
        for key, value in (types or {}).items():
            command += ['--type', f'{key}={value}']
        for key, value in (enums or {}).items():
            command += ['--enum', f'{key}={value}']
        for capability in capabilities or []:
            command += ['--capability', capability]
        expected_error = False
        try:
            original, rows = capability_query.find_variants(
                db, op, types, capabilities, enums)
        except ValueError:
            expected_error = True
        process = subprocess.run(command, capture_output=True, text=True,
                                 timeout=30, check=False)
        result = json.loads(process.stdout)
        if expected_error:
            assert process.returncode != 0 and result['status'] == 'failed', command
        else:
            assert process.returncode == 0, (command, result)
            assert result['variants'] == rows, (command, 'rows differ')
            for key, value in original.items():
                if key != 'variants':
                    assert result[key] == value, (command, key)
            assert result['candidate_count'] == len(rows)
            assert result['device_support'] == 'not_queried'
            assert result['shape_constraints'] == 'not_evaluated'
        checked += 1

    for op in db['operators']:
        check(op['name'])
        for profile in db['profiles']:
            check(op['name'], capabilities=[profile])
        for row in op['variants']:
            types = {k: v['element_type'] for k, v in row['types'].items()}
            check(op['name'], types=types, enums=row['parameters'])
    check('conv2d', {'input': 'fp16_t'}, ['PRO-FP'], {'acc_type': 'FP32'})
    check('CAST', {'input': 'bf16_t', 'output': 'fp8e4m3_t'},
          ['PRO-FP', 'EXT-BF16', 'EXT-FP8E4M3'])
    check('CAST', {'input': 'bf16_t', 'output': 'fp8e4m3_t'},
          ['PRO-FP', 'EXT-BF16'])
    for caps in [['PRO-INT'], ['PRO-INT', 'EXT-DOUBLEROUND']]:
        check('RESCALE', capabilities=caps, enums={'rounding_mode': 'DOUBLE_ROUND'})
    check('UNKNOWN')
    check('ADD', {'unknown': 'fp32_t'})
    check('ADD', {'input1': 'unknown'})
    check('ADD', capabilities=['EXT-INT4'])
    check('CONV2D', enums={'acc_type': 'UNKNOWN'})
    check('CONV2D', enums={'input': 'FP32'})
    print(f'{checked} C++/Python query cases matched')


if __name__ == '__main__':
    main()
