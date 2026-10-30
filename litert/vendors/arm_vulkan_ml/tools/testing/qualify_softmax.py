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
"""Qualify INT8 Softmax through the C++ tool, with synthetic inputs only.

Uses the same host flatbuffers/numpy/TFLite bindings as the other fixture
generators. Both LUT layouts are compiled and SPIR-V validated. INT8 runs
against the independent FP64 oracle and XNNPACK QU8. INT16 execution requires
an engine advertising EXT-INT16; unavailability is recorded, never bypassed.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from tflite.BuiltinOptions import BuiltinOptions
from tflite.SoftmaxOptions import SoftmaxOptionsT

from qualify_graph_optimizations import Fixture


def fixtures():
    for channels in [1, 2, 3, 5, 32, 255, 256, 512, 1000, 1001, 1024, 2048, 4095]:
        yield f'channels_{channels}', [2, channels], .125, -17, 1.0
    for name, shape, scale, zp, beta in [
        ('rank1', [256], .03125, 127, 1.0),
        ('rank3', [2, 3, 32], .25, -128, .5),
        ('rank4', [1, 2, 2, 5], .0625, 0, 2.0),
        ('small_scale', [2, 1000], 1e-5, 0, 1.0),
        ('large_scale', [2, 1000], 8.0, -31, 1.0),
        ('beta_fraction', [2, 255], .13, 19, .7),
    ]:
        yield name, shape, scale, zp, beta


def write_fixture(directory, name, shape, scale, zp, beta):
    fixture = Fixture()
    source = fixture.tensor(shape, scale, zp)
    output = fixture.tensor(shape, 1 / 256, -128)
    fixture.graph.inputs = [source]
    fixture.graph.outputs = [output]
    options = SoftmaxOptionsT()
    options.beta = beta
    fixture.op('SOFTMAX', [source], [output], options, BuiltinOptions.SoftmaxOptions)
    fixture.model.operatorCodes[0].version = 2
    # Fixture.write also packs the model; replace its generic input manifest.
    model, manifest = fixture.write(directory, name)
    count, channels = int(np.prod(shape)), shape[-1]
    rng = np.random.default_rng(71)
    values = {
        'uniform': np.full(count, -37, dtype=np.int8),
        'random': rng.integers(-128, 128, count, dtype=np.int8),
        'all_codes': (np.arange(count) % 256 - 128).astype(np.int8),
        'near_ties': rng.integers(123, 128, count, dtype=np.int8),
        'one_peak': np.full(count, -128, dtype=np.int8),
        'long_tail': np.maximum(-128, 127 - np.arange(count) % channels).astype(np.int8),
    }
    values['one_peak'][::channels] = 127
    cases = []
    for case, data in values.items():
        path = directory / f'{name}_{case}.bin'
        path.write_bytes(data.tobytes())
        cases.append(dict(id=case, inputs=[dict(index=0, file=path.name)]))
    manifest.write_text(json.dumps(dict(cases=cases), indent=2) + '\n')
    return model, manifest, cases


def run(tool, command, report, log, arguments):
    report.unlink(missing_ok=True)
    with log.open('a') as output:
        completed = subprocess.run(
            [tool, command, '--report_json=' + str(report), *arguments],
            stdout=output, stderr=subprocess.STDOUT, check=False)
    if not report.is_file():
        return completed.returncode, dict(status='failed', error=f'No report; process exited with {completed.returncode}')
    return completed.returncode, json.loads(report.read_text())


def require(condition, message, log):
    if not condition:
        raise RuntimeError(message + '; see ' + str(log))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--allow_emulation', action='store_true')
    parser.add_argument('--vulkan_validation', choices=['off', 'core', 'sync'], default='sync')
    parser.add_argument('--generate_only', action='store_true')
    parser.add_argument('--fixture', action='append', help='Select a fixture; repeat to select several')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    if args.fixture:
        unknown = set(args.fixture) - {name for name, *_ in fixtures()}
        if unknown:
            parser.error('Unknown fixtures: ' + ', '.join(sorted(unknown)))
    for name, shape, scale, zp, beta in fixtures():
        if args.fixture and name not in args.fixture:
            continue
        model, manifest, cases = write_fixture(args.output_dir, name, shape, scale, zp, beta)
        if args.generate_only:
            print(name, flush=True)
            continue
        log = args.output_dir / (name + '.log')
        base = ['--model=' + str(model), '--tensor_inputs=' + str(manifest),
                '--allow_emulation=' + str(int(args.allow_emulation)),
                '--vulkan_validation=' + args.vulkan_validation,
                '--max_code_error=1', '--mean_code_error=0.5']
        row = dict(fixture=name, shape=shape, scale=scale, beta=beta, layouts={})
        for lut in ['int8', 'int16']:
            stem = args.output_dir / (name + '-' + lut)
            code, compiled = run(args.tool, 'compile', Path(str(stem) + '.compile.json'), log,
                                 ['--model=' + str(model), '--softmax_lut_type=' + lut,
                                  '--output_dir=' + str(stem)])
            require(code == 0 and compiled.get('complete_coverage'), 'Compilation failed', log)
            partitions = [p for s in compiled['subgraphs'] for p in s['partitions']]
            require(len(partitions) == 1 and partitions[0]['spirv_valid'], 'SPIR-V preflight failed', log)
            partition = partitions[0]
            require(partition['profiles'] == ['PRO-INT'], 'Unexpected profile', log)
            require(partition['extensions'] == (['EXT-INT16'] if lut == 'int16' else []),
                    'Incorrect extension requirements', log)
            capability_path = Path(str(stem) + '.capabilities.json')
            code, caps = run(args.tool, 'capabilities', capability_path, log,
                             ['--model=' + str(model), '--softmax_lut_type=' + lut,
                              '--allow_emulation=' + str(int(args.allow_emulation)),
                              '--vulkan_validation=' + args.vulkan_validation])
            # Check the explicit engine-match result before trying a pipeline.
            supported = any(e.get('reported_requirements_match') for d in caps.get('devices', []) for e in d.get('engines', []))
            if code or not supported:
                rejections = [e.get('rejection', '') for d in caps.get('devices', []) for e in d.get('engines', [])]
                require(lut == 'int16' and any('TOSA extension is not advertised as conformant: EXT-INT16' in r for r in rejections),
                        'Required INT8 capabilities unavailable', log)
                row['layouts'][lut] = dict(compiled=True, executed=False,
                                           reason='No engine accepts EXT-INT16 artifact', capabilities=str(capability_path))
                continue
            layout = dict(compiled=True, executed=True, references={})
            for reference in ['fp64', 'xnnpack', 'tflite']:
                path = Path(str(stem) + '.' + reference + '.json')
                code, report = run(args.tool, 'validate', path, log,
                                   [*base, '--softmax_lut_type=' + lut, '--reference=' + reference,
                                    '--max_samples=' + str(len(cases))])
                case_reports = report.get('cases', [report])
                unavailable = [r for r in case_reports if r.get('status') == 'failed' and
                               r.get('reference_status') == 'unavailable' and
                               reference == 'xnnpack' and 'exceeds UINT32' in r.get('error', '')]
                failed = [r for r in case_reports if r.get('status') != 'passed' and r not in unavailable]
                # Legacy TFLite fixed-point arithmetic is a compatibility
                # measurement, not the mathematical correctness authority.
                if reference != 'tflite':
                    require(not failed and (not code or bool(unavailable)), reference + ' gate failed', log)
                layout['references'][reference] = dict(
                    report=str(path), passed=len(case_reports) - len(failed) - len(unavailable),
                    unavailable=[r.get('case_id') for r in unavailable],
                    failed=[r.get('case_id') for r in failed],
                    errors=[r.get('error', '') for r in failed])
            row['layouts'][lut] = layout
        results.append(row)
        (args.output_dir / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(name, 'passed', flush=True)


if __name__ == '__main__':
    main()
