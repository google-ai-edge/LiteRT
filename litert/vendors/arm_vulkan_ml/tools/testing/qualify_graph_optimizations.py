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
"""Generate small graph rewrite fixtures; compare off/exact/qualified on device.

Requires flatbuffers, numpy, and generated TFLite Python object bindings on the
host. The C++ tool runs on Linux or Android; deploy the generated .tflite,
.inputs.json and .bin files together. No ImageNet images are used here.
"""
import argparse
import json
from pathlib import Path
import subprocess

import flatbuffers
import numpy as np
from tflite.Buffer import BufferT
from tflite.BuiltinOperator import BuiltinOperator
from tflite.BuiltinOptions import BuiltinOptions
from tflite.Conv2DOptions import Conv2DOptionsT
from tflite.Model import ModelT
from tflite.Operator import OperatorT
from tflite.OperatorCode import OperatorCodeT
from tflite.Padding import Padding
from tflite.QuantizationParameters import QuantizationParametersT
from tflite.SubGraph import SubGraphT
from tflite.Tensor import TensorT
from tflite.TensorType import TensorType


class Fixture:
    def __init__(self):
        self.model = ModelT()
        self.model.version = 3
        self.model.buffers = [BufferT()]
        self.graph = SubGraphT()
        self.graph.tensors = []
        self.graph.operators = []
        self.graph.inputs = []
        self.graph.outputs = []
        self.model.subgraphs = [self.graph]
        self.model.operatorCodes = []

    def tensor(self, shape, scale=.125, zp=-17, data=None, dtype=TensorType.INT8):
        tensor = TensorT()
        tensor.shape = shape
        tensor.type = dtype
        tensor.buffer = 0
        if scale is not None:
            tensor.quantization = QuantizationParametersT()
            tensor.quantization.scale = scale if isinstance(scale, list) else [scale]
            tensor.quantization.zeroPoint = [zp] * len(tensor.quantization.scale)
            tensor.quantization.quantizedDimension = 0
        if data is not None:
            buffer = BufferT()
            buffer.data = data.tobytes()
            tensor.buffer = len(self.model.buffers)
            self.model.buffers.append(buffer)
        index = len(self.graph.tensors)
        self.graph.tensors.append(tensor)
        return index

    def control(self, values):
        data = np.array(values, dtype='<i4')
        return self.tensor(list(data.shape), None, data=data, dtype=TensorType.INT32)

    def op(self, name, inputs, outputs, options=None, options_type=None):
        code = OperatorCodeT()
        code.builtinCode = getattr(BuiltinOperator, name)
        code.deprecatedBuiltinCode = min(code.builtinCode, 127)
        code.version = 2 if name in ['CONV_2D', 'QUANTIZE', 'TANH', 'LOGISTIC'] else 1
        op = OperatorT()
        op.opcodeIndex = len(self.model.operatorCodes)
        op.inputs = inputs
        op.outputs = outputs
        op.builtinOptions = options
        if options_type is not None:
            op.builtinOptionsType = options_type
        self.model.operatorCodes.append(code)
        self.graph.operators.append(op)

    def write(self, directory, name):
        builder = flatbuffers.Builder(1024)
        builder.Finish(self.model.Pack(builder), file_identifier=b'TFL3')
        model = directory / (name + '.tflite')
        model.write_bytes(builder.Output())
        cases = []
        for case in range(3):
            inputs = []
            for slot, index in enumerate(self.graph.inputs):
                shape = self.graph.tensors[index].shape
                count = int(np.prod(shape))
                data = (np.linspace(-128, 127, count).astype(np.int8) if case == 0 else
                        np.random.default_rng(31 * case + slot).integers(-128, 128, count, dtype=np.int8))
                path = directory / f'{name}_{case}_{slot}.bin'
                path.write_bytes(data.tobytes())
                inputs.append(dict(index=slot, file=path.name))
            cases.append(dict(id=str(case), inputs=inputs))
        manifest = directory / (name + '.inputs.json')
        manifest.write_text(json.dumps(dict(cases=cases)))
        return model, manifest


def fixtures():
    for name in ['clamp_chain', 'table_chain', 'affine_before_table', 'clamp_fanout']:
        f = Fixture()
        source = f.tensor([1, 256])
        f.graph.inputs = [source]
        if name == 'table_chain':
            middle = f.tensor([1, 256], 1 / 256, -128)
            output = f.tensor([1, 256], 1 / 128, 0)
            f.op('LOGISTIC', [source], [middle])
            f.op('TANH', [middle], [output])
        elif name == 'affine_before_table':
            middle = f.tensor([1, 256], .0625, 0)
            output = f.tensor([1, 256], 1 / 128, 0)
            f.op('QUANTIZE', [source], [middle])
            f.op('TANH', [middle], [output])
        else:
            middle, output = f.tensor([1, 256]), f.tensor([1, 256])
            f.op('RELU', [source], [middle])
            f.op('RELU6', [middle], [output])
        f.graph.outputs = [middle, output] if name == 'clamp_fanout' else [output]
        yield name, f, 'compose_int8_tables' if name == 'table_chain' else 'native_requantization' if name == 'affine_before_table' else 'compose_clamps' if name == 'clamp_chain' else None
    for name in ['transpose_inverse', 'constant_transpose']:
        f = Fixture()
        source = f.tensor([1, 2, 3, 1], data=np.arange(6, dtype=np.int8) if name == 'constant_transpose' else None)
        # Keep one unused runtime input for the common raw-tensor runner.
        f.graph.inputs = [f.tensor([1, 1])] if name == 'constant_transpose' else [source]
        middle = f.tensor([1, 3, 2, 1])
        permutation = f.control([0, 2, 1, 3])
        f.op('TRANSPOSE', [source, permutation], [middle])
        if name == 'transpose_inverse':
            output = f.tensor([1, 2, 3, 1])
            f.op('TRANSPOSE', [middle, permutation], [output])
        else:
            output = middle
        f.graph.outputs = [output]
        yield name, f, 'fold_constant_layout' if name == 'constant_transpose' else 'compose_shapes'
    for name in ['zero_pad_conv', 'nonzero_pad_conv', 'exposed_pad_conv']:
        f = Fixture()
        source = f.tensor([1, 4, 4, 1])
        middle = f.tensor([1, 6, 6, 1])
        output = f.tensor([1, 4, 4, 2], .25, 7)
        padding = f.control([[0, 0], [1, 1], [1, 1], [0, 0]])
        if name == 'nonzero_pad_conv':
            fill = f.tensor([], data=np.array([0], dtype=np.int8))
            f.op('PADV2', [source, padding, fill], [middle])
        else:
            f.op('PAD', [source, padding], [middle])
        weights = f.tensor([2, 3, 3, 1], [.5, 1e-12], 0, np.ones(18, dtype=np.int8))
        bias = f.tensor([2], [.0625, 1.25e-13], 0, np.array([0, 1073741824], dtype='<i4'), TensorType.INT32)
        options = Conv2DOptionsT()
        options.padding = Padding.VALID
        options.strideH = options.strideW = 1
        options.dilationHFactor = options.dilationWFactor = 1
        f.op('CONV_2D', [middle, weights, bias], [output], options, BuiltinOptions.Conv2DOptions)
        f.graph.inputs = [source]
        f.graph.outputs = [middle, output] if name == 'exposed_pad_conv' else [output]
        yield name, f, 'fold_zero_pad' if name == 'zero_pad_conv' else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--allow_emulation', action='store_true')
    parser.add_argument('--vulkan_validation', choices=['off', 'core', 'sync'], default='off')
    parser.add_argument('--generate_only', action='store_true', help='Prepare fixtures for adb deployment')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, fixture, expected_pass in fixtures():
        model, manifest = fixture.write(args.output_dir, name)
        if args.generate_only:
            print(name, flush=True)
            continue
        for mode in ['exact', 'qualified']:
            stem = args.output_dir / (name + '-' + mode)
            common = [args.tool, 'compile', '--model=' + str(model), '--optimization_mode=' + mode,
                      '--output_dir=' + str(stem), '--report_json=' + str(stem) + '.compile.json']
            with Path(str(stem) + '.log').open('w') as log:
                compiled = subprocess.run(common, stdout=log, stderr=subprocess.STDOUT, check=False)
                if compiled.returncode:
                    raise SystemExit('Compilation failed: ' + str(stem))
                report = json.loads(Path(str(stem) + '.compile.json').read_text())
                decisions = [d for s in report['subgraphs'] for p in s['partitions'] for d in p['optimization']['decisions']]
                if not report['complete_coverage'] or (expected_pass and not any(d['pass'] == expected_pass and d['applied'] for d in decisions)):
                    raise SystemExit('Expected rewrite/coverage missing: ' + str(stem))
                if name in ['nonzero_pad_conv', 'exposed_pad_conv'] and any(d['pass'] == 'fold_zero_pad' and d['applied'] for d in decisions):
                    raise SystemExit('Illegal PAD fusion: ' + name)
                compare = [args.tool, 'compare', '--model=' + str(model), '--tensor_inputs=' + str(manifest),
                           '--optimization_mode=' + mode, '--max_samples=3', '--allow_emulation=' + str(int(args.allow_emulation)),
                           '--vulkan_validation=' + args.vulkan_validation, '--report_json=' + str(stem) + '.compare.json']
                result = subprocess.run(compare, stdout=log, stderr=subprocess.STDOUT, check=False)
                if result.returncode:
                    raise SystemExit('Output/validation gate failed: ' + str(stem))
            results.append(dict(fixture=name, mode=mode, status='passed'))
            print(name, mode, 'passed', flush=True)
    (args.output_dir / 'results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
