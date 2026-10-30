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
"""Qualify new frontend routes using synthetic tensors and the C++ devtool.

This generates TFLite models directly; it does not invoke or depend on the TOSA
converter. Its source is a learning reference, not a test oracle. Numerical
comparisons use the local TFLite kernels. No dataset images are consumed.
"""
import argparse
import importlib
import json
from pathlib import Path
import re
import subprocess

import flatbuffers
import numpy as np
from tflite.BuiltinOptions import BuiltinOptions
from tflite.TensorType import TensorType

from qualify_graph_optimizations import Fixture

DTYPES = {TensorType.FLOAT32: np.dtype('<f4'), TensorType.INT8: np.dtype('i1'),
          TensorType.INT32: np.dtype('<i4'), TensorType.BOOL: np.dtype('?'),
          TensorType.COMPLEX64: np.dtype('<c8')}
NAMES = {TensorType.FLOAT32: 'FP32', TensorType.INT8: 'INT8',
         TensorType.INT32: 'INT32', TensorType.BOOL: 'BOOL', TensorType.COMPLEX64: 'COMPLEX64'}
REGISTRY = Path(__file__).resolve().parents[5] / 'tflite/kernels/register_ref.cc'


def tensor(f, shape, dtype, data=None, scale=.125, zp=-17):
    if data is not None:
        data = np.asarray(data, dtype=DTYPES[dtype]).reshape(shape)
    return f.tensor(shape, scale if dtype == TensorType.INT8 else None, zp,
                    data=data, dtype=dtype)


def inputs(f, specs):
    result = [tensor(f, shape, dtype) for shape, dtype in specs]
    f.graph.inputs = result
    return result


def op(f, name, operands, output, options=None, **kwargs):
    values = None
    if options:
        values = getattr(importlib.import_module('tflite.' + options), options + 'T')()
        for key, value in kwargs.items():
            setattr(values, key, value)
    f.op(name, operands, [output], values,
         getattr(BuiltinOptions, options) if options else None)
    f.graph.outputs = [output]


def fixtures():
    f32, i8, i32, boolean = TensorType.FLOAT32, TensorType.INT8, TensorType.INT32, TensorType.BOOL
    for name in ['GREATER', 'GREATER_EQUAL', 'LESS', 'LESS_EQUAL']:
        for dtype in [f32, i8, i32]:
            f = Fixture()
            a, b = inputs(f, [([2, 3], dtype), ([3], dtype)])
            out = tensor(f, [2, 3], boolean)
            op(f, name, [a, b], out)
            yield name + '_' + NAMES[dtype], f
    for name in ['SELECT', 'SELECT_V2']:
        for dtype in [f32, i8, i32, boolean]:
            f = Fixture()
            cond_shape = [2] if name == 'SELECT' else [2, 1]
            y_shape = [2, 3] if name == 'SELECT' else [3]
            c, x, y = inputs(f, [(cond_shape, boolean), ([2, 3], dtype), (y_shape, dtype)])
            out = tensor(f, [2, 3], dtype)
            op(f, name, [c, x, y], out)
            yield name + '_' + NAMES[dtype], f
    for name in ['SELECT', 'SELECT_V2']:
        for singleton in [False, True]:
            f = Fixture()
            c, x, y = inputs(f, [([1, 1, 1] if singleton else [], boolean),
                                ([1, 1] if singleton else [2, 3], f32),
                                ([] if singleton else [2, 3], f32)])
            out = tensor(f, [] if singleton else [2, 3], f32)
            op(f, name, [c, x, y], out)
            yield name + ('_SINGLETON' if singleton else '_SCALAR_CONDITION'), f
    for name in ['LEAKY_RELU', 'PRELU', 'RELU_N1_TO_1', 'RELU_0_TO_1']:
        for dtype in [f32, i8]:
            f = Fixture()
            source = inputs(f, [([2, 3], dtype)])[0]
            out = tensor(f, [2, 3], dtype)
            if name == 'LEAKY_RELU':
                op(f, name, [source], out, 'LeakyReluOptions', alpha=.25)
            elif name == 'PRELU':
                alpha = tensor(f, [3], dtype,
                               [2, 8, -4] if dtype == i8 else [.25, 1, -.5], zp=0)
                op(f, name, [source, alpha], out)
            else:
                op(f, name, [source], out)
            yield name + '_' + NAMES[dtype], f
    for name in ['EXP', 'LOG', 'RSQRT', 'GELU', 'SQUARE']:
        f = Fixture()
        source = tensor(f, [1, 256], i8, scale=.03125, zp=-128 if name in ['LOG', 'RSQRT'] else 0)
        f.graph.inputs = [source]
        out = tensor(f, [1, 256], i8, scale=.0625, zp=0)
        if name == 'GELU':
            op(f, name, [source], out, 'GeluOptions', approximate=False)
        else:
            op(f, name, [source], out)
        yield name + '_INT8', f
    for name in ['SPACE_TO_DEPTH', 'DEPTH_TO_SPACE', 'SPACE_TO_BATCH_ND', 'BATCH_TO_SPACE_ND']:
        for dtype in [f32, i8, i32]:
            f = Fixture()
            source_shape = [1, 4, 6, 2]
            if name == 'DEPTH_TO_SPACE': source_shape = [1, 2, 3, 8]
            if name == 'BATCH_TO_SPACE_ND': source_shape = [6, 3, 2, 2]
            source = inputs(f, [(source_shape, dtype)])[0]
            target = {'SPACE_TO_DEPTH': [1, 2, 3, 8], 'DEPTH_TO_SPACE': [1, 4, 6, 2],
                      'SPACE_TO_BATCH_ND': [6, 3, 2, 2], 'BATCH_TO_SPACE_ND': [1, 4, 6, 2]}[name]
            out = tensor(f, target, dtype)
            if name in ['SPACE_TO_DEPTH', 'DEPTH_TO_SPACE']:
                op(f, name, [source], out,
                   'SpaceToDepthOptions' if name == 'SPACE_TO_DEPTH' else 'DepthToSpaceOptions', blockSize=2)
            else:
                block = tensor(f, [2], i32, [2, 3])
                padding = tensor(f, [2, 2], i32, [[1, 1], [0, 0]])
                op(f, name, [source, block, padding], out)
            yield name + '_' + NAMES[dtype], f
    for dtype in [f32, i8, i32]:
        for mode in [0, 1]:
            f = Fixture()
            source = inputs(f, [([1, 2, 3, 1], dtype)])[0]
            padding = tensor(f, [4, 2], i32, [[0, 0], [1, 1], [1, 1], [0, 0]])
            out = tensor(f, [1, 4, 5, 1], dtype)
            op(f, 'MIRROR_PAD', [source, padding], out, 'MirrorPadOptions', mode=mode)
            yield f'MIRROR_PAD_{NAMES[dtype]}_{mode}', f
    for dtype in [f32, i8, i32]:
        f = Fixture()
        source = inputs(f, [([1, 3], dtype)])[0]
        shape = tensor(f, [3], i32, [2, 4, 3])
        out = tensor(f, [2, 4, 3], dtype)
        op(f, 'BROADCAST_TO', [source, shape], out)
        yield 'BROADCAST_TO_' + NAMES[dtype], f
    for source_type, target_type in [(a, b) for a in [i8, f32, i32, boolean]
                                    for b in [i8, f32, i32, boolean]]:
        f = Fixture()
        source = inputs(f, [([1, 17], source_type)])[0]
        out = tensor(f, [1, 17], target_type)
        op(f, 'CAST', [source], out, 'CastOptions', inDataType=source_type, outDataType=target_type)
        yield 'CAST_' + NAMES[source_type] + '_' + NAMES[target_type], f
    for name in ['ADD_N', 'SIGN', 'ZEROS_LIKE']:
        for dtype in [f32, i32]:
            f = Fixture()
            operands = inputs(f, [([2, 3], dtype)] * (3 if name == 'ADD_N' else 1))
            out = tensor(f, [2, 3], dtype)
            op(f, name, operands, out)
            yield name + '_' + NAMES[dtype], f
    for name in ['ROUND', 'ELU', 'LOG_SOFTMAX', 'L2_NORMALIZATION']:
        f = Fixture()
        source = inputs(f, [([2, 6], f32)])[0]
        out = tensor(f, [2, 6], f32)
        op(f, name, [source], out, 'L2NormOptions' if name == 'L2_NORMALIZATION' else None)
        yield name, f
    for name in ['SQUARED_DIFFERENCE', 'FLOOR_DIV', 'FLOOR_MOD']:
        for dtype in ([f32, i32] if name == 'FLOOR_MOD' else [f32]):
            f = Fixture()
            source = inputs(f, [([2, 3], dtype)])[0]
            rhs = tensor(f, [3], dtype,
                         [2, 4, 8] if dtype == f32 and name != 'SQUARED_DIFFERENCE' else [2, 3, 4])
            out = tensor(f, [2, 3], dtype)
            op(f, name, [source, rhs], out)
            yield name + '_' + NAMES[dtype], f
    f = Fixture()
    source = inputs(f, [([2, 3], f32)])[0]
    rhs = tensor(f, [3], f32, [2.0 ** -30, 2.0 ** 126, -(2.0 ** 126)])
    out = tensor(f, [2, 3], f32)
    op(f, 'FLOOR_MOD', [source, rhs], out)
    yield 'FLOOR_MOD_FP32_EXTREMES', f
    for dtype in [f32, i8, i32]:
        for scalar in [False, True]:
            f = Fixture()
            source = inputs(f, [([6] if scalar else [2, 3], dtype)])[0]
            axis = tensor(f, [], i32, 0 if scalar else 1)
            out = tensor(f, [] if scalar else [2], i32)
            op(f, 'ARG_MIN', [source, axis], out, 'ArgMinOptions', outputType=i32)
            yield f'ARG_MIN_{NAMES[dtype]}_{scalar}', f
    for name in ['SHAPE', 'RANK']:
        for dtype in [f32, i8, i32, boolean]:
            f = Fixture()
            source = inputs(f, [([2, 3], dtype)])[0]
            out = tensor(f, [2] if name == 'SHAPE' else [], i32)
            if name == 'SHAPE':
                op(f, name, [source], out, 'ShapeOptions', outType=i32)
            else:
                op(f, name, [source], out)
            yield name + '_' + NAMES[dtype], f
    for dtype in [f32, i32]:
        f = Fixture()
        source = inputs(f, [([], dtype)])[0]
        dimensions = tensor(f, [2], i32, [2, 3])
        out = tensor(f, [2, 3], dtype)
        op(f, 'FILL', [dimensions, source], out)
        yield 'FILL_' + NAMES[dtype], f
    for dtype in [f32, i8, i32]:
        for scalar in [False, True]:
            f = Fixture()
            source = inputs(f, [([2, 3, 2], dtype)])[0]
            indices = tensor(f, [3] if scalar else [3, 2], i32,
                             [1, 2, 1] if scalar else [[1, 2], [0, 0], [1, 0]])
            out = tensor(f, [] if scalar else [3, 2], dtype)
            op(f, 'GATHER_ND', [source, indices], out)
            yield f'GATHER_ND_{NAMES[dtype]}_{scalar}', f
    for dtype in [f32, i8, i32, boolean]:
        for axis in [-1, 0, 1]:
            f = Fixture()
            source = inputs(f, [([2, 3], i32)])[0]
            depth = tensor(f, [], i32, 4)
            on = tensor(f, [], dtype, 1)
            off = tensor(f, [], dtype, 0)
            shape = [2, 3, 4] if axis == -1 else [4, 2, 3] if axis == 0 else [2, 4, 3]
            out = tensor(f, shape, dtype)
            op(f, 'ONE_HOT', [source, depth, on, off], out, 'OneHotOptions', axis=axis)
            yield f'ONE_HOT_{NAMES[dtype]}_{axis}', f
    for dtype in [f32, i32]:
        for scalar in [False, True]:
            f = Fixture()
            source = inputs(f, [([] if scalar else [3], dtype)])[0]
            indices = tensor(f, [3, 2], i32, [[0, 1], [1, 0], [1, 2]])
            dimensions = tensor(f, [2], i32, [2, 3])
            default = tensor(f, [], dtype, -2)
            out = tensor(f, [2, 3], dtype)
            op(f, 'SPARSE_TO_DENSE', [indices, dimensions, source, default], out,
               'SparseToDenseOptions', validateIndices=True)
            yield f'SPARSE_TO_DENSE_{NAMES[dtype]}_{scalar}', f
        f = Fixture()
        source = inputs(f, [([2, 3], dtype)])[0]
        out = tensor(f, [2, 3], i32)
        op(f, 'BUCKETIZE', [source], out, 'BucketizeOptions', boundaries=[-2.5, 0., .5, 2.5])
        yield 'BUCKETIZE_' + NAMES[dtype], f
    for bits in [4, 8, 16]:
        f = Fixture()
        source = inputs(f, [([2, 6], f32)])[0]
        out = tensor(f, [2, 6], f32)
        op(f, 'FAKE_QUANT', [source], out, 'FakeQuantOptions', min=-2.3, max=3.1,
           numBits=bits, narrowRange=False)
        yield f'FAKE_QUANT_{bits}', f
    f = Fixture()
    source, other = inputs(f, [([2, 6], f32), ([2, 6], f32)])
    out = tensor(f, [2, 6], f32)
    op(f, 'ATAN2', [source, other], out)
    yield 'ATAN2', f
    for name in ['REAL', 'IMAG']:
        for shape in [[], [2, 3]]:
            f = Fixture()
            source = inputs(f, [(shape, TensorType.COMPLEX64)])[0]
            out = tensor(f, shape, f32)
            op(f, name, [source], out)
            yield f'{name}_' + ('scalar' if not shape else 'matrix'), f
    for height, width in [(4, 8), (2, 4), (8, 16)]:
        f = Fixture()
        source = inputs(f, [([2, 4, 8], f32)])[0]
        length = tensor(f, [2], i32, [height, width])
        out = tensor(f, [2, height, width // 2 + 1], TensorType.COMPLEX64)
        op(f, 'RFFT2D', [source, length], out)
        yield f'RFFT2D_{height}_{width}', f
    for axes, keep in [([1], True), ([1], False), ([0, 2], False), ([0, 1, 2], False)]:
        f = Fixture()
        source = inputs(f, [([2, 3, 4], i8)])[0]
        axis = tensor(f, [len(axes)], i32, axes)
        shape = [1 if a in axes else dim for a, dim in enumerate([2, 3, 4])]
        if not keep:
            shape = [dim for a, dim in enumerate(shape) if a not in axes]
        out = tensor(f, shape, i8, scale=.0625, zp=3)
        op(f, 'MEAN', [source, axis], out, 'ReducerOptions', keepDims=keep)
        yield f'MEAN_INT8_{axes}_{keep}', f
    for adj_x in [False, True]:
        for adj_y in [False, True]:
            f = Fixture()
            lhs_shape = [2, 1] + ([4, 3] if adj_x else [3, 4])
            rhs_shape = [1, 3] + ([5, 4] if adj_y else [4, 5])
            lhs, rhs = inputs(f, [(lhs_shape, i8), (rhs_shape, i8)])
            out = tensor(f, [2, 3, 3, 5], i8, scale=.25, zp=7)
            op(f, 'BATCH_MATMUL', [lhs, rhs], out, 'BatchMatMulOptions', adjX=adj_x, adjY=adj_y)
            yield f'BATCH_MATMUL_INT8_{adj_x}_{adj_y}', f
    for dtype in [f32, i8]:
        for multiplier in [2, 3]:
            for padding in [0, 1]:
                f = Fixture()
                source = inputs(f, [([1, 5, 6, 2], dtype)])[0]
                channels = 2 * multiplier
                weights = (np.arange(3 * 3 * channels) % 7 - 3).reshape([1, 3, 3, channels])
                weight = tensor(f, list(weights.shape), dtype, weights, scale=.0625, zp=0)
                if dtype == i8:
                    bias = f.tensor([channels], [.0078125] * channels, 0,
                                    np.arange(channels, dtype='<i4'), TensorType.INT32)
                else:
                    bias = tensor(f, [channels], f32, np.arange(channels) * .1)
                out_shape = [1, 3, 3, channels] if padding == 0 else [1, 2, 2, channels]
                out = tensor(f, out_shape, dtype, scale=.5, zp=-3)
                op(f, 'DEPTHWISE_CONV_2D', [source, weight, bias], out,
                   'DepthwiseConv2DOptions', padding=padding, strideH=2, strideW=2,
                   depthMultiplier=multiplier, dilationHFactor=1, dilationWFactor=1)
                yield f'DEPTHWISE_{NAMES[dtype]}_{multiplier}_{padding}', f
    for name in ['AVERAGE_POOL_2D', 'MAX_POOL_2D']:
        for kernel, stride in [(3, 2), (4, 3), (2, 1)]:
            f = Fixture()
            source = inputs(f, [([1, 5, 6, 2], i8)])[0]
            out = tensor(f, [1, (5 + stride - 1) // stride, (6 + stride - 1) // stride, 2], i8)
            op(f, name, [source], out, 'Pool2DOptions', padding=0, strideH=stride, strideW=stride,
               filterHeight=kernel, filterWidth=kernel)
            yield f'{name}_SAME_{kernel}_{stride}', f
    f = Fixture()
    source = inputs(f, [([2, 6], f32)])[0]
    out = tensor(f, [2, 6], f32)
    op(f, 'GELU', [source], out, 'GeluOptions', approximate=True)
    yield 'GELU_FP32_APPROXIMATE', f
    f = Fixture()
    source = tensor(f, [256], i8, scale=.03125, zp=0)
    f.graph.inputs = [source]
    out = tensor(f, [256], i8, scale=.0625, zp=0)
    op(f, 'GELU', [source], out, 'GeluOptions', approximate=True)
    yield 'GELU_INT8_APPROXIMATE', f
    for padding in [0, 1]:
        for stride in [1, 2]:
            f = Fixture()
            source = inputs(f, [([1, 4, 5, 6, 2], i8)])[0]
            weights = (np.arange(2 * 3 * 2 * 2 * 3) % 7 - 3).astype(np.int8).reshape([2, 3, 2, 2, 3])
            weight = f.tensor(list(weights.shape), [.03125, .0625, .125], 0, weights)
            f.graph.tensors[weight].quantization.quantizedDimension = 4
            bias = f.tensor([3], [.00390625, .0078125, .015625], 0,
                            np.array([-3, 0, 7], dtype='<i4'), TensorType.INT32)
            shape = [1] + ([(n + stride - 1) // stride for n in [4, 5, 6]] if padding == 0 else
                           [(n - k) // stride + 1 for n, k in zip([4, 5, 6], [2, 3, 2])]) + [3]
            out = tensor(f, shape, i8, scale=.5, zp=7)
            op(f, 'CONV_3D', [source, weight, bias], out, 'Conv3DOptions', padding=padding,
               strideD=stride, strideH=stride, strideW=stride,
               dilationDFactor=1, dilationHFactor=1, dilationWFactor=1)
            yield f'CONV3D_INT8_{padding}_{stride}', f


def write(f, directory, name):
    # Use the latest supported registration version in this checkout; this is
    # independent of the compiler plugin's coverage registry.
    from tflite.BuiltinOperator import BuiltinOperator
    source = REGISTRY.read_text()
    versions = {}
    for operator, body in re.findall(r'AddBuiltin\(BuiltinOperator_(\w+),([\s\S]*?);', source):
        maximum = re.search(r'max_version\s*=\s*\*/\s*(\d+)', body)
        versions[getattr(BuiltinOperator, operator)] = int(maximum.group(1)) if maximum else 1
    for code in f.model.operatorCodes:
        code.version = versions.get(code.builtinCode, 1)
    builder = flatbuffers.Builder(1024)
    builder.Finish(f.model.Pack(builder), file_identifier=b'TFL3')
    model = directory / (name + '.tflite')
    model.write_bytes(builder.Output())
    cases = []
    for case in range(3):
        raw = []
        for slot, index in enumerate(f.graph.inputs):
            spec = f.graph.tensors[index]
            count = int(np.prod(spec.shape))
            rng = np.random.default_rng(67 + case * 7 + slot)
            if spec.type == TensorType.FLOAT32:
                data = (np.linspace(-3.5, 3.5, count) if case == 0 else
                        np.zeros(count) if case == 1 else rng.uniform(-5, 5, count))
            elif spec.type == TensorType.INT8:
                data = (np.linspace(-128, 127, count) if case == 0 else
                        np.zeros(count) if case == 1 else rng.integers(-128, 128, count))
            elif spec.type == TensorType.BOOL:
                data = np.arange(count) % 2 if case == 0 else rng.integers(0, 2, count)
            elif spec.type == TensorType.COMPLEX64:
                data = rng.normal(size=count) + 1j * rng.normal(size=count)
            else:
                data = (np.linspace(-513, 511, count) if case == 0 else
                        np.zeros(count) if case == 1 else rng.integers(-1000, 1001, count))
            if name.startswith('ONE_HOT'):
                data = np.resize([-1, 0, 1, 2, 3, 4, 9], count) if case == 0 else rng.integers(-2, 6, count)
            if name == 'ROUND' and case == 0:
                data = np.arange(-6, 6, dtype=np.float32) + .5
            if spec.type == TensorType.INT32 and case == 0 and name.startswith(('CAST_', 'ARG_MIN_')):
                data = np.resize([-2147483648, 2147483647, -257, -128, -1, 0, 1, 127, 128, 255, 256], count)
            if name == 'FLOOR_MOD_FP32_EXTREMES' and case == 0:
                data = [-np.finfo(np.float32).max, np.finfo(np.float32).max,
                        -np.finfo(np.float32).max, 1e-30, -1e-30, -0.0]
            path = directory / f'{name}_{case}_{slot}.bin'
            path.write_bytes(np.asarray(data, dtype=DTYPES[spec.type]).tobytes())
            raw.append(dict(index=slot, file=path.name))
        cases.append(dict(id=str(case), inputs=raw))
    manifest = directory / (name + '.inputs.json')
    manifest.write_text(json.dumps(dict(cases=cases), indent=2) + '\n')
    return model, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--allow_emulation', action='store_true')
    parser.add_argument('--vulkan_validation', choices=['off', 'core', 'sync'], default='sync')
    parser.add_argument('--fixture', action='append')
    parser.add_argument('--generate_only', action='store_true')
    parser.add_argument('--skip_passed', action='store_true',
                        help='Reuse passing results from this output directory')
    args = parser.parse_args()
    if args.fixture:
        unknown = set(args.fixture) - {name for name, _ in fixtures()}
        if unknown:
            parser.error('Unknown fixtures: ' + ', '.join(sorted(unknown)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / 'results.json'
    results = json.loads(results_path.read_text()) if args.skip_passed and results_path.exists() else []
    passed = {row['fixture'] for row in results if row['status'] == 'passed'}
    failures = []
    for name, fixture in fixtures():
        if args.fixture and name not in args.fixture:
            continue
        if name in passed:
            continue
        model, manifest = write(fixture, args.output_dir, name)
        if args.generate_only:
            print(name, flush=True)
            continue
        stem = args.output_dir / name
        log = args.output_dir / (name + '.log')
        commands = [
            ['compile', '--model=' + str(model), '--output_dir=' + str(stem),
             '--report_json=' + str(stem) + '.compile.json'],
            ['validate', '--model=' + str(model), '--tensor_inputs=' + str(manifest),
             '--max_samples=3', '--allow_emulation=' + str(int(args.allow_emulation)),
             '--vulkan_validation=' + args.vulkan_validation, '--max_code_error=1',
             '--mean_code_error=0.5', '--report_json=' + str(stem) + '.validate.json'],
        ]
        # These two INT8 types have no local TFLite CPU kernel. Use independent
        # mathematical references rather than treating a missing oracle as a pass.
        if name == 'SQUARE_INT8' or name.startswith('CONV3D_INT8_'):
            commands[1].append('--reference=fp64')
        compile_only = name.startswith('RFFT2D_')
        if compile_only:
            commands = commands[:1]
        failed = False
        with log.open('w') as output:
            for command in commands:
                result = subprocess.run([args.tool, *command], stdout=output,
                                        stderr=subprocess.STDOUT, check=False)
                if result.returncode:
                    failures.append(name + ' ' + command[0] + '; see ' + str(log))
                    failed = True
                    break
            report = json.loads(Path(str(stem) + '.compile.json').read_text())
            if not failed and not report['complete_coverage']:
                failures.append('Partial coverage: ' + name)
                failed = True
        status = 'failed' if failed else 'compile_only_requires_EXT_FFT' if compile_only else 'passed'
        results = [row for row in results if row['fixture'] != name]
        results.append(dict(fixture=name, status=status))
        results_path.write_text(json.dumps(results, indent=2) + '\n')
        print(name, status, flush=True)
    if failures:
        raise RuntimeError('\n'.join(failures))


if __name__ == '__main__':
    main()
