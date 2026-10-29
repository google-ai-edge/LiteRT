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
"""Generate the small int8 convolution fixture using flatc object-API bindings.

Usage: PYTHONPATH=<flatc --python --gen-object-api output> python this.py OUTPUT
"""
import struct
import sys
from pathlib import Path
import flatbuffers
from tflite.Model import ModelT
from tflite.SubGraph import SubGraphT
from tflite.Tensor import TensorT
from tflite.Buffer import BufferT
from tflite.Operator import OperatorT
from tflite.OperatorCode import OperatorCodeT
from tflite.Conv2DOptions import Conv2DOptionsT
from tflite.QuantizationParameters import QuantizationParametersT
from tflite.TensorType import TensorType
from tflite.BuiltinOperator import BuiltinOperator
from tflite.BuiltinOptions import BuiltinOptions


def generate(groups=1):
    model = ModelT()
    model.version = 3
    model.description = "Arm Vulkan ML direct lowering test: int8 convolution"
    model.buffers = []
    for data in [b"", bytes([2] * (54 * groups)), struct.pack("<" + "i" * (3 * groups), *([10, -20, 30] * groups))]:
        buffer = BufferT()
        buffer.data = bytearray(data)
        model.buffers.append(buffer)
    graph = SubGraphT()
    graph.name = "main"
    graph.tensors = []
    for name, shape, dtype, buffer, scales, zps in [
        ("input", [1,8,8,2 * groups], TensorType.INT8, 0, [.125], [-14]),
        ("weights", [3 * groups,3,3,2], TensorType.INT8, 1, [.25,.125,1e-12] * groups, [0,0,0] * groups),
        ("bias", [3 * groups], TensorType.INT32, 2, [.03125,.015625,1.25e-13] * groups, [0,0,0] * groups),
        ("output", [1,3,3,3 * groups], TensorType.INT8, 0, [.25], [-128]),
    ]:
        tensor = TensorT()
        tensor.name, tensor.shape, tensor.type, tensor.buffer = name, shape, dtype, buffer
        tensor.quantization = QuantizationParametersT()
        tensor.quantization.scale = scales
        tensor.quantization.zeroPoint = zps
        tensor.quantization.quantizedDimension = 0
        graph.tensors.append(tensor)
    code = OperatorCodeT()
    code.builtinCode = code.deprecatedBuiltinCode = BuiltinOperator.CONV_2D
    code.version = 3
    model.operatorCodes = [code]
    op = OperatorT()
    op.inputs, op.outputs = [0,1,2], [3]
    op.builtinOptionsType = BuiltinOptions.Conv2DOptions
    op.builtinOptions = Conv2DOptionsT()
    op.builtinOptions.padding = 1
    op.builtinOptions.strideH = op.builtinOptions.strideW = 2
    op.builtinOptions.dilationHFactor = op.builtinOptions.dilationWFactor = 1
    op.builtinOptions.fusedActivationFunction = 1
    graph.operators, graph.inputs, graph.outputs = [op], [0], [3]
    model.subgraphs = [graph]
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


if __name__ == "__main__":
    Path(sys.argv[1]).write_bytes(generate(int(sys.argv[2]) if len(sys.argv) > 2 else 1))
