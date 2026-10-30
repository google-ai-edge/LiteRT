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
"""Requires flatbuffers, numpy and TFLite Python bindings with object APIs.

Configure Vulkan emulation through the environment before invoking this script.
Generated models, raw inputs and reports belong in an ignored output directory.
"""
from pathlib import Path
import json
import subprocess
import flatbuffers
import numpy as np
from tflite.Model import ModelT
from tflite.SubGraph import SubGraphT
from tflite.Tensor import TensorT
from tflite.Buffer import BufferT
from tflite.Operator import OperatorT
from tflite.OperatorCode import OperatorCodeT
from tflite.BuiltinOperator import BuiltinOperator
from tflite.TensorType import TensorType
from tflite.QuantizationParameters import QuantizationParametersT
from tflite.BuiltinOptions import BuiltinOptions
import argparse
import os
parser = argparse.ArgumentParser(description="Generate bounded FP32 fixtures and qualify them with the developer tool")
parser.add_argument('--output_dir', type=Path, required=True)
parser.add_argument('--tool', required=True)
parser.add_argument('--quantized_only', action='store_true')
parser.add_argument('--allow_emulation', action='store_true')
parser.add_argument('--vulkan_validation', choices=['off','core','sync'], default='off')
args = parser.parse_args()
out=args.output_dir
out.mkdir(parents=True, exist_ok=True)
env=os.environ.copy()
tool=args.tool
failures=[]
database=Path(__file__).resolve().parents[1]/'tosa_capabilities/tosa_001000_1.json'
if not database.exists():raise SystemExit('Generate the pinned TOSA database first')
opcodes={op['name']:op['opcode'] for op in json.loads(database.read_text())['operators']}
expected_tosa={name:[name] for name in ['SIN','COS','POW','LOGICAL_AND','LOGICAL_OR','LOGICAL_NOT','REDUCE_ALL','REDUCE_ANY','BITWISE_XOR']}
expected_tosa.update({'POW_POSITIVE':['POW'],'GELU':['ERF'],'GELU_APPROXIMATE':['TANH'],'REVERSE_V2':['REVERSE'],'CONV_3D':['CONV3D'],'ARG_MAX':['ARGMAX'],'NOT_EQUAL_BOOL':['LOGICAL_XOR'],'EQUAL_FP32':['EQUAL'],'BITWISE_NOT_IDIOM':['BITWISE_NOT'],'RIGHT_SHIFT':['LOGICAL_RIGHT_SHIFT'],'FLOOR_DIV':['INTDIV'],'SCATTER_ND':['SCATTER']})
def fixture(name, shapes, outputs, constants=None, options=None, label=None, supported=True, input_types=None, output_type=None, quantization=None, code_error=0):
 opcode_name=name;name=label or name
 m=ModelT();m.version=3;m.buffers=[BufferT()];g=SubGraphT();g.tensors=[];g.inputs=[];g.outputs=[];op=OperatorT();op.opcodeIndex=0;op.inputs=[]
 for i,s in enumerate(shapes):
  t=TensorT();t.shape=s;t.type=input_types[i] if input_types else TensorType.FLOAT32;t.buffer=0
  if constants and i in constants:
   data=constants[i];t.type=TensorType.INT32 if data.dtype==np.int32 else TensorType.UINT32 if data.dtype==np.uint32 else TensorType.INT8 if data.dtype==np.int8 else TensorType.FLOAT32;t.buffer=len(m.buffers);b=BufferT();b.data=data.tobytes();m.buffers.append(b)
  else:g.inputs.append(i)
  g.tensors.append(t);op.inputs.append(i)
 for s in outputs:
  t=TensorT();t.shape=s;t.type=output_type if output_type is not None else TensorType.FLOAT32;t.buffer=0;g.outputs.append(len(g.tensors));g.tensors.append(t)
 if quantization is not None:
  for i,t in enumerate(g.tensors):
   if t.type==TensorType.INT8 or i in quantization:
    scales,zps,axis=quantization.get(i,([.125],[-7],0));q=QuantizationParametersT();q.scale=scales;q.zeroPoint=zps;q.quantizedDimension=axis;t.quantization=q
 op.outputs=g.outputs
 if options:
  typ,values=options
  mod=__import__('tflite.'+typ,fromlist=[typ+'T']);o=getattr(mod,typ+'T')()
  for k,v in values.items():setattr(o,k,v)
  op.builtinOptionsType=getattr(BuiltinOptions,typ);op.builtinOptions=o
 g.operators=[op];m.subgraphs=[g];c=OperatorCodeT();c.builtinCode=getattr(BuiltinOperator,opcode_name);c.deprecatedBuiltinCode=min(c.builtinCode,127);c.version=3 if opcode_name=='TRANSPOSE_CONV' and quantization is not None else 2 if quantization is not None and opcode_name!='EXPAND_DIMS' else 1;m.operatorCodes=[c]
 b=flatbuffers.Builder(1024);b.Finish(m.Pack(b),file_identifier=b'TFL3');path=out/(name+'.tflite');path.write_bytes(b.Output())
 cases=[]
 for j in range(4):
  inputs=[]
  for slot,i in enumerate(g.inputs):
   x=np.random.default_rng(j*17+i).uniform(.1+j*.1,2+j,shapes[i]).astype('<f4');
   if j==1 and name not in ['LOG','RSQRT','SQRT','REDUCE_PROD','POW_POSITIVE']:x=-x
   if j==2 and name not in ['LOG','RSQRT','DIV','POW_POSITIVE']:x.fill(0)
   if j==3 and name in ['SOFTMAX','MAXIMUM','MINIMUM']:x.fill(1)
   if g.tensors[i].type==TensorType.INT8:x=(np.resize(np.arange(-128,128,dtype=np.int16),shapes[i]).astype(np.int8) if j==0 else np.random.default_rng(j*17+i).integers(-128,128,shapes[i],dtype=np.int8))
   if g.tensors[i].type==TensorType.BOOL:x=(np.random.default_rng(j*17+i).integers(0,2,shapes[i])).astype(np.uint8)
   if g.tensors[i].type==TensorType.INT32:x=np.random.default_rng(j*17+i).integers(-100,101,shapes[i],dtype=np.int32)
   if g.tensors[i].type==TensorType.INT32 and j==0:x.flat[:min(x.size,4)]=np.array([-2147483648,2147483647,0,-1],np.int32)[:min(x.size,4)]
   if g.tensors[i].type==TensorType.UINT32:x=np.random.default_rng(j*17+i).integers(0,2**32,shapes[i],dtype=np.uint32)
   f=out/f'{name}_{j}_{slot}.bin';f.write_bytes(x.tobytes());inputs.append(dict(index=slot,file=f.name))
  cases.append(dict(id=str(j),inputs=inputs))
 manifest=out/(name+'.inputs.json');manifest.write_text(json.dumps(dict(cases=cases)))
 # These paths use native SINGLE_ROUND RESCALE rather than reproducing the
 # reference kernel's double rounding. Keep exact gates on all other routes.
 if quantization is not None and opcode_name in ['SUB','TRANSPOSE_CONV']:
  code_error=max(code_error,1)
 gates=['--max_code_error='+str(code_error),'--mean_code_error='+str(code_error)] if quantization is not None else []
 command_args=[tool,'compile','--model='+str(path),'--output_dir='+str(out/name),'--report_json='+str(out/(name+'.compile.json'))]
 with (out/(name+'.log')).open('w') as log:
  r=subprocess.run(command_args,env=env,stdout=log,stderr=subprocess.STDOUT)
  if r.returncode==0:
   report=json.loads((out/(name+'.compile.json')).read_text())
   if not supported:
    if report.get('complete_coverage'):failures.append(name)
    print(name,'rejected' if not report.get('complete_coverage') else 'UNEXPECTED COVERAGE',flush=True)
    return
   if not report.get('complete_coverage'):failures.append(name);return
   emitted={opcode for sg in report['subgraphs'] for part in sg['partitions'] for opcode in part['tosa_opcodes']}
   expected_ops=({'TANH':['TABLE'],'ABS':['TABLE'],'RESIZE_BILINEAR':['RESIZE'],'TRANSPOSE_CONV':['TRANSPOSE_CONV2D'],'AVERAGE_POOL_2D':['AVG_POOL2D']}.get(opcode_name,[]) if quantization is not None else expected_tosa.get(name,expected_tosa.get(opcode_name,[])))
   for expected in expected_ops:
    if opcodes[expected] not in emitted:failures.append(name+':missing-'+expected);return
   for command in ['inspect','capabilities']:
    check=subprocess.run([tool,command]+(['--vulkan_validation='+args.vulkan_validation] if command!='inspect' else [])+['--model='+str(path),'--allow_emulation='+str(int(args.allow_emulation)),'--report_json='+str(out/(name+'.'+command+'.json'))],env=env,stdout=log,stderr=subprocess.STDOUT)
    if check.returncode:failures.append(name+':'+command)
  if r.returncode==0:r=subprocess.run([tool,'validate']+gates+['--vulkan_validation='+args.vulkan_validation,'--model='+str(path),'--tensor_inputs='+str(manifest),'--max_samples=4','--allow_emulation='+str(int(args.allow_emulation)),'--report_json='+str(out/(name+'.result.json'))],env=env,stdout=log,stderr=subprocess.STDOUT)
  if r.returncode==0:
   for mode in ['accumulated','isolated']:
    trace=subprocess.run([tool,'trace']+gates+['--vulkan_validation='+args.vulkan_validation,'--model='+str(path),'--tensor_inputs='+str(manifest),'--mode='+mode,'--allow_emulation='+str(int(args.allow_emulation)),'--report_json='+str(out/(name+'.'+mode+'.json'))],env=env,stdout=log,stderr=subprocess.STDOUT)
    if trace.returncode:failures.append(name+':'+mode)
 print(name,r.returncode,flush=True)
 if r.returncode:failures.append(name)
if args.quantized_only:
 # Native pooling may differ from LiteRT by one code at rounding boundaries.
 # Include Inception's overlapping windows, unused trailing input and RELU6.
 for zp in [-128, -7, 127]:
  for cropped in [False, True]:
   source=[1,6,7,2];stride=2 if cropped else 1
   target=[1,(source[1]-3)//stride+1,(source[2]-3)//stride+1,2]
   fixture('AVERAGE_POOL_2D',[source],[target],
    options=('Pool2DOptions',{'padding':1,'filterHeight':3,'filterWidth':3,'strideH':stride,'strideW':stride,'fusedActivationFunction':3 if cropped else 0}),
    label='AVG_POOL_INT8_'+str(zp)+'_'+str(cropped),input_types=[TensorType.INT8],output_type=TensorType.INT8,
    quantization={0:([.125],[zp],0),1:([.125],[zp],0)},code_error=1)
 for scale,zp in [(.03125,-7),(.0001,0),(2.0,27)]:
  fixture('TANH',[[1,256]],[[1,256]],label='TANH_INT8_'+str(scale),input_types=[TensorType.INT8],output_type=TensorType.INT8,quantization={0:([scale],[zp],0),1:([1/128],[0],0)})
 fixture('ABS',[[1,256]],[[1,256]],label='ABS_INT8',input_types=[TensorType.INT8],output_type=TensorType.INT8,quantization={})
 fixture('ABS',[[1,256]],[[1,256]],label='ABS_INT8_RESCALE',input_types=[TensorType.INT8],output_type=TensorType.INT8,quantization={0:([.0837],[31],0),1:([.1342],[-17],0)})
 for name in ['MAXIMUM','MINIMUM','SUB']:
  fixture(name,[[2,3],[1,3]],[[2,3]],label=name+'_INT8',options=('SubOptions',{}) if name=='SUB' else None,input_types=[TensorType.INT8]*2,output_type=TensorType.INT8,quantization={})
 fixture('SUB',[[2,3],[1,3]],[[2,3]],label='SUB_INT8_SCALES',options=('SubOptions',{'fusedActivationFunction':3}),input_types=[TensorType.INT8]*2,output_type=TensorType.INT8,quantization={0:([.0625],[11],0),1:([.125],[-5],0),2:([.03125],[7],0)})
 fixture('SUB',[[256,1],[1,256]],[[256,256]],label='SUB_INT8_EXHAUSTIVE',options=('SubOptions',{}),input_types=[TensorType.INT8]*2,output_type=TensorType.INT8,quantization={0:([.0837],[31],0),1:([.0213],[-11],0),2:([.1342],[-17],0)})
 for name,shapes,outputs,constants,opts in [
  ('SQUEEZE',[[1,2,3]],[[2,3]],{},('SqueezeOptions',{'squeezeDims':[0]})),
  ('EXPAND_DIMS',[[2,3],[1]],[[1,2,3]],{1:np.array([0],np.int32)},None),
  ('TILE',[[2,3],[2]],[[2,6]],{1:np.array([1,2],np.int32)},None),
  ('PACK',[[2,3],[2,3]],[[2,2,3]],{},('PackOptions',{'axis':0,'valuesCount':2})),
  ('UNPACK',[[2,3]],[[3],[3]],{},('UnpackOptions',{'axis':0,'num':2})),
  ('SPLIT',[[1],[2,4]],[[2,2],[2,2]],{0:np.array([1],np.int32)},('SplitOptions',{'numSplits':2})),
  ('SPLIT_V',[[2,4],[2],[1]],[[2,1],[2,3]],{1:np.array([1,3],np.int32),2:np.array([1],np.int32)},('SplitVOptions',{'numSplits':2})),
  ('STRIDED_SLICE',[[3,4],[2],[2],[2]],[[2,2]],{1:np.array([0,0],np.int32),2:np.array([3,4],np.int32),3:np.array([2,2],np.int32)},('StridedSliceOptions',{})),
  ('REVERSE_V2',[[2,3],[2]],[[2,3]],{1:np.array([0,-1],np.int32)},None)]:
  fixture(name,shapes,outputs,constants,opts,label=name+'_INT8',input_types=[TensorType.INT8]*len(shapes),output_type=TensorType.INT8,quantization={})
 fixture('TANH',[[1,256]],[[1,256]],label='REJECT_TANH_QUANT',input_types=[TensorType.INT8],output_type=TensorType.INT8,quantization={},supported=False)
 fixture('TILE',[[2,3],[2]],[[2,6]],{1:np.array([1,2],np.int32)},label='REJECT_TILE_QUANT',input_types=[TensorType.INT8,TensorType.INT32],output_type=TensorType.INT8,quantization={2:([.25],[-7],0)},supported=False)
 fixture('MAXIMUM',[[2,3],[1,3]],[[2,3]],label='REJECT_MAXIMUM_QUANT',input_types=[TensorType.INT8]*2,output_type=TensorType.INT8,quantization={1:([.25],[-7],0)},supported=False)
 for source,target in [([1,2,3,2],[1,4,6,2]),([1,5,4,2],[1,3,2,2]),([1,1,3,1],[1,4,5,1])]:
  for align,half in [(False,False),(False,True),(True,False)]:
   if align and (source[1]==1 or source[2]==1):continue
   fixture('RESIZE_BILINEAR',[source,[2]],[target],{1:np.array(target[1:3],np.int32)},('ResizeBilinearOptions',{'alignCorners':align,'halfPixelCenters':half}),label='RESIZE_INT8_'+str(source[1])+'_'+str(align)+'_'+str(half),input_types=[TensorType.INT8,TensorType.INT32],output_type=TensorType.INT8,quantization={})
 weights=np.arange(-9,9,dtype=np.int8).reshape(2,3,3,1)
 for padding,size in [(0,4),(1,5)]:
  for bias in [False,True]:
   constants={0:np.array([1,size,size,2],np.int32),1:weights}
   shapes=[[4],[2,3,3,1],[1,2,2,1]];types=[TensorType.INT32,TensorType.INT8,TensorType.INT8]
   q={1:([.0625,.125],[0,0],0),2:([.125],[-7],0)}
   if bias:shapes.append([2]);types.append(TensorType.INT32);constants[3]=np.array([3,-5],np.int32);q[3]=([.0078125,.015625],[0,0],0)
   q[len(shapes)]=([.0625],[-3],0)
   fixture('TRANSPOSE_CONV',shapes,[[1,size,size,2]],constants,('TransposeConvOptions',{'padding':padding,'strideH':2,'strideW':2}),label='TRANSPOSE_CONV_INT8_'+str(padding)+'_'+str(bias),input_types=types,output_type=TensorType.INT8,quantization=q)
 q={1:([.0625],[0],0),2:([.125],[-7],0),3:([.0078125],[0],0),4:([.0625],[-3],0)}
 fixture('TRANSPOSE_CONV',[[4],[2,3,3,1],[1,2,2,1],[2]],[[1,4,4,2]],{0:np.array([1,4,4,2],np.int32),1:(weights*10).astype(np.int8),3:np.array([30000,-30000],np.int32)},('TransposeConvOptions',{'padding':0,'strideH':2,'strideW':2,'fusedActivationFunction':3}),label='TRANSPOSE_CONV_INT8_PER_TENSOR_RELU6',input_types=[TensorType.INT32,TensorType.INT8,TensorType.INT8,TensorType.INT32],output_type=TensorType.INT8,quantization=q)
 fixture('TRANSPOSE_CONV',[[4],[2,3,3,1],[1,2,2,1],[2]],[[1,4,4,2]],{0:np.array([1,4,4,2],np.int32),1:weights,3:np.array([2147483647,0],np.int32)},('TransposeConvOptions',{'padding':0,'strideH':2,'strideW':2}),label='REJECT_TRANSPOSE_ACCUMULATOR',input_types=[TensorType.INT32,TensorType.INT8,TensorType.INT8,TensorType.INT32],output_type=TensorType.INT8,quantization=q,supported=False)
 q[4]=([1e-30],[-3],0)
 fixture('TRANSPOSE_CONV',[[4],[2,3,3,1],[1,2,2,1],[2]],[[1,4,4,2]],{0:np.array([1,4,4,2],np.int32),1:weights,3:np.array([0,0],np.int32)},('TransposeConvOptions',{'padding':0,'strideH':2,'strideW':2}),label='REJECT_TRANSPOSE_RESCALE',input_types=[TensorType.INT32,TensorType.INT8,TensorType.INT8,TensorType.INT32],output_type=TensorType.INT8,quantization=q,supported=False)
 # Reject inconsistent per-channel bias scaling before emission.
 q={1:([.0625,.125],[0,0],0),2:([.125],[-7],0),3:([.5,.5],[0,0],0),4:([.0625],[-3],0)}
 fixture('TRANSPOSE_CONV',[[4],[2,3,3,1],[1,2,2,1],[2]],[[1,4,4,2]],{0:np.array([1,4,4,2],np.int32),1:weights,3:np.array([3,-5],np.int32)},('TransposeConvOptions',{'padding':0,'strideH':2,'strideW':2}),label='REJECT_TRANSPOSE_BIAS_QUANT',input_types=[TensorType.INT32,TensorType.INT8,TensorType.INT8,TensorType.INT32],output_type=TensorType.INT8,quantization=q,supported=False)

else:
 for name in ['LOGICAL_AND','LOGICAL_OR','NOT_EQUAL','EQUAL']:
  fixture(name,[[2,3],[1,3]],[[2,3]],label=name+'_BOOL',input_types=[TensorType.BOOL]*2,output_type=TensorType.BOOL)
 fixture('LOGICAL_NOT',[[2,3]],[[2,3]],input_types=[TensorType.BOOL],output_type=TensorType.BOOL)
 for name in ['REDUCE_ALL','REDUCE_ANY']:
  fixture(name,[[2,3],[1]],[[2]],{1:np.array([1],np.int32)},('ReducerOptions',{}),input_types=[TensorType.BOOL,TensorType.INT32],output_type=TensorType.BOOL)
 fixture('ARG_MAX',[[2,3],[1]],[[2]],{1:np.array([1],np.int32)},('ArgMaxOptions',{'outputType':TensorType.INT32}),output_type=TensorType.INT32)
 fixture('BITWISE_XOR',[[2,3],[1,3]],[[2,3]],input_types=[TensorType.INT32]*2,output_type=TensorType.INT32)
 fixture('BITWISE_XOR',[[2,3],[1]],[[2,3]],{1:np.array([-1],np.int32)},label='BITWISE_NOT_IDIOM',input_types=[TensorType.INT32]*2,output_type=TensorType.INT32)
 fixture('RIGHT_SHIFT',[[2,3],[1,3]],[[2,3]],{1:np.array([[0,1,31]],np.uint32)},input_types=[TensorType.UINT32]*2,output_type=TensorType.UINT32)
 fixture('FLOOR_DIV',[[2,3],[1,3]],[[2,3]],{1:np.array([[1,3,7]],np.int32)},input_types=[TensorType.INT32]*2,output_type=TensorType.INT32)
 fixture('SCATTER_ND',[[2,1],[2,2],[2]],[[3,2]],{0:np.array([[0],[2]],np.int32),2:np.array([3,2],np.int32)})
 fixture('EQUAL',[[2,3],[1,3]],[[2,3]],label='EQUAL_FP32',output_type=TensorType.BOOL)
 fixture('FLOOR_DIV',[[2,3],[1]],[[2,3]],{1:np.array([0],np.int32)},label='REJECT_DIV_ZERO',supported=False,input_types=[TensorType.INT32]*2,output_type=TensorType.INT32)
 fixture('RIGHT_SHIFT',[[2,3],[1]],[[2,3]],{1:np.array([32],np.uint32)},label='REJECT_SHIFT_RANGE',supported=False,input_types=[TensorType.UINT32]*2,output_type=TensorType.UINT32)
 fixture('SCATTER_ND',[[2,1],[2,2],[2]],[[3,2]],{0:np.array([[0],[0]],np.int32),2:np.array([3,2],np.int32)},label='REJECT_SCATTER_DUPLICATE',supported=False)
 for name in ['RELU','RELU6','ABS','NEG','SQUARE','SQRT','RSQRT','EXP','LOG','FLOOR','CEIL','TANH','SIN','COS','SOFTMAX']:
  fixture(name,[[2,3]],[[2,3]],options=('SoftmaxOptions',{'beta':.7}) if name=='SOFTMAX' else None)
 fixture('POW',[[2,3],[1,3]],[[2,3]],label='POW_POSITIVE')
 fixture('GELU',[[2,3]],[[2,3]],options=('GeluOptions',{'approximate':False}))
 fixture('GELU',[[2,3]],[[2,3]],options=('GeluOptions',{'approximate':True}),label='GELU_APPROXIMATE')
 fixture('REVERSE_V2',[[2,3],[2]],[[2,3]],{1:np.array([0,-1],np.int32)})
 fixture('CONV_3D',[[1,3,3,3,2],[2,2,2,2,3],[3]],[[1,2,2,2,3]],{1:np.linspace(-.2,.3,48,dtype=np.float32).reshape(2,2,2,2,3),2:np.array([.1,-.2,.3],np.float32)},('Conv3DOptions',{'padding':1,'strideD':1,'strideH':1,'strideW':1,'dilationDFactor':1,'dilationHFactor':1,'dilationWFactor':1}))
 for name in ['SUB','DIV','MAXIMUM','MINIMUM']:
  fixture(name,[[2,3],[1,3]],[[2,3]],options=(name.title()+'Options',{}) if name in ['SUB','DIV'] else None)
 fixture('SLICE',[[2,3],[2],[2]],[[2,2]],{1:np.array([0,1],np.int32),2:np.array([2,2],np.int32)})
 fixture('SQUEEZE',[[1,2,3]],[[2,3]],options=('SqueezeOptions',{'squeezeDims':[0]}))
 fixture('EXPAND_DIMS',[[2,3],[1]],[[1,2,3]],{1:np.array([0],np.int32)})
 fixture('TILE',[[2,3],[2]],[[2,6]],{1:np.array([1,2],np.int32)})
 fixture('CONCATENATION',[[2,3],[2,3]],[[2,6]],options=('ConcatenationOptions',{'axis':1}))
 fixture('PACK',[[2,3],[2,3]],[[2,2,3]],options=('PackOptions',{'axis':0,'valuesCount':2}))
 fixture('UNPACK',[[2,3]],[[3],[3]],options=('UnpackOptions',{'axis':0,'num':2}))
 fixture('SPLIT',[[1],[2,4]],[[2,2],[2,2]],{0:np.array([1],np.int32)},('SplitOptions',{'numSplits':2}))
 fixture('SPLIT_V',[[2,4],[2],[1]],[[2,1],[2,3]],{1:np.array([1,3],np.int32),2:np.array([1],np.int32)},('SplitVOptions',{'numSplits':2}))
 for name in ['MEAN','REDUCE_PROD']:
  fixture(name,[[2,3],[1]],[[2]],{1:np.array([1],np.int32)},('ReducerOptions',{'keepDims':False}))
 fixture('PADV2',[[2,3],[2,2],[1]],[[4,5]],{1:np.array([[1,1],[1,1]],np.int32),2:np.array([.3],np.float32)})

 fixture('BATCH_MATMUL',[[2,3,4],[1,4,5]],[[2,3,5]],options=('BatchMatMulOptions',{}))
 fixture('STRIDED_SLICE',[[3,4],[2],[2],[2]],[[2,2]],{1:np.array([0,0],np.int32),2:np.array([3,4],np.int32),3:np.array([2,2],np.int32)},('StridedSliceOptions',{}))
 for name in ['MAX_POOL_2D','AVERAGE_POOL_2D']:
  fixture(name,[[1,5,5,2]],[[1,3,3,2]],options=('Pool2DOptions',{'padding':0,'filterHeight':2,'filterWidth':2,'strideH':2,'strideW':2}))

 fixture('BATCH_MATMUL',[[2,4,3],[1,5,4]],[[2,3,5]],options=('BatchMatMulOptions',{'adjX':True,'adjY':True}),label='BATCH_MATMUL_TRANSPOSE')
 fixture('STRIDED_SLICE',[[3,4],[2],[2],[2]],[[2]],{1:np.array([1,0],np.int32),2:np.array([2,4],np.int32),3:np.array([1,2],np.int32)},('StridedSliceOptions',{'shrinkAxisMask':1}),label='STRIDED_SLICE_SHRINK')
 fixture('MEAN',[[2,3,4],[2]],[[1,3,1]],{1:np.array([-1,0],np.int32)},('ReducerOptions',{'keepDims':True}),label='MEAN_AXES')
 fixture('SPLIT_V',[[2,4],[2],[1]],[[2,1],[2,3]],{1:np.array([1,-1],np.int32),2:np.array([-1],np.int32)},('SplitVOptions',{'numSplits':2}),label='SPLIT_V_INFER')
 for name in ['MAX_POOL_2D','AVERAGE_POOL_2D']:
  fixture(name,[[1,5,5,2]],[[1,2,2,2]],options=('Pool2DOptions',{'padding':1,'filterHeight':2,'filterWidth':2,'strideH':2,'strideW':2}),label=name+'_VALID')
 fixture('MEAN',[[2,3],[2]],[[]],{1:np.array([0,1],np.int32)},('ReducerOptions',{}),label='REJECT_SCALAR',supported=False)
 fixture('SUB',[[2,3],[2,2]],[[2,3]],options=('SubOptions',{}),label='REJECT_BROADCAST',supported=False)
 fixture('TILE',[[2,3],[2]],[[2,6]],{1:np.array([1,-2],np.int32)},label='REJECT_TILE',supported=False)
 fixture('STRIDED_SLICE',[[3,4],[2],[2],[2]],[[2,2]],{1:np.array([0,0],np.int32),2:np.array([3,4],np.int32),3:np.array([-1,2],np.int32)},('StridedSliceOptions',{}),label='REJECT_STRIDE',supported=False)
if failures: raise SystemExit("Failed: " + ", ".join(failures))
