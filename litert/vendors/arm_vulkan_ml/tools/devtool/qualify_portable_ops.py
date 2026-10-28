# Copyright 2026 The LiteRT Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
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
from tflite.BuiltinOptions import BuiltinOptions
import argparse
import os
parser = argparse.ArgumentParser(description="Generate bounded FP32 fixtures and qualify them with the developer tool")
parser.add_argument('--output_dir', type=Path, required=True)
parser.add_argument('--tool', required=True)
parser.add_argument('--allow_emulation', action='store_true')
args = parser.parse_args()
out=args.output_dir
out.mkdir(parents=True, exist_ok=True)
env=os.environ.copy()
tool=args.tool
failures=[]
def fixture(name, shapes, outputs, constants=None, options=None, label=None, supported=True):
 opcode_name=name;name=label or name
 m=ModelT();m.version=3;m.buffers=[BufferT()];g=SubGraphT();g.tensors=[];g.inputs=[];g.outputs=[];op=OperatorT();op.opcodeIndex=0;op.inputs=[]
 for i,s in enumerate(shapes):
  t=TensorT();t.shape=s;t.type=TensorType.FLOAT32;t.buffer=0
  if constants and i in constants:
   data=constants[i];t.type=TensorType.INT32 if data.dtype==np.int32 else TensorType.FLOAT32;t.buffer=len(m.buffers);b=BufferT();b.data=data.tobytes();m.buffers.append(b)
  else:g.inputs.append(i)
  g.tensors.append(t);op.inputs.append(i)
 for s in outputs:
  t=TensorT();t.shape=s;t.type=TensorType.FLOAT32;t.buffer=0;g.outputs.append(len(g.tensors));g.tensors.append(t)
 op.outputs=g.outputs
 if options:
  typ,values=options
  mod=__import__('tflite.'+typ,fromlist=[typ+'T']);o=getattr(mod,typ+'T')()
  for k,v in values.items():setattr(o,k,v)
  op.builtinOptionsType=getattr(BuiltinOptions,typ);op.builtinOptions=o
 g.operators=[op];m.subgraphs=[g];c=OperatorCodeT();c.builtinCode=getattr(BuiltinOperator,opcode_name);c.deprecatedBuiltinCode=min(c.builtinCode,127);m.operatorCodes=[c]
 b=flatbuffers.Builder(1024);b.Finish(m.Pack(b),file_identifier=b'TFL3');path=out/(name+'.tflite');path.write_bytes(b.Output())
 cases=[]
 for j in range(4):
  inputs=[]
  for slot,i in enumerate(g.inputs):
   x=np.random.default_rng(j*17+i).uniform(.1+j*.1,2+j,shapes[i]).astype('<f4');
   if j==1 and name not in ['LOG','RSQRT','SQRT','REDUCE_PROD']:x=-x
   if j==2 and name not in ['LOG','RSQRT','DIV']:x.fill(0)
   if j==3 and name in ['SOFTMAX','MAXIMUM','MINIMUM']:x.fill(1)
   f=out/f'{name}_{j}_{slot}.bin';f.write_bytes(x.tobytes());inputs.append(dict(index=slot,file=f.name))
  cases.append(dict(id=str(j),inputs=inputs))
 manifest=out/(name+'.inputs.json');manifest.write_text(json.dumps(dict(cases=cases)))
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
   for command in ['inspect','capabilities']:
    check=subprocess.run([tool,command,'--model='+str(path),'--allow_emulation='+str(int(args.allow_emulation)),'--report_json='+str(out/(name+'.'+command+'.json'))],env=env,stdout=log,stderr=subprocess.STDOUT)
    if check.returncode:failures.append(name+':'+command)
  if r.returncode==0:r=subprocess.run([tool,'validate','--model='+str(path),'--tensor_inputs='+str(manifest),'--max_samples=4','--allow_emulation='+str(int(args.allow_emulation)),'--report_json='+str(out/(name+'.result.json'))],env=env,stdout=log,stderr=subprocess.STDOUT)
  if r.returncode==0:
   for mode in ['accumulated','isolated']:
    trace=subprocess.run([tool,'trace','--model='+str(path),'--tensor_inputs='+str(manifest),'--mode='+mode,'--allow_emulation='+str(int(args.allow_emulation)),'--report_json='+str(out/(name+'.'+mode+'.json'))],env=env,stdout=log,stderr=subprocess.STDOUT)
    if trace.returncode:failures.append(name+':'+mode)
 print(name,r.returncode,flush=True)
 if r.returncode:failures.append(name)
for name in ['RELU','RELU6','ABS','NEG','SQUARE','SQRT','RSQRT','EXP','LOG','FLOOR','CEIL','TANH','SOFTMAX']:
 fixture(name,[[2,3]],[[2,3]],options=('SoftmaxOptions',{'beta':.7}) if name=='SOFTMAX' else None)
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
