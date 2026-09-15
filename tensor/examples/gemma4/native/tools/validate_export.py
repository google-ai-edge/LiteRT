#!/usr/bin/env python3
# Copyright 2026 Google LLC.
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

"""Independent readback of all exported bytes/codes, scales, and graph coverage."""
from pathlib import Path
from collections import Counter
import hashlib,json,mmap,re,sys,time
import numpy as np
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--schema-dir', type=Path, required=True,
                    help='Generated TFLite Python package parent (contains tflite/Model.py).')
parser.add_argument('--inventory', type=Path, required=True)
parser.add_argument('--trace', type=Path, required=True)
parser.add_argument('--kv', type=Path, required=True)
parser.add_argument('--bundle-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if not (args.schema_dir / 'tflite/Model.py').is_file():
    parser.error('--schema-dir must contain generated tflite/Model.py')
sys.path.insert(0, str(args.schema_dir.resolve()))

from tflite.Model import Model
ROOT=args.bundle_dir.resolve();manifest=json.loads((ROOT/'manifest.json').read_text());assert manifest['status']=='complete' and not manifest['unmapped_float_coefficients']
p=json.loads(args.inventory.read_text());trace=json.loads(args.trace.read_text())
if args.output.exists(): raise FileExistsError(args.output)
f=open(p['path'],'rb');mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ);sections={s['index']:s for s in p['sections'] if s['type']=='TFLiteModel'};models={i:Model.GetRootAsModel(memoryview(mm)[s['begin']:s['end']],0) for i,s in sections.items()}
assert hashlib.sha256(mm).hexdigest() == manifest['source_bundle']['sha256']
B=4*1024*1024;stats=Counter();start=time.time();byname={x['name']:x for x in manifest['tensors']};assert len(byname)==1093

def digest_file(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while chunk:=f.read(B):h.update(chunk)
 return h.hexdigest()
def source_tensor(src):return models[src['section_index']].Subgraphs(src['subgraph_index']).Tensors(src['tensor_index'])
def source_data(src):
 t=source_tensor(src);return models[src['section_index']].Buffers(t.Buffer()).DataAsNumpy()
def verify_file(record):
 path=(ROOT/record['file']).resolve();assert path.is_relative_to(ROOT)
 assert path.stat().st_size==record['bytes']
 assert digest_file(path)==record['sha256'],path
 stats['files_checked']+=1;stats['file_bytes_checked']+=record['bytes']
 return path

for r in manifest['tensors']:
 path=verify_file(r);count=int(np.prod(r['shape'],dtype=np.int64))
 expected=count*4 if r['dtype']=='float32' else count if r['dtype']=='int8' else (count+1)//2
 assert path.stat().st_size==expected,(r['name'],expected,path.stat().st_size)
 if 'quantization' not in r:
  a=np.memmap(path,mode='r',dtype='<f4');assert np.isfinite(a).all()
  s=r['sources'][0]
  src=source_tensor(s).Quantization().ScaleAsNumpy() if s.get('field')=='quantization.scale' else source_data(s).view('<f4')
  assert np.array_equal(a.view('u4'),src.view('u4').reshape(-1)),r['name']
  stats['float_coefficients_or_activation_scales_checked']+=len(a);continue
 q=r['quantization'];sp=ROOT/q['scales_file'];assert sp.stat().st_size==q['scales_bytes'];assert digest_file(sp)==q['scales_sha256'];stats['files_checked']+=1;stats['file_bytes_checked']+=q['scales_bytes'];scales=np.memmap(sp,mode='r',dtype='<f4',shape=tuple(q['scales_shape']));assert np.isfinite(scales).all() and np.all(scales>0);assert q['zero_points']==[0] and q['quantized_dimension']==0
 if q['kind']=='per_channel':
  s=r['sources'][0];src=source_data(s);t=source_tensor(s);assert np.array_equal(scales.view('u4'),t.Quantization().ScaleAsNumpy().view('u4'));assert np.all(t.Quantization().ZeroPointAsNumpy()==0);out=np.memmap(path,mode='r',dtype='u1')
  if s['dtype']=='INT2':
   # Decode source and destination independently into signed int8 values.
   for begin in range(0,len(src),B):
    sb=src[begin:begin+B];ob=out[begin*2:(begin+len(sb))*2]
    sx=np.stack([((sb>>shift)&3).astype(np.int8) for shift in [0,2,4,6]],axis=1).reshape(-1);sx[sx>=2]-=4
    ox=np.stack([(ob&15).astype(np.int8),(ob>>4).astype(np.int8)],axis=1).reshape(-1);ox[ox>=8]-=16
    assert np.array_equal(sx,ox),r['name']
   stats['independently_decoded_int2_codes']+=count
  else:assert np.array_equal(out,src),r['name']
  assert hashlib.sha256(src).hexdigest()==s['source_data_sha256']
 else:
  assert q['kind']=='blockwise' and q['block_size']==256 and r['shape']==[262144,8960] and len(r['sources'])==35
  out=np.memmap(path,mode='r',dtype='u1',shape=(262144,35,128));parts=[]
  for layer,s in enumerate(r['sources']):
   assert s['destination_layer_partition']==layer;src=source_data(s);assert hashlib.sha256(src).hexdigest()==s['source_data_sha256'];parts.append(src.reshape(262144,128));assert np.array_equal(scales[:,layer].view('u4'),source_tensor(s).Quantization().ScaleAsNumpy().view('u4'))
  for begin in range(0,262144,4096):
   block=np.asarray(out[begin:begin+4096])
   for layer,src in enumerate(parts):assert np.array_equal(block[:,layer,:],src[begin:begin+4096]),layer
  stats['combined_embedding_packed_codes_checked']+=count
 stats['weight_codes_checked']+=count;stats['weight_scales_checked']+=scales.size
for c in manifest['constants']:
 path=verify_file(c);out=np.fromfile(path,dtype='u1');assert np.array_equal(out,source_data(c['sources'][0]));stats['fixed_constants_checked']+=1
assert stats['weight_codes_checked']==5030936576 and stats['independently_decoded_int2_codes']==1937768448
# Confirm other bundle signatures use the same active FC coefficients/scales.
def name_for_fc(path):
 if 'per_layer_model_projection/' in path:return 'model.per_layer_model_projection'
 if 'decode_softmax/' in path:return 'lm_head'
 layer=int(re.search(r'/layer_(\d+)/',path).group(1));pats=[('/q_einsum/','self_attn.q_proj'),('/k_einsum/','self_attn.k_proj'),('/v_einsum/','self_attn.v_proj'),('/attn_vec_einsum/','self_attn.o_proj'),('/gating_einsum1/','mlp.gate_proj'),('/gating_einsum2/','mlp.up_proj'),('/mlp/linear/','mlp.down_proj'),('/per_layer_embedding_gate/','per_layer_input_gate'),('/per_layer_embedding_projection/','per_layer_projection')]
 suffix=next(s for p,s in pats if p in path);return f'model.layers.{layer}.{suffix}'
cross=[]
for g in trace['graphs']:
 counter=0
 for op in g['operators']:
  if op['type']!='FULLY_CONNECTED':continue
  w=op['inputs'][1];name=name_for_fc(w['name']);rec=byname[name+'.weight'];src={'section_index':10,'subgraph_index':g['index'],'tensor_index':w['index']};data=source_data(src);assert hashlib.sha256(data).hexdigest()==rec['sources'][0]['source_data_sha256'],(g['name'],name)
  assert np.array_equal(source_tensor(src).Quantization().ScaleAsNumpy().view('u4'),np.fromfile(ROOT/rec['quantization']['scales_file'],dtype='<u4'))
  for role,t in [('input',op['inputs'][0]),('output',op['outputs'][0])]:
   if 'scales' in t:assert np.array_equal(np.fromfile(ROOT/byname[name+'.'+role+'_scale']['file'],dtype='<f4'),np.array(t['scales'],dtype='<f4'))
   else:assert name+'.'+role+'_scale' not in byname
  counter+=1
 cross.append({'signature':g['name'],'fc_weight_and_scale_arrays_exact':counter})
kv=json.loads(args.kv.read_text())['graphs'][0]['cache_updates'];assert len(kv)==15
for a,b in zip(manifest['kv_cache_specs'],kv):assert a['owner']==b['owner'] and a['key_scale']==b['key_scale'] and a['value_scale']==b['value_scale'] and a['zero_point']==0
result={'status':'pass','manifest_sha256':digest_file(ROOT/'manifest.json'),'validator_sha256':digest_file(Path(__file__)),'counters':dict(stats),'cross_signature_fc_validation':cross,'unmapped_float_coefficients':0,'seconds':time.time()-start,'model_runs':0}
args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
