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

"""Export actual published text-model tensors. Read-only source; no inference."""
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
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
if not (args.schema_dir / 'tflite/Model.py').is_file():
    parser.error('--schema-dir must contain generated tflite/Model.py')
sys.path.insert(0, str(args.schema_dir.resolve()))
from tflite.Model import Model
from tflite.TensorType import TensorType
TT={v:k for k,v in vars(TensorType).items() if isinstance(v,int)}
OUT=args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=False)
INV=args.inventory.resolve()
TRACE=args.trace.resolve()
KV=args.kv.resolve()
p=json.loads(INV.read_text());trace=json.loads(TRACE.read_text());dec_trace=next(g for g in trace['graphs'] if g['name']=='decode')
f=open(p['path'],'rb');mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
secs={s['items'].get('model_type'):s for s in p['sections'] if s['type']=='TFLiteModel'}
models={k:Model.GetRootAsModel(memoryview(mm)[s['begin']:s['end']],0) for k,s in secs.items()}
DKEY='tf_lite_prefill_decode';dec=models[DKEY];dg=dec.Subgraphs(0)
for sub in ['tensors','scales','constants']:(OUT/sub).mkdir(exist_ok=True)
CHUNK=8*1024*1024
stats=Counter();records=[];constants=[];exported_main_float=set();names=set();unmapped=[]
start=time.time()

def sha(data):return hashlib.sha256(data).hexdigest()
def filehash(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while block:=f.read(CHUNK):h.update(block)
 return h.hexdigest()
def raw(key,gi,ti):
 model=models[key];t=model.Subgraphs(gi).Tensors(ti);b=model.Buffers(t.Buffer())
 assert b.DataLength()>0,(key,gi,ti,b.Offset(),b.Size())
 return b.DataAsNumpy()
def source(key,gi,ti):
 model=models[key];t=model.Subgraphs(gi).Tensors(ti);data=raw(key,gi,ti)
 return {'section_index':secs[key]['index'],'section_model_type':key,'subgraph_index':gi,'tensor_index':ti,'tensor_name':t.Name().decode(),'buffer_index':int(t.Buffer()),'dtype':TT[t.Type()],'shape':t.ShapeAsNumpy().tolist(),'source_bytes':len(data),'source_data_sha256':sha(data)}
def write_bytes(rel,data):
 path=OUT/rel
 assert not path.exists(),f'Refusing to overwrite export {path}'
 with path.open('wb') as f:
  view=memoryview(data).cast('B')
  for i in range(0,len(view),CHUNK):f.write(view[i:i+CHUNK])
 return {'file':rel,'bytes':path.stat().st_size,'sha256':filehash(path)}
def name_for_fc(wname):
 if 'per_layer_model_projection/' in wname:return 'model.per_layer_model_projection'
 if 'decode_softmax/' in wname:return 'lm_head'
 layer=int(re.search(r'/layer_(\d+)/',wname).group(1))
 pats=[('/q_einsum/','self_attn.q_proj'),('/k_einsum/','self_attn.k_proj'),('/v_einsum/','self_attn.v_proj'),('/attn_vec_einsum/','self_attn.o_proj'),('/gating_einsum1/','mlp.gate_proj'),('/gating_einsum2/','mlp.up_proj'),('/mlp/linear/','mlp.down_proj'),('/per_layer_embedding_gate/','per_layer_input_gate'),('/per_layer_embedding_projection/','per_layer_projection')]
 suffix=next((s for pat,s in pats if pat in wname),None);assert suffix,wname
 return f'model.layers.{layer}.{suffix}'
def add(record):
 assert record['name'] not in names,record['name'];names.add(record['name']);records.append(record)
def export_float(name,key,gi,ti,shape=None):
 t=models[key].Subgraphs(gi).Tensors(ti);assert TT[t.Type()]=='FLOAT32'
 data=raw(key,gi,ti);values=data.view('<f4');assert np.isfinite(values).all(),name
 shape=t.ShapeAsNumpy().tolist() if shape is None else shape
 assert int(np.prod(shape,dtype=np.int64))==len(values)
 record={'name':name,'dtype':'float32','shape':shape,'encoding':'little_endian_float32',**write_bytes('tensors/'+name+'.f32',data),'sources':[source(key,gi,ti)]}
 add(record)
 if key==DKEY and gi==0:exported_main_float.add(ti)
 return record
# LUT preserves signed two-bit numeric values while widening to signed nibbles.
lookup=np.empty((256,2),dtype=np.uint8)
for byte in range(256):
 codes=[((byte>>(2*i))&3) for i in range(4)];codes=[x if x<2 else x-4 for x in codes]
 lookup[byte]=[(codes[0]&15)|((codes[1]&15)<<4),(codes[2]&15)|((codes[3]&15)<<4)]
def export_quant(name,key,gi,ti):
 model=models[key];t=model.Subgraphs(gi).Tensors(ti);dtype=TT[t.Type()];shape=t.ShapeAsNumpy().tolist();data=raw(key,gi,ti);assert dtype in ['INT2','INT4','INT8']
 target='int4' if dtype in ['INT2','INT4'] else 'int8';rel='tensors/'+name+('.i4' if target=='int4' else '.i8');path=OUT/rel;assert not path.exists(),path
 h=hashlib.sha256()
 with path.open('wb') as f:
  for i in range(0,len(data),CHUNK):
   chunk=data[i:i+CHUNK]
   if dtype=='INT2':
    widened=lookup[chunk];lo=widened[:,0];hi=widened[:,1]
    restored=(lo&3)|(((lo>>4)&3)<<2)|((hi&3)<<4)|(((hi>>4)&3)<<6)
    assert np.array_equal(restored,chunk)
    assert np.all(((lo&15)<=1)|((lo&15)>=14))
    out=widened
   else:out=chunk
   f.write(memoryview(out).cast('B'));h.update(memoryview(out).cast('B'))
 stats['numeric_weight_codes_preserved']+=int(np.prod(shape,dtype=np.int64))
 stats['int2_widened_codes']+=int(np.prod(shape,dtype=np.int64)) if dtype=='INT2' else 0
 q=t.Quantization();assert q and q.QuantizedDimension()==0
 scales=q.ScaleAsNumpy();zeros=q.ZeroPointAsNumpy();assert len(scales)==shape[0] and np.isfinite(scales).all() and np.all(scales>0) and np.all(zeros==0)
 info=write_bytes('scales/'+name+'.f32',scales)
 record={'name':name,'dtype':target,'shape':shape,'encoding':'signed_twos_complement_low_nibble_first' if target=='int4' else 'signed_int8', 'file':rel,'bytes':path.stat().st_size,'sha256':h.hexdigest(),'quantization':{'kind':'per_channel','quantized_dimension':0,'scales_file':info['file'],'scales_shape':[len(scales)],'scales_dtype':'float32','scales_bytes':info['bytes'],'scales_sha256':info['sha256'],'zero_points':[0]},'sources':[source(key,gi,ti)]}
 if dtype!='INT2':assert record['sha256']==record['sources'][0]['source_data_sha256']
 add(record);return record
# Main 277 active FC weights and all 552 explicit static activation scale scalars.
fcs=[o for o in dec_trace['operators'] if o['type']=='FULLY_CONNECTED'];assert len(fcs)==277
for index,op in enumerate(fcs):
 wt=op['inputs'][1];name=name_for_fc(wt['name']);export_quant(name+'.weight',DKEY,0,wt['index'])
 for role,tinfo in [('input',op['inputs'][0]),('output',op['outputs'][0])]:
  if 'scales' not in tinfo:continue
  tensor=dg.Tensors(tinfo['index']);q=tensor.Quantization();scales=q.ScaleAsNumpy();assert len(scales)==1 and q.ZeroPointAsNumpy()[0]==0
  record={'name':name+'.'+role+'_scale','dtype':'float32','shape':[1],'encoding':'little_endian_float32',**write_bytes('tensors/'+name+'.'+role+'_scale.f32',scales),'sources':[{'section_index':secs[DKEY]['index'],'subgraph_index':0,'tensor_index':tinfo['index'],'tensor_name':tinfo['name'],'field':'quantization.scale','source_data_sha256':sha(scales)}]};add(record)
 if index%35==0:print('FC',index+1,'of277',round(time.time()-start,2),'s',flush=True)
# All learned RMSNorm weights mapped by actual consumers, not nearest-value guesses.
for op in dec_trace['operators']:
 if op.get('composite_name')!='odml.rms_norm':continue
 output=op['outputs'][0];path=output['name'];ti=op['inputs'][1]['index'];layer_match=re.search(r'/layer_(\d+)/',path)
 if '/value_norm/' in path:
  assert np.all(raw(DKEY,0,ti).view('<f4')==1);stats['unit_value_norm_uses']+=1;continue
 if layer_match:
  role=path.split('/')[-2];suffix={'pre_attention_norm':'input_layernorm','query_norm':'self_attn.q_norm','key_norm':'self_attn.k_norm','post_attention_norm':'post_attention_layernorm','pre_ffw_norm':'pre_feedforward_layernorm','post_ffw_norm':'post_feedforward_layernorm','post_per_layer_input_norm':'post_per_layer_input_norm'}.get(role)
  assert suffix,path;name=f'model.layers.{int(layer_match.group(1))}.{suffix}.weight'
 elif '/per_layer_embedding_projection_norm/' in path:name='model.per_layer_projection_norm.weight'
 elif op['index']==fcs[-1]['index']-1 and output['index']==fcs[-1]['inputs'][0]['index']:name='model.norm.weight'
 else:raise ValueError('Unmapped norm '+path)
 rec=export_float(name,DKEY,0,ti);rec['mapped_by_consumer']={'operator_index':op['index'],'output_name':path};stats['learned_norms']+=1
assert stats['learned_norms']==227
# Every layer residual scalar is a stored float constant with a named MUL consumer.
for op in dec_trace['operators']:
 path=op['outputs'][0]['name']
 if op['type']!='MUL' or '._maybe_apply_skip_scale/mul' not in path:continue
 layer=int(re.search(r'/layer_(\d+)/',path).group(1));static=[]
 for inp in op['inputs']:
  t=dg.Tensors(inp['index']);b=dec.Buffers(t.Buffer())
  if b.DataLength():static.append(inp['index'])
 assert len(static)==1,path
 rec=export_float(f'model.layers.{layer}.layer_scalar',DKEY,0,static[0],[1]);rec['mapped_by_consumer']={'operator_index':op['index'],'output_name':path};stats['layer_scalars']+=1
assert stats['layer_scalars']==35
# Main table is independently sourced from the actual embedder section.
key='tf_lite_embedder';g=models[key].Subgraphs(0)
tables=[i for i in range(g.TensorsLength()) if TT[g.Tensors(i).Type()] in ['INT2','INT4']];assert len(tables)==1
export_quant('model.embed_tokens.weight',key,0,tables[0]);print('Main embedding exported',flush=True)
# Assemble all 35 per-layer packed tables and full FP32 scale columns into native token-major layout.
key='tf_lite_per_layer_embedder';model=models[key];g=model.Subgraphs(0);tis=[i for i in range(g.TensorsLength()) if TT[g.Tensors(i).Type()]=='INT4'];assert len(tis)==35
parts=[];scale_columns=[];sources=[]
for layer,ti in enumerate(tis):
 t=g.Tensors(ti);assert t.ShapeAsNumpy().tolist()==[262144,256];q=t.Quantization();assert q.QuantizedDimension()==0 and q.ScaleLength()==262144 and np.all(q.ZeroPointAsNumpy()==0)
 parts.append(raw(key,0,ti).reshape(262144,128));scales=q.ScaleAsNumpy();assert np.all(np.isfinite(scales)) and np.all(scales>0);scale_columns.append(scales);sources.append({**source(key,0,ti),'destination_layer_partition':layer,'source_scale_sha256':sha(scales)})
name='model.embed_tokens_per_layer.weight';rel='tensors/'+name+'.i4';path=OUT/rel;assert not path.exists();h=hashlib.sha256()
with path.open('wb') as f:
 for row in range(0,262144,4096):
  end=min(row+4096,262144);block=np.empty((end-row,35,128),dtype=np.uint8)
  for layer,part in enumerate(parts):block[:,layer,:]=part[row:end]
  for layer,part in enumerate(parts):assert np.array_equal(block[:,layer,:],part[row:end])
  f.write(memoryview(block).cast('B'));h.update(memoryview(block).cast('B'))
scale_matrix=np.stack(scale_columns,axis=1).astype('<f4',copy=False);info=write_bytes('scales/'+name+'.f32',scale_matrix)
add({'name':name,'dtype':'int4','shape':[262144,8960],'encoding':'signed_twos_complement_low_nibble_first','file':rel,'bytes':path.stat().st_size,'sha256':h.hexdigest(),'quantization':{'kind':'blockwise','block_size':256,'quantized_dimension':0,'scales_file':info['file'],'scales_shape':[262144,35],'scales_dtype':'float32','scales_bytes':info['bytes'],'scales_sha256':info['sha256'],'zero_points':[0]},'sources':sources})
stats['numeric_weight_codes_preserved']+=262144*8960;stats['per_layer_table_partitions']=35
print('Per-layer embeddings exported',round(time.time()-start,2),'s',flush=True)
# Inventory every remaining main-graph FP32 constant. Unknown variable coefficients are fatal.
for ti in range(dg.TensorsLength()):
 t=dg.Tensors(ti);b=dec.Buffers(t.Buffer())
 if TT[t.Type()]!='FLOAT32' or not b.DataLength() or ti in exported_main_float:continue
 name=t.Name().decode();data=raw(DKEY,0,ti)
 cons=[{'operator_index':op['index'],'type':op['type'],'output_name':op['outputs'][0]['name']} for op in dec_trace['operators'] if any(i['index']==ti for i in op['inputs'])]
 record={'name':'decode.tensor_'+str(ti),'dtype':'float32','shape':t.ShapeAsNumpy().tolist(),**write_bytes('constants/decode.tensor_'+str(ti)+'.f32',data),'sources':[source(DKEY,0,ti)],'consumers':cons}
 if 'jax2tf_arg_' in name or 'ReadVariableOp' in name:unmapped.append(record)
 constants.append(record)
# Composite decomposition scalar arithmetic constants (epsilon, inverse reduction length etc.), deduplicated by source buffer.
seen=set()
for op in dec_trace['operators']:
 if 'decomposition_subgraph' not in op:continue
 gi=op['decomposition_subgraph'];g=dec.Subgraphs(gi)
 for ti in range(g.TensorsLength()):
  t=g.Tensors(ti);b=dec.Buffers(t.Buffer())
  if TT[t.Type()]!='FLOAT32' or not b.DataLength() or t.Buffer() in seen:continue
  seen.add(t.Buffer());constants.append({'name':f'decomposition_{gi}.tensor_{ti}','dtype':'float32','shape':t.ShapeAsNumpy().tolist(),**write_bytes(f'constants/decomposition_{gi}.tensor_{ti}.f32',raw(DKEY,gi,ti)),'sources':[source(DKEY,gi,ti)]})
# Main/per-layer embedder fixed scalar constants are included as metadata artifacts too.
for key in ['tf_lite_embedder','tf_lite_per_layer_embedder']:
 model=models[key];g=model.Subgraphs(0)
 for ti in range(g.TensorsLength()):
  t=g.Tensors(ti);b=model.Buffers(t.Buffer())
  if TT[t.Type()]!='FLOAT32' or not b.DataLength():continue
  constants.append({'name':f'{key}.tensor_{ti}','dtype':'float32','shape':t.ShapeAsNumpy().tolist(),**write_bytes(f'constants/{key}.tensor_{ti}.f32',raw(key,0,ti)),'sources':[source(key,0,ti)]})
kv=json.loads(KV.read_text())['graphs'][0]['cache_updates'];kv=[{'owner_layer':x['owner'],'key_scale':x['key_scale'],'value_scale':x['value_scale'],'zero_point':0,'source_cache_update_operator':x['operator'],'source_decomposition_subgraph':x['cache_update_decomposition']} for x in kv]
kv_specs=[{'owner':x['owner_layer'],'head_dim':512 if x['owner_layer']%5==4 else 256,'key_scale':x['key_scale'],'value_scale':x['value_scale'],'zero_point':0,'key_shape_template':[1,1,'capacity',512 if x['owner_layer']%5==4 else 256],'value_shape_template':[1,1,512 if x['owner_layer']%5==4 else 256,'capacity']} for x in kv]
manifest={'schema_version':1,'status':'complete' if not unmapped else 'incomplete_unmapped_float_coefficients','source_bundle':{'path':p['path'],'bytes':p['bytes'],'sha256':sha(mm)},'source_inventory_sha256':sha(INV.read_bytes()),'exporter_sha256':sha(Path(__file__).read_bytes()),'tensors':records,'constants':constants,'kv_cache':kv,'kv_cache_specs':kv_specs,'unmapped_float_coefficients':unmapped,'coverage':dict(stats),'tensor_counts':dict(Counter(r['dtype'] for r in records)),'export_seconds':time.time()-start,'constraints':['All weights and float coefficients come directly from published bundle buffers, not from CT.','INT2 codes widened losslessly to signed INT4; native dynamic head may repack to actual QC2.','Fixed scalar and RoPE constants separately exported for explicit graph alignment.','No inference, builds or device actions performed.']}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('MANIFEST',manifest['status'],'tensors',len(records),'constants',len(constants),'unmapped',len(unmapped),dict(stats),flush=True)
if unmapped:raise SystemExit(2)
