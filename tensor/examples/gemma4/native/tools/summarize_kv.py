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

from pathlib import Path
import json,re
import argparse
parser=argparse.ArgumentParser(description='Inspect published owner sharing and KV quantization from a graph trace.')
parser.add_argument('--trace', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
if args.output.exists(): raise FileExistsError(args.output)
p=json.loads(args.trace.read_text());summary={'graphs':[]}
for g in p['graphs']:
 # Include full verify graph to check sharing on a multi-token path too.
 cachemap={};updates=[];reads=[]
 for op in g['operators']:
  if op.get('composite_name')!='odml.cache_update':continue
  kin=next(t for t in op['inputs'] if '_kv_cache_k_' in t['name']);vin=next(t for t in op['inputs'] if '_kv_cache_v_' in t['name']);owner=int(re.search(r'_kv_cache_k_(\d+)',kin['name']).group(1))
  for kind,t in zip(['K','V'],op['outputs']):cachemap[t['index']]={'owner':owner,'kind':kind,'scales':t['scales'],'dtype':t['dtype']}
  decomp=g['decompositions'][str(op['decomposition_subgraph'])]
  qs=[o for o in decomp['operators'] if o['type']=='QUANTIZE'];assert len(qs)==2
  updates.append({'owner':owner,'operator':op['index'],'key_scale':kin['scales'][0],'value_scale':vin['scales'][0],'cache_update_decomposition':op['decomposition_subgraph'],'first_two_inputs':[t for t in op['inputs'][:2]],'quantize_operators':qs})
 for op in g['operators']:
  if op.get('composite_name')!='odml.runtime_bmm':continue
  info=cachemap[op['inputs'][1]['index']];layer=int(re.search(r'/layer_(\d+)/',op['outputs'][0]['name']).group(1));decomp=g['decompositions'][str(op['decomposition_subgraph'])]
  assert any(o['type']=='DEQUANTIZE' and o['inputs'][0]['dtype']=='INT8' for o in decomp['operators'])
  assert any(o['type']=='BATCH_MATMUL' and all(t['dtype']=='FLOAT32' for t in o['inputs']) for o in decomp['operators'])
  reads.append({'layer':layer,'operator':op['index'],**info,'dequantized_inside_bmm':True})
 assert len(updates)==15 and len(reads)==(28 if g['name'].startswith('prefill') else 70)
 expected={i:(i if i<15 else 14 if i%5==4 else 13) for i in range(35)}
 assert all(expected[r['layer']]==r['owner'] for r in reads)
 summary['graphs'].append({'name':g['name'],'index':g['index'],'cache_updates':updates,'attention_reads':reads,'output_dtypes':[t['dtype'] for t in g['outputs']],'output_shapes':[t['shape'] for t in g['outputs']],'fully_connected_count':sum(o['type']=='FULLY_CONNECTED' for o in g['operators'])})
args.output.write_text(json.dumps(summary,indent=2)+'\n')
for g in summary['graphs']:
 print(g['name'],len(g['cache_updates']),'owner cache updates,',len(g['attention_reads']),'INT8 cache reads through dequantized FP32 BMM; sharing checked')
print('scales',[(x['owner'],x['key_scale'],x['value_scale']) for x in summary['graphs'][0]['cache_updates']])
