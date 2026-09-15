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
import json,mmap,sys
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--schema-dir', type=Path, required=True,
                    help='Generated TFLite Python package parent (contains tflite/Model.py).')
parser.add_argument('--inventory', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if not (args.schema_dir / 'tflite/Model.py').is_file():
    parser.error('--schema-dir must contain generated tflite/Model.py')
sys.path.insert(0, str(args.schema_dir.resolve()))

from tflite.Model import Model
from tflite.TensorType import TensorType
from tflite.BuiltinOperator import BuiltinOperator
from tflite.StableHLOCompositeOptions import StableHLOCompositeOptions
TT={v:k for k,v in vars(TensorType).items() if isinstance(v,int)};BT={v:k for k,v in vars(BuiltinOperator).items() if isinstance(v,int)}
p=json.loads(args.inventory.read_text())
if args.output.exists(): raise FileExistsError(args.output)
s=next(x for x in p['sections'] if x['items'].get('model_type')=='tf_lite_prefill_decode')
f=open(p['path'],'rb');mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ);m=Model.GetRootAsModel(memoryview(mm)[s['begin']:s['end']],0)
def tensor(g,i):
 t=g.Tensors(i);q=t.Quantization();r={'index':int(i),'name':t.Name().decode(),'dtype':TT[t.Type()],'shape':t.ShapeAsNumpy().tolist()}
 if q and q.ScaleLength():r['scales']=q.ScaleAsNumpy()[:4].tolist();r['zero_points']=q.ZeroPointAsNumpy()[:4].tolist()
 return r

def opinfo(g,i):
 op=g.Operators(i);r={'index':i,'type':BT[m.OperatorCodes(op.OpcodeIndex()).BuiltinCode()],'inputs':[tensor(g,int(j)) for j in op.InputsAsNumpy() if j>=0],'outputs':[tensor(g,int(j)) for j in op.OutputsAsNumpy() if j>=0]}
 if r['type']=='STABLEHLO_COMPOSITE':
  table=op.BuiltinOptions2();opts=StableHLOCompositeOptions();opts.Init(table.Bytes,table.Pos);r['composite_name']=opts.Name().decode();r['decomposition_subgraph']=opts.DecompositionSubgraphIndex()
 return r
report={'bundle':p['path'],'graphs':[]}
for si in range(m.SubgraphsLength()):
 g=m.Subgraphs(si);name=g.Name().decode()
 if name not in ['decode','prefill_128','prefill_1024','verify']:continue
 gr={'index':si,'name':name,'inputs':[tensor(g,int(i)) for i in g.InputsAsNumpy()],'outputs':[tensor(g,int(i)) for i in g.OutputsAsNumpy()],'operators':[opinfo(g,i) for i in range(g.OperatorsLength())]};report['graphs'].append(gr)
 print('GRAPH',si,name)
 for x in gr['operators'][:80]:
  ins=','.join(str(t['index'])+':'+t['dtype'] for t in x['inputs']);outs=','.join(str(t['index'])+':'+t['dtype']+' '+t['name'].split('/')[-1] for t in x['outputs'])
  print(x['index'],x['type'],x.get('composite_name',''),x.get('decomposition_subgraph',''),ins,'->',outs)
 # Include the body of each referenced composite, once, in structured evidence.
 decomps={x['decomposition_subgraph'] for x in gr['operators'] if 'decomposition_subgraph' in x}
 gr['decompositions']={str(d):{'name':m.Subgraphs(d).Name().decode(),'inputs':[tensor(m.Subgraphs(d),int(i)) for i in m.Subgraphs(d).InputsAsNumpy()],'outputs':[tensor(m.Subgraphs(d),int(i)) for i in m.Subgraphs(d).OutputsAsNumpy()],'operators':[opinfo(m.Subgraphs(d),i) for i in range(m.Subgraphs(d).OperatorsLength())]} for d in decomps}
args.output.write_text(json.dumps(report,indent=2)+'\n')
