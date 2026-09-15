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

"""Read-only bundle metadata and graph inventory; no inference or repository edits."""
from pathlib import Path
from collections import Counter
import hashlib,json,mmap,struct,sys,zlib
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--schema-dir', type=Path, required=True,
                    help='Generated TFLite Python package parent (contains tflite/Model.py).')
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
if not (args.schema_dir / 'tflite/Model.py').is_file():
    parser.error('--schema-dir must contain generated tflite/Model.py')
sys.path.insert(0, str(args.schema_dir.resolve()))
from schema import litertlm_header_schema_py_generated as schema
from tflite.Model import Model
from tflite.TensorType import TensorType
from tflite.BuiltinOperator import BuiltinOperator
OUT=args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=False)
CANDIDATES=[args.model.resolve()]
DTYPES={v:k for k,v in vars(schema.AnySectionDataType).items() if isinstance(v,int)}
VTYPES={v:k for k,v in vars(schema.VData).items() if isinstance(v,int)}
TTYPES={v:k for k,v in vars(TensorType).items() if isinstance(v,int)}
BTYPES={v:k for k,v in vars(BuiltinOperator).items() if isinstance(v,int)}

def kv(obj):
 table=obj.Value();value=getattr(schema,VTYPES[obj.ValueType()])();value.Init(table.Bytes,table.Pos)
 v=value.Value()
 return obj.Key().decode(),v.decode() if isinstance(v,bytes) else v

def tensor_info(model,sg,index):
 t=sg.Tensors(index);q=t.Quantization()
 result={'index':int(index),'name':t.Name().decode() if t.Name() else None,'dtype':TTYPES[t.Type()],'shape':t.ShapeAsNumpy().tolist(),'buffer':int(t.Buffer())}
 if q is not None and q.ScaleLength():
  scales=q.ScaleAsNumpy();zeros=q.ZeroPointAsNumpy()
  result['quantization']={'scale_count':q.ScaleLength(),'scale_first':scales[:8].tolist(),'scale_sha256':hashlib.sha256(scales.tobytes()).hexdigest(),'zero_point_count':q.ZeroPointLength(),'zero_point_first':zeros[:8].tolist(),'quantized_dimension':q.QuantizedDimension()}
 return result

reports=[]
for candidate in CANDIDATES:
 name='published'
 with candidate.open('rb') as f:
  prefix=f.read(32)
  assert len(prefix)==32
  header_end=int.from_bytes(prefix[24:32],'little')
  assert 32 <= header_end <= candidate.stat().st_size
  prefix += f.read(header_end-32)
 assert prefix[:8]==b'LITERTLM'
 end=int.from_bytes(prefix[24:32],'little');header=schema.LiteRTLMMetaData.GetRootAs(prefix[32:end],0)
 sm=header.SystemMetadata();report={'path':str(candidate),'resolved_path':str(candidate.resolve()),'bytes':candidate.stat().st_size,'version':list(struct.unpack('<III',prefix[8:20])),'system_metadata':dict(kv(sm.Entries(i)) for i in range(sm.EntriesLength())),'sections':[]}
 f=candidate.open('rb');mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
 sections=header.SectionMetadata()
 for i in range(sections.ObjectsLength()):
  obj=sections.Objects(i);begin=obj.BeginOffset();end=obj.EndOffset();kind=DTYPES[obj.DataType()]
  s={'index':i,'type':kind,'begin':begin,'end':end,'bytes':end-begin,'items':dict(kv(obj.Items(j)) for j in range(obj.ItemsLength()))}
  if kind=='HF_Tokenizer_Zlib':
   raw=zlib.decompress(mm[begin:end]);p=OUT/(name+'-tokenizer.json');p.write_bytes(raw)
   parsed=json.loads(raw);s.update(tokenizer_file=str(p),tokenizer_sha256=hashlib.sha256(raw).hexdigest(),tokenizer_keys=list(parsed))
   tm=parsed.get('model',{});s['tokenizer_model_type']=tm.get('type');s['tokenizer_vocab_size']=len(tm.get('vocab',{}));s['added_tokens_count']=len(parsed.get('added_tokens',[]))
  if kind=='SP_Tokenizer':s['tokenizer_sha256']=hashlib.sha256(mm[begin:end]).hexdigest()
  if kind=='TFLiteModel':
   view=memoryview(mm)[begin:end];model=Model.GetRootAsModel(view,0)
   s['model_description']=model.Description().decode() if model.Description() else None
   s['signature_keys']=[model.SignatureDefs(j).SignatureKey().decode() for j in range(model.SignatureDefsLength())]
   s['subgraphs']=[]
   opcode={j:(model.OperatorCodes(j).CustomCode().decode() if model.OperatorCodes(j).BuiltinCode()==32 else BTYPES.get(model.OperatorCodes(j).BuiltinCode(),str(model.OperatorCodes(j).BuiltinCode()))) for j in range(model.OperatorCodesLength())}
   for j in range(model.SubgraphsLength()):
    sg=model.Subgraphs(j);sginfo={'index':j,'name':sg.Name().decode() if sg.Name() else None,'tensors':sg.TensorsLength(),'operators':sg.OperatorsLength(),'tensor_dtypes':dict(Counter(TTYPES[sg.Tensors(k).Type()] for k in range(sg.TensorsLength()))),'operator_types':dict(Counter(opcode[sg.Operators(k).OpcodeIndex()] for k in range(sg.OperatorsLength()))),'inputs':[tensor_info(model,sg,k) for k in (sg.InputsAsNumpy() if sg.InputsLength() else [])],'outputs':[tensor_info(model,sg,k) for k in (sg.OutputsAsNumpy() if sg.OutputsLength() else [])]}
    fcs=[]
    for k in range(sg.OperatorsLength()):
     op=sg.Operators(k)
     if opcode[op.OpcodeIndex()]=='FULLY_CONNECTED':fcs.append({'operator_index':k,'inputs':[tensor_info(model,sg,int(t)) for t in op.InputsAsNumpy() if t>=0],'outputs':[tensor_info(model,sg,int(t)) for t in op.OutputsAsNumpy() if t>=0]})
    sginfo['fully_connected']=fcs
    s['subgraphs'].append(sginfo)
  report['sections'].append(s)
 reports.append(report)
 (OUT/(name+'-inventory.json')).write_text(json.dumps(report,indent=2)+'\n')
 print(name,'system',report['system_metadata'],flush=True)
 for s in report['sections']:
  print(s['index'],s['type'],s['bytes'],s['items'],s.get('signature_keys'),flush=True)
  if 'metadata' in s:print('METADATA',json.dumps({k:v for k,v in s['metadata'].items() if k != 'jinja_prompt_template'}),flush=True)
  if 'tokenizer_sha256' in s:print('TOKENIZER',s['tokenizer_sha256'],s.get('tokenizer_vocab_size'),flush=True)
(OUT/'all-inventory.json').write_text(json.dumps(reports,indent=2)+'\n')
