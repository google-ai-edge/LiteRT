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

"""Read-only bundle metadata and graph inventory; no inference or edits."""

import collections
from collections.abc import Sequence
import hashlib
import json
import mmap
import pathlib
import struct
from typing import Any
import zlib

from absl import app
from absl import flags
import litertlm_header_schema_py_generated as schema
from tflite import BuiltinOperator as tflite_builtin_operator
from tflite import Model as tflite_model
from tflite import TensorType as tflite_tensor_type

_SCHEMA_DIR = flags.DEFINE_string(
    "schema_dir",
    None,
    "Generated TFLite Python package parent (contains tflite/Model.py).",
    required=True,
)
_MODEL = flags.DEFINE_string(
    "model",
    None,
    "Path to the LiteRT-LM model bundle file.",
    required=True,
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Directory where inventory JSON files will be written.",
    required=True,
)

_SECTION_DTYPES = {
    v: k
    for k, v in vars(schema.AnySectionDataType).items()
    if isinstance(v, int)
}
_VALUE_TYPES = {
    v: k for k, v in vars(schema.VData).items() if isinstance(v, int)
}
_TENSOR_TYPES = {
    v: k
    for k, v in vars(tflite_tensor_type.TensorType).items()
    if isinstance(v, int)
}
_BUILTIN_OPS = {
    v: k
    for k, v in vars(tflite_builtin_operator.BuiltinOperator).items()
    if isinstance(v, int)
}


def _extract_kv(entry_obj: Any) -> tuple[str, Any]:
  """Decodes a key-value metadata entry from the FlatBuffers header."""
  table = entry_obj.Value()
  value = getattr(schema, _VALUE_TYPES[entry_obj.ValueType()])()
  value.Init(table.Bytes, table.Pos)
  raw_val = value.Value()
  decoded_val = raw_val.decode() if isinstance(raw_val, bytes) else raw_val
  return entry_obj.Key().decode(), decoded_val


def _extract_tensor_info(subgraph: Any, index: int) -> dict[str, Any]:
  """Builds a summary dictionary for a tensor in a TFLite subgraph."""
  tensor_obj = subgraph.Tensors(index)
  quant = tensor_obj.Quantization()
  result: dict[str, Any] = {
      "index": int(index),
      "name": tensor_obj.Name().decode() if tensor_obj.Name() else None,
      "dtype": _TENSOR_TYPES[tensor_obj.Type()],
      "shape": tensor_obj.ShapeAsNumpy().tolist(),
      "buffer": int(tensor_obj.Buffer()),
  }
  if quant is not None and quant.ScaleLength():
    scales = quant.ScaleAsNumpy()
    zeros = quant.ZeroPointAsNumpy()
    result["quantization"] = {
        "scale_count": quant.ScaleLength(),
        "scale_first": scales[:8].tolist(),
        "scale_sha256": hashlib.sha256(scales.tobytes()).hexdigest(),
        "zero_point_count": quant.ZeroPointLength(),
        "zero_point_first": zeros[:8].tolist(),
        "quantized_dimension": quant.QuantizedDimension(),
    }
  return result


def main(argv: Sequence[str]) -> None:
  """Inspects the LiteRT-LM bundle sections and writes inventory artifacts."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  schema_dir = pathlib.Path(_SCHEMA_DIR.value)
  if not (schema_dir / "tflite/Model.py").is_file():
    raise app.UsageError("--schema_dir must contain generated tflite/Model.py")
  out_dir = pathlib.Path(_OUTPUT_DIR.value).resolve()
  out_dir.mkdir(parents=True, exist_ok=False)
  candidates = [pathlib.Path(_MODEL.value).resolve()]

  reports = []
  for candidate in candidates:
    name = "published"
    with candidate.open("rb") as bundle_file:
      prefix = bundle_file.read(32)
      assert len(prefix) == 32
      header_end = int.from_bytes(prefix[24:32], "little")
      assert 32 <= header_end <= candidate.stat().st_size
      prefix += bundle_file.read(header_end - 32)
    assert prefix[:8] == b"LITERTLM"
    end_offset = int.from_bytes(prefix[24:32], "little")
    header = schema.LiteRTLMMetaData.GetRootAs(prefix[32:end_offset], 0)
    sys_meta = header.SystemMetadata()
    report: dict[str, Any] = {
        "path": str(candidate),
        "resolved_path": str(candidate.resolve()),
        "bytes": candidate.stat().st_size,
        "version": list(struct.unpack("<III", prefix[8:20])),
        "system_metadata": dict(
            _extract_kv(sys_meta.Entries(i))
            for i in range(sys_meta.EntriesLength())
        ),
        "sections": [],
    }
    with (
        candidate.open("rb") as bundle_file,
        mmap.mmap(bundle_file.fileno(), 0, access=mmap.ACCESS_READ) as mm,
    ):
      sections = header.SectionMetadata()
      for i in range(sections.ObjectsLength()):
        sec_obj = sections.Objects(i)
        begin = sec_obj.BeginOffset()
        end = sec_obj.EndOffset()
        kind = _SECTION_DTYPES[sec_obj.DataType()]
        sec_info: dict[str, Any] = {
            "index": i,
            "type": kind,
            "begin": begin,
            "end": end,
            "bytes": end - begin,
            "items": dict(
                _extract_kv(sec_obj.Items(j))
                for j in range(sec_obj.ItemsLength())
            ),
        }
        if kind == "HF_Tokenizer_Zlib":
          raw_bytes = zlib.decompress(mm[begin:end])
          tok_path = out_dir / f"{name}-tokenizer.json"
          tok_path.write_bytes(raw_bytes)
          parsed = json.loads(raw_bytes)
          sec_info.update(
              tokenizer_file=str(tok_path),
              tokenizer_sha256=hashlib.sha256(raw_bytes).hexdigest(),
              tokenizer_keys=list(parsed),
          )
          tok_model = parsed.get("model", {})
          sec_info["tokenizer_model_type"] = tok_model.get("type")
          sec_info["tokenizer_vocab_size"] = len(tok_model.get("vocab", {}))
          sec_info["added_tokens_count"] = len(parsed.get("added_tokens", []))
        if kind == "SP_Tokenizer":
          sec_info["tokenizer_sha256"] = hashlib.sha256(
              mm[begin:end]
          ).hexdigest()
        if kind == "TFLiteModel":
          view = memoryview(mm)[begin:end]
          model = tflite_model.Model.GetRootAsModel(view, 0)
          sec_info["model_description"] = (
              model.Description().decode() if model.Description() else None
          )
          sec_info["signature_keys"] = [
              model.SignatureDefs(j).SignatureKey().decode()
              for j in range(model.SignatureDefsLength())
          ]
          sec_info["subgraphs"] = []
          opcode_map = {}
          for j in range(model.OperatorCodesLength()):
            op_code = model.OperatorCodes(j)
            if op_code.BuiltinCode() == 32:
              opcode_map[j] = op_code.CustomCode().decode()
            else:
              opcode_map[j] = _BUILTIN_OPS.get(
                  op_code.BuiltinCode(), str(op_code.BuiltinCode())
              )
          for j in range(model.SubgraphsLength()):
            subgraph = model.Subgraphs(j)
            sg_inputs = (
                subgraph.InputsAsNumpy() if subgraph.InputsLength() else []
            )
            sg_outputs = (
                subgraph.OutputsAsNumpy() if subgraph.OutputsLength() else []
            )
            sg_info: dict[str, Any] = {
                "index": j,
                "name": subgraph.Name().decode() if subgraph.Name() else None,
                "tensors": subgraph.TensorsLength(),
                "operators": subgraph.OperatorsLength(),
                "tensor_dtypes": dict(
                    collections.Counter(
                        _TENSOR_TYPES[subgraph.Tensors(k).Type()]
                        for k in range(subgraph.TensorsLength())
                    )
                ),
                "operator_types": dict(
                    collections.Counter(
                        opcode_map[subgraph.Operators(k).OpcodeIndex()]
                        for k in range(subgraph.OperatorsLength())
                    )
                ),
                "inputs": [
                    _extract_tensor_info(subgraph, k) for k in sg_inputs
                ],
                "outputs": [
                    _extract_tensor_info(subgraph, k) for k in sg_outputs
                ],
            }
            fc_ops = []
            for k in range(subgraph.OperatorsLength()):
              op = subgraph.Operators(k)
              if opcode_map[op.OpcodeIndex()] == "FULLY_CONNECTED":
                fc_ops.append({
                    "operator_index": k,
                    "inputs": [
                        _extract_tensor_info(subgraph, int(t))
                        for t in op.InputsAsNumpy()
                        if t >= 0
                    ],
                    "outputs": [
                        _extract_tensor_info(subgraph, int(t))
                        for t in op.OutputsAsNumpy()
                        if t >= 0
                    ],
                })
            sg_info["fully_connected"] = fc_ops
            sec_info["subgraphs"].append(sg_info)
        report["sections"].append(sec_info)
    reports.append(report)
    (out_dir / f"{name}-inventory.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(name, "system", report["system_metadata"], flush=True)
    for sec_entry in report["sections"]:
      print(
          sec_entry["index"],
          sec_entry["type"],
          sec_entry["bytes"],
          sec_entry["items"],
          sec_entry.get("signature_keys"),
          flush=True,
      )
      if "metadata" in sec_entry:
        print(
            "METADATA",
            json.dumps({
                k: v
                for k, v in sec_entry["metadata"].items()
                if k != "jinja_prompt_template"
            }),
            flush=True,
        )
      if "tokenizer_sha256" in sec_entry:
        print(
            "TOKENIZER",
            sec_entry["tokenizer_sha256"],
            sec_entry.get("tokenizer_vocab_size"),
            flush=True,
        )
  (out_dir / "all-inventory.json").write_text(
      json.dumps(reports, indent=2) + "\n"
  )


if __name__ == "__main__":
  app.run(main)
