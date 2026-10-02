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

"""Export actual published text-model tensors from a LiteRT-LM bundle."""

import collections
from collections.abc import Sequence
import hashlib
import json
import mmap
import pathlib
import re
import time
from typing import Any

from absl import app
from absl import flags
import numpy as np
from tflite import Model as tflite_model
from tflite import TensorType as tflite_tensor_type

_SCHEMA_DIR = flags.DEFINE_string(
    "schema_dir",
    None,
    "Generated TFLite Python package parent (contains tflite/Model.py).",
    required=True,
)
_INVENTORY = flags.DEFINE_string(
    "inventory",
    None,
    "Path to the published bundle inventory JSON file.",
    required=True,
)
_TRACE = flags.DEFINE_string(
    "trace",
    None,
    "Path to the KV graph trace JSON file.",
    required=True,
)
_KV = flags.DEFINE_string(
    "kv",
    None,
    "Path to the KV placement summary JSON file.",
    required=True,
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Output directory where exported tensors and manifest will be written.",
    required=True,
)

_TENSOR_TYPES = {
    v: k
    for k, v in vars(tflite_tensor_type.TensorType).items()
    if isinstance(v, int)
}
_CHUNK_BYTES = 8 * 1024 * 1024
_DECODE_KEY = "tf_lite_prefill_decode"


def _sha256_bytes(data: Any) -> str:
  """Returns the hex SHA-256 digest of a byte-like object or array."""
  return hashlib.sha256(data).hexdigest()


def _sha256_file(file_path: pathlib.Path) -> str:
  """Returns the hex SHA-256 digest of a file on disk."""
  hasher = hashlib.sha256()
  with file_path.open("rb") as input_file:
    while block := input_file.read(_CHUNK_BYTES):
      hasher.update(block)
  return hasher.hexdigest()


def _name_for_fc(weight_name: str) -> str:
  """Maps a TFLite fully-connected weight name to its canonical layer name."""
  if "per_layer_model_projection/" in weight_name:
    return "model.per_layer_model_projection"
  if "decode_softmax/" in weight_name:
    return "lm_head"
  layer_match = re.search(r"/layer_(\d+)/", weight_name)
  assert layer_match is not None, weight_name
  layer = int(layer_match.group(1))
  patterns = [
      ("/q_einsum/", "self_attn.q_proj"),
      ("/k_einsum/", "self_attn.k_proj"),
      ("/v_einsum/", "self_attn.v_proj"),
      ("/attn_vec_einsum/", "self_attn.o_proj"),
      ("/gating_einsum1/", "mlp.gate_proj"),
      ("/gating_einsum2/", "mlp.up_proj"),
      ("/mlp/linear/", "mlp.down_proj"),
      ("/per_layer_embedding_gate/", "per_layer_input_gate"),
      ("/per_layer_embedding_projection/", "per_layer_projection"),
  ]
  suffix = next((s for pat, s in patterns if pat in weight_name), None)
  assert suffix is not None, weight_name
  return f"model.layers.{layer}.{suffix}"


def _build_int2_lookup() -> np.ndarray:
  """Builds a 256x2 lookup table widening signed INT2 codes to signed INT4."""
  lookup = np.empty((256, 2), dtype=np.uint8)
  for byte_val in range(256):
    codes = [((byte_val >> (2 * i)) & 3) for i in range(4)]
    codes = [x if x < 2 else x - 4 for x in codes]
    lookup[byte_val] = [
        (codes[0] & 15) | ((codes[1] & 15) << 4),
        (codes[2] & 15) | ((codes[3] & 15) << 4),
    ]
  return lookup


def main(argv: Sequence[str]) -> None:
  """Exports model weights, scales, and constants from the published bundle."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  schema_dir = pathlib.Path(_SCHEMA_DIR.value)
  if not (schema_dir / "tflite/Model.py").is_file():
    raise app.UsageError("--schema_dir must contain generated tflite/Model.py")

  out_dir = pathlib.Path(_OUTPUT_DIR.value).resolve()
  out_dir.mkdir(parents=True, exist_ok=False)
  inv_path = pathlib.Path(_INVENTORY.value).resolve()
  trace_path = pathlib.Path(_TRACE.value).resolve()
  kv_path = pathlib.Path(_KV.value).resolve()

  inventory = json.loads(inv_path.read_text())
  trace = json.loads(trace_path.read_text())
  dec_trace = next(g for g in trace["graphs"] if g["name"] == "decode")

  for sub in ["tensors", "scales", "constants"]:
    (out_dir / sub).mkdir(exist_ok=True)

  stats: collections.Counter[str] = collections.Counter()
  records: list[dict[str, Any]] = []
  constants: list[dict[str, Any]] = []
  exported_main_float: set[int] = set()
  seen_names: set[str] = set()
  unmapped: list[dict[str, Any]] = []
  lookup = _build_int2_lookup()
  start_time = time.time()

  with (
      open(inventory["path"], "rb") as bundle_file,
      mmap.mmap(bundle_file.fileno(), 0, access=mmap.ACCESS_READ) as mm,
  ):
    sections = {
        s["items"].get("model_type"): s
        for s in inventory["sections"]
        if s["type"] == "TFLiteModel"
    }
    models = {
        k: tflite_model.Model.GetRootAsModel(
            memoryview(mm)[s["begin"] : s["end"]], 0
        )
        for k, s in sections.items()
    }
    dec_model = models[_DECODE_KEY]
    dec_subgraph = dec_model.Subgraphs(0)

    def raw_tensor_bytes(
        section_key: str, sg_idx: int, tensor_idx: int
    ) -> np.ndarray:
      """Returns the raw buffer data for a tensor in the specified section."""
      target_model = models[section_key]
      tensor_obj = target_model.Subgraphs(sg_idx).Tensors(tensor_idx)
      buf_obj = target_model.Buffers(tensor_obj.Buffer())
      assert buf_obj.DataLength() > 0, (
          section_key,
          sg_idx,
          tensor_idx,
          buf_obj.Offset(),
          buf_obj.Size(),
      )
      return buf_obj.DataAsNumpy()

    def build_source_record(
        section_key: str, sg_idx: int, tensor_idx: int
    ) -> dict[str, Any]:
      """Constructs provenance metadata for a source tensor."""
      target_model = models[section_key]
      tensor_obj = target_model.Subgraphs(sg_idx).Tensors(tensor_idx)
      raw_data = raw_tensor_bytes(section_key, sg_idx, tensor_idx)
      return {
          "section_index": sections[section_key]["index"],
          "section_model_type": section_key,
          "subgraph_index": sg_idx,
          "tensor_index": tensor_idx,
          "tensor_name": tensor_obj.Name().decode(),
          "buffer_index": int(tensor_obj.Buffer()),
          "dtype": _TENSOR_TYPES[tensor_obj.Type()],
          "shape": tensor_obj.ShapeAsNumpy().tolist(),
          "source_bytes": len(raw_data),
          "source_data_sha256": _sha256_bytes(raw_data),
      }

    def write_bytes(rel_path: str, array_data: Any) -> dict[str, Any]:
      """Writes array bytes to a relative path under out_dir."""
      dst_path = out_dir / rel_path
      assert not dst_path.exists(), f"Refusing to overwrite export {dst_path}"
      with dst_path.open("wb") as dst_file:
        view = memoryview(array_data).cast("B")
        for offset in range(0, len(view), _CHUNK_BYTES):
          dst_file.write(view[offset : offset + _CHUNK_BYTES])
      return {
          "file": rel_path,
          "bytes": dst_path.stat().st_size,
          "sha256": _sha256_file(dst_path),
      }

    def add_record(record: dict[str, Any]) -> None:
      """Registers an exported tensor record in the manifest."""
      assert record["name"] not in seen_names, record["name"]
      seen_names.add(record["name"])
      records.append(record)

    def export_float(
        tensor_name: str,
        section_key: str,
        sg_idx: int,
        tensor_idx: int,
        override_shape: list[int] | None = None,
    ) -> dict[str, Any]:
      """Exports a float32 tensor and records its provenance."""
      tensor_obj = models[section_key].Subgraphs(sg_idx).Tensors(tensor_idx)
      assert _TENSOR_TYPES[tensor_obj.Type()] == "FLOAT32"
      raw_data = raw_tensor_bytes(section_key, sg_idx, tensor_idx)
      values = raw_data.view("<f4")
      assert np.isfinite(values).all(), tensor_name
      final_shape = (
          tensor_obj.ShapeAsNumpy().tolist()
          if override_shape is None
          else override_shape
      )
      assert int(np.prod(final_shape, dtype=np.int64)) == len(values)
      record = {
          "name": tensor_name,
          "dtype": "float32",
          "shape": final_shape,
          "encoding": "little_endian_float32",
          **write_bytes(f"tensors/{tensor_name}.f32", raw_data),
          "sources": [build_source_record(section_key, sg_idx, tensor_idx)],
      }
      add_record(record)
      if section_key == _DECODE_KEY and sg_idx == 0:
        exported_main_float.add(tensor_idx)
      return record

    def export_quant(
        tensor_name: str, section_key: str, sg_idx: int, tensor_idx: int
    ) -> dict[str, Any]:
      """Exports a quantized weight tensor and its per-channel scales."""
      target_model = models[section_key]
      tensor_obj = target_model.Subgraphs(sg_idx).Tensors(tensor_idx)
      dtype = _TENSOR_TYPES[tensor_obj.Type()]
      shape = tensor_obj.ShapeAsNumpy().tolist()
      raw_data = raw_tensor_bytes(section_key, sg_idx, tensor_idx)
      assert dtype in ["INT2", "INT4", "INT8"]
      target = "int4" if dtype in ["INT2", "INT4"] else "int8"
      suffix = ".i4" if target == "int4" else ".i8"
      rel_path = f"tensors/{tensor_name}{suffix}"
      dst_path = out_dir / rel_path
      assert not dst_path.exists(), dst_path
      hasher = hashlib.sha256()
      with dst_path.open("wb") as dst_file:
        for offset in range(0, len(raw_data), _CHUNK_BYTES):
          chunk = raw_data[offset : offset + _CHUNK_BYTES]
          if dtype == "INT2":
            widened = lookup[chunk]
            lo = widened[:, 0]
            hi = widened[:, 1]
            restored = (
                (lo & 3)
                | (((lo >> 4) & 3) << 2)
                | ((hi & 3) << 4)
                | (((hi >> 4) & 3) << 6)
            )
            assert np.array_equal(restored, chunk)
            assert np.all(((lo & 15) <= 1) | ((lo & 15) >= 14))
            out_chunk = widened
          else:
            out_chunk = chunk
          dst_file.write(memoryview(out_chunk).cast("B"))
          hasher.update(memoryview(out_chunk).cast("B"))
      num_codes = int(np.prod(shape, dtype=np.int64))
      stats["numeric_weight_codes_preserved"] += num_codes
      if dtype == "INT2":
        stats["int2_widened_codes"] += num_codes
      quant = tensor_obj.Quantization()
      assert quant and quant.QuantizedDimension() == 0
      scales = quant.ScaleAsNumpy()
      zeros = quant.ZeroPointAsNumpy()
      assert (
          len(scales) == shape[0]
          and np.isfinite(scales).all()
          and np.all(scales > 0)
          and np.all(zeros == 0)
      )
      scale_info = write_bytes(f"scales/{tensor_name}.f32", scales)
      encoding = (
          "signed_twos_complement_low_nibble_first"
          if target == "int4"
          else "signed_int8"
      )
      record = {
          "name": tensor_name,
          "dtype": target,
          "shape": shape,
          "encoding": encoding,
          "file": rel_path,
          "bytes": dst_path.stat().st_size,
          "sha256": hasher.hexdigest(),
          "quantization": {
              "kind": "per_channel",
              "quantized_dimension": 0,
              "scales_file": scale_info["file"],
              "scales_shape": [len(scales)],
              "scales_dtype": "float32",
              "scales_bytes": scale_info["bytes"],
              "scales_sha256": scale_info["sha256"],
              "zero_points": [0],
          },
          "sources": [build_source_record(section_key, sg_idx, tensor_idx)],
      }
      if dtype != "INT2":
        assert record["sha256"] == record["sources"][0]["source_data_sha256"]
      add_record(record)
      return record

    # Main 277 active FC weights and all 552 explicit static activation scales.
    fc_ops = [
        o for o in dec_trace["operators"] if o["type"] == "FULLY_CONNECTED"
    ]
    assert len(fc_ops) == 277
    for index, op in enumerate(fc_ops):
      weight_tensor = op["inputs"][1]
      fc_name = _name_for_fc(weight_tensor["name"])
      export_quant(f"{fc_name}.weight", _DECODE_KEY, 0, weight_tensor["index"])
      for role, t_info in [
          ("input", op["inputs"][0]),
          ("output", op["outputs"][0]),
      ]:
        if "scales" not in t_info:
          continue
        tensor_obj = dec_subgraph.Tensors(t_info["index"])
        quant = tensor_obj.Quantization()
        scales = quant.ScaleAsNumpy()
        assert len(scales) == 1 and quant.ZeroPointAsNumpy()[0] == 0
        scale_rel = f"tensors/{fc_name}.{role}_scale.f32"
        add_record({
            "name": f"{fc_name}.{role}_scale",
            "dtype": "float32",
            "shape": [1],
            "encoding": "little_endian_float32",
            **write_bytes(scale_rel, scales),
            "sources": [{
                "section_index": sections[_DECODE_KEY]["index"],
                "subgraph_index": 0,
                "tensor_index": t_info["index"],
                "tensor_name": t_info["name"],
                "field": "quantization.scale",
                "source_data_sha256": _sha256_bytes(scales),
            }],
        })
      if index % 35 == 0:
        print(
            "FC",
            index + 1,
            "of277",
            round(time.time() - start_time, 2),
            "s",
            flush=True,
        )

    # All learned RMSNorm weights mapped by actual consumers.
    for op in dec_trace["operators"]:
      if op.get("composite_name") != "odml.rms_norm":
        continue
      output_entry = op["outputs"][0]
      out_path = output_entry["name"]
      tensor_idx = op["inputs"][1]["index"]
      layer_match = re.search(r"/layer_(\d+)/", out_path)
      if "/value_norm/" in out_path:
        assert np.all(
            raw_tensor_bytes(_DECODE_KEY, 0, tensor_idx).view("<f4") == 1
        )
        stats["unit_value_norm_uses"] += 1
        continue
      if layer_match:
        role = out_path.split("/")[-2]
        suffix = {
            "pre_attention_norm": "input_layernorm",
            "query_norm": "self_attn.q_norm",
            "key_norm": "self_attn.k_norm",
            "post_attention_norm": "post_attention_layernorm",
            "pre_ffw_norm": "pre_feedforward_layernorm",
            "post_ffw_norm": "post_feedforward_layernorm",
            "post_per_layer_input_norm": "post_per_layer_input_norm",
        }.get(role)
        assert suffix is not None, out_path
        norm_name = f"model.layers.{int(layer_match.group(1))}.{suffix}.weight"
      elif "/per_layer_embedding_projection_norm/" in out_path:
        norm_name = "model.per_layer_projection_norm.weight"
      elif (
          op["index"] == fc_ops[-1]["index"] - 1
          and output_entry["index"] == fc_ops[-1]["inputs"][0]["index"]
      ):
        norm_name = "model.norm.weight"
      else:
        raise ValueError(f"Unmapped norm {out_path}")
      rec = export_float(norm_name, _DECODE_KEY, 0, tensor_idx)
      rec["mapped_by_consumer"] = {
          "operator_index": op["index"],
          "output_name": out_path,
      }
      stats["learned_norms"] += 1
    assert stats["learned_norms"] == 227

    # Every layer residual scalar is a stored float constant with a MUL
    # consumer.
    for op in dec_trace["operators"]:
      out_path = op["outputs"][0]["name"]
      if op["type"] != "MUL" or "._maybe_apply_skip_scale/mul" not in out_path:
        continue
      layer_match = re.search(r"/layer_(\d+)/", out_path)
      assert layer_match is not None
      layer = int(layer_match.group(1))
      static_indices = []
      for inp in op["inputs"]:
        tensor_obj = dec_subgraph.Tensors(inp["index"])
        buf_obj = dec_model.Buffers(tensor_obj.Buffer())
        if buf_obj.DataLength():
          static_indices.append(inp["index"])
      assert len(static_indices) == 1, out_path
      rec = export_float(
          f"model.layers.{layer}.layer_scalar",
          _DECODE_KEY,
          0,
          static_indices[0],
          [1],
      )
      rec["mapped_by_consumer"] = {
          "operator_index": op["index"],
          "output_name": out_path,
      }
      stats["layer_scalars"] += 1
    assert stats["layer_scalars"] == 35

    # Main table is independently sourced from the embedder section.
    emb_key = "tf_lite_embedder"
    emb_subgraph = models[emb_key].Subgraphs(0)
    tables = [
        i
        for i in range(emb_subgraph.TensorsLength())
        if _TENSOR_TYPES[emb_subgraph.Tensors(i).Type()] in ["INT2", "INT4"]
    ]
    assert len(tables) == 1
    export_quant("model.embed_tokens.weight", emb_key, 0, tables[0])
    print("Main embedding exported", flush=True)

    # Assemble all 35 per-layer packed tables into token-major lookup layout.
    ple_key = "tf_lite_per_layer_embedder"
    ple_model = models[ple_key]
    ple_subgraph = ple_model.Subgraphs(0)
    ple_indices = [
        i
        for i in range(ple_subgraph.TensorsLength())
        if _TENSOR_TYPES[ple_subgraph.Tensors(i).Type()] == "INT4"
    ]
    assert len(ple_indices) == 35
    parts = []
    scale_columns = []
    sources = []
    for layer, tensor_idx in enumerate(ple_indices):
      tensor_obj = ple_subgraph.Tensors(tensor_idx)
      assert tensor_obj.ShapeAsNumpy().tolist() == [262144, 256]
      quant = tensor_obj.Quantization()
      assert (
          quant.QuantizedDimension() == 0
          and quant.ScaleLength() == 262144
          and np.all(quant.ZeroPointAsNumpy() == 0)
      )
      parts.append(
          raw_tensor_bytes(ple_key, 0, tensor_idx).reshape(262144, 128)
      )
      scales = quant.ScaleAsNumpy()
      assert np.all(np.isfinite(scales)) and np.all(scales > 0)
      scale_columns.append(scales)
      sources.append({
          **build_source_record(ple_key, 0, tensor_idx),
          "destination_layer_partition": layer,
          "source_scale_sha256": _sha256_bytes(scales),
      })
    ple_name = "model.embed_tokens_per_layer.weight"
    ple_rel = f"tensors/{ple_name}.i4"
    ple_path = out_dir / ple_rel
    assert not ple_path.exists()
    ple_hasher = hashlib.sha256()
    with ple_path.open("wb") as ple_file:
      for row in range(0, 262144, 4096):
        end = min(row + 4096, 262144)
        block = np.empty((end - row, 35, 128), dtype=np.uint8)
        for layer, part in enumerate(parts):
          block[:, layer, :] = part[row:end]
        for layer, part in enumerate(parts):
          assert np.array_equal(block[:, layer, :], part[row:end])
        ple_file.write(memoryview(block).cast("B"))
        ple_hasher.update(memoryview(block).cast("B"))
    scale_matrix = np.stack(scale_columns, axis=1).astype("<f4", copy=False)
    scale_info = write_bytes(f"scales/{ple_name}.f32", scale_matrix)
    add_record({
        "name": ple_name,
        "dtype": "int4",
        "shape": [262144, 8960],
        "encoding": "signed_twos_complement_low_nibble_first",
        "file": ple_rel,
        "bytes": ple_path.stat().st_size,
        "sha256": ple_hasher.hexdigest(),
        "quantization": {
            "kind": "blockwise",
            "block_size": 256,
            "quantized_dimension": 0,
            "scales_file": scale_info["file"],
            "scales_shape": [262144, 35],
            "scales_dtype": "float32",
            "scales_bytes": scale_info["bytes"],
            "scales_sha256": scale_info["sha256"],
            "zero_points": [0],
        },
        "sources": sources,
    })
    stats["numeric_weight_codes_preserved"] += 262144 * 8960
    stats["per_layer_table_partitions"] = 35
    print(
        "Per-layer embeddings exported",
        round(time.time() - start_time, 2),
        "s",
        flush=True,
    )

    # Inventory every remaining main-graph FP32 constant.
    for tensor_idx in range(dec_subgraph.TensorsLength()):
      tensor_obj = dec_subgraph.Tensors(tensor_idx)
      buf_obj = dec_model.Buffers(tensor_obj.Buffer())
      if (
          _TENSOR_TYPES[tensor_obj.Type()] != "FLOAT32"
          or not buf_obj.DataLength()
          or tensor_idx in exported_main_float
      ):
        continue
      tensor_name = tensor_obj.Name().decode()
      raw_data = raw_tensor_bytes(_DECODE_KEY, 0, tensor_idx)
      consumers = [
          {
              "operator_index": op["index"],
              "type": op["type"],
              "output_name": op["outputs"][0]["name"],
          }
          for op in dec_trace["operators"]
          if any(i["index"] == tensor_idx for i in op["inputs"])
      ]
      const_record = {
          "name": f"decode.tensor_{tensor_idx}",
          "dtype": "float32",
          "shape": tensor_obj.ShapeAsNumpy().tolist(),
          **write_bytes(f"constants/decode.tensor_{tensor_idx}.f32", raw_data),
          "sources": [build_source_record(_DECODE_KEY, 0, tensor_idx)],
          "consumers": consumers,
      }
      if "jax2tf_arg_" in tensor_name or "ReadVariableOp" in tensor_name:
        unmapped.append(const_record)
      constants.append(const_record)

    # Composite decomposition scalar arithmetic constants.
    seen_buffers: set[int] = set()
    for op in dec_trace["operators"]:
      if "decomposition_subgraph" not in op:
        continue
      sg_idx = op["decomposition_subgraph"]
      subgraph = dec_model.Subgraphs(sg_idx)
      for tensor_idx in range(subgraph.TensorsLength()):
        tensor_obj = subgraph.Tensors(tensor_idx)
        buf_obj = dec_model.Buffers(tensor_obj.Buffer())
        if (
            _TENSOR_TYPES[tensor_obj.Type()] != "FLOAT32"
            or not buf_obj.DataLength()
            or tensor_obj.Buffer() in seen_buffers
        ):
          continue
        seen_buffers.add(tensor_obj.Buffer())
        constants.append({
            "name": f"decomposition_{sg_idx}.tensor_{tensor_idx}",
            "dtype": "float32",
            "shape": tensor_obj.ShapeAsNumpy().tolist(),
            **write_bytes(
                f"constants/decomposition_{sg_idx}.tensor_{tensor_idx}.f32",
                raw_tensor_bytes(_DECODE_KEY, sg_idx, tensor_idx),
            ),
            "sources": [build_source_record(_DECODE_KEY, sg_idx, tensor_idx)],
        })

    # Main/per-layer embedder fixed scalar constants.
    for sec_key in ["tf_lite_embedder", "tf_lite_per_layer_embedder"]:
      sec_model = models[sec_key]
      sec_subgraph = sec_model.Subgraphs(0)
      for tensor_idx in range(sec_subgraph.TensorsLength()):
        tensor_obj = sec_subgraph.Tensors(tensor_idx)
        buf_obj = sec_model.Buffers(tensor_obj.Buffer())
        if (
            _TENSOR_TYPES[tensor_obj.Type()] != "FLOAT32"
            or not buf_obj.DataLength()
        ):
          continue
        constants.append({
            "name": f"{sec_key}.tensor_{tensor_idx}",
            "dtype": "float32",
            "shape": tensor_obj.ShapeAsNumpy().tolist(),
            **write_bytes(
                f"constants/{sec_key}.tensor_{tensor_idx}.f32",
                raw_tensor_bytes(sec_key, 0, tensor_idx),
            ),
            "sources": [build_source_record(sec_key, 0, tensor_idx)],
        })

    kv_raw = json.loads(kv_path.read_text())["graphs"][0]["cache_updates"]
    kv_entries = [
        {
            "owner_layer": x["owner"],
            "key_scale": x["key_scale"],
            "value_scale": x["value_scale"],
            "zero_point": 0,
            "source_cache_update_operator": x["operator"],
            "source_decomposition_subgraph": x["cache_update_decomposition"],
        }
        for x in kv_raw
    ]
    kv_specs = []
    for entry in kv_entries:
      dim = 512 if entry["owner_layer"] % 5 == 4 else 256
      kv_specs.append({
          "owner": entry["owner_layer"],
          "head_dim": dim,
          "key_scale": entry["key_scale"],
          "value_scale": entry["value_scale"],
          "zero_point": 0,
          "key_shape_template": [1, 1, "capacity", dim],
          "value_shape_template": [1, 1, dim, "capacity"],
      })
    status_str = (
        "complete" if not unmapped else "incomplete_unmapped_float_coefficients"
    )
    manifest = {
        "schema_version": 1,
        "status": status_str,
        "source_bundle": {
            "path": inventory["path"],
            "bytes": inventory["bytes"],
            "sha256": _sha256_bytes(mm),
        },
        "source_inventory_sha256": _sha256_bytes(inv_path.read_bytes()),
        "exporter_sha256": _sha256_bytes(pathlib.Path(__file__).read_bytes()),
        "tensors": records,
        "constants": constants,
        "kv_cache": kv_entries,
        "kv_cache_specs": kv_specs,
        "unmapped_float_coefficients": unmapped,
        "coverage": dict(stats),
        "tensor_counts": dict(collections.Counter(r["dtype"] for r in records)),
        "export_seconds": time.time() - start_time,
        "constraints": [
            (
                "All weights and float coefficients come directly from"
                " published bundle buffers, not from CT."
            ),
            (
                "INT2 codes widened losslessly to signed INT4; model exporter"
                " restores the original compact INT2 representation."
            ),
            (
                "Fixed scalar and RoPE constants separately exported for"
                " explicit graph alignment."
            ),
            "No inference, builds or device actions performed.",
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        "MANIFEST",
        manifest["status"],
        "tensors",
        len(records),
        "constants",
        len(constants),
        "unmapped",
        len(unmapped),
        dict(stats),
        flush=True,
    )
    if unmapped:
      raise SystemExit(2)


if __name__ == "__main__":
  app.run(main)
