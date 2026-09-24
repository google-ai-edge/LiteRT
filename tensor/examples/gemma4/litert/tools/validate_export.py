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

"""Independent readback validation of exported weights, scales, and coverage."""

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
_BUNDLE_DIR = flags.DEFINE_string(
    "bundle_dir",
    None,
    "Path to the exported bundle directory containing manifest.json.",
    required=True,
)
_OUTPUT = flags.DEFINE_string(
    "output",
    None,
    "Path to the output validation report JSON file.",
    required=True,
)

_BLOCK_BYTES = 4 * 1024 * 1024


def _digest_file(file_path: pathlib.Path) -> str:
  """Computes the SHA-256 hex digest of a file."""
  hasher = hashlib.sha256()
  with file_path.open("rb") as input_file:
    while chunk := input_file.read(_BLOCK_BYTES):
      hasher.update(chunk)
  return hasher.hexdigest()


def _name_for_fc(weight_path: str) -> str:
  """Maps a TFLite fully-connected weight path to its canonical layer name."""
  if "per_layer_model_projection/" in weight_path:
    return "model.per_layer_model_projection"
  if "decode_softmax/" in weight_path:
    return "lm_head"
  layer_match = re.search(r"/layer_(\d+)/", weight_path)
  assert layer_match is not None, weight_path
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
  suffix = next(s for pat, s in patterns if pat in weight_path)
  return f"model.layers.{layer}.{suffix}"


def _expected_byte_count(dtype: str, element_count: int) -> int:
  """Returns the expected byte size for an exported tensor file."""
  if dtype == "float32":
    return element_count * 4
  if dtype == "int8":
    return element_count
  return (element_count + 1) // 2


def main(argv: Sequence[str]) -> None:
  """Validates all exported tensors and constants against the source bundle."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  schema_dir = pathlib.Path(_SCHEMA_DIR.value)
  if not (schema_dir / "tflite/Model.py").is_file():
    raise app.UsageError("--schema_dir must contain generated tflite/Model.py")

  root_dir = pathlib.Path(_BUNDLE_DIR.value).resolve()
  manifest = json.loads((root_dir / "manifest.json").read_text())
  assert (
      manifest["status"] == "complete"
      and not manifest["unmapped_float_coefficients"]
  )
  inventory = json.loads(pathlib.Path(_INVENTORY.value).read_text())
  trace = json.loads(pathlib.Path(_TRACE.value).read_text())
  output_path = pathlib.Path(_OUTPUT.value)
  if output_path.exists():
    raise FileExistsError(output_path)

  stats: collections.Counter[str] = collections.Counter()
  start_time = time.time()
  by_name = {x["name"]: x for x in manifest["tensors"]}
  assert len(by_name) == 1093

  with (
      open(inventory["path"], "rb") as bundle_file,
      mmap.mmap(bundle_file.fileno(), 0, access=mmap.ACCESS_READ) as mm,
  ):
    sections = {
        s["index"]: s
        for s in inventory["sections"]
        if s["type"] == "TFLiteModel"
    }
    models = {
        i: tflite_model.Model.GetRootAsModel(
            memoryview(mm)[s["begin"] : s["end"]], 0
        )
        for i, s in sections.items()
    }
    assert hashlib.sha256(mm).hexdigest() == manifest["source_bundle"]["sha256"]

    def get_source_tensor(src_spec: dict[str, Any]) -> Any:
      """Returns the TFLite tensor object for a source reference."""
      return (
          models[src_spec["section_index"]]
          .Subgraphs(src_spec["subgraph_index"])
          .Tensors(src_spec["tensor_index"])
      )

    def get_source_data(src_spec: dict[str, Any]) -> np.ndarray:
      """Returns the raw buffer numpy array for a source reference."""
      tensor_obj = get_source_tensor(src_spec)
      return (
          models[src_spec["section_index"]]
          .Buffers(tensor_obj.Buffer())
          .DataAsNumpy()
      )

    def verify_file(record: dict[str, Any]) -> pathlib.Path:
      """Verifies file size and SHA-256 digest against a manifest record."""
      target_path = (root_dir / record["file"]).resolve()
      assert target_path.is_relative_to(root_dir)
      assert target_path.stat().st_size == record["bytes"]
      assert _digest_file(target_path) == record["sha256"], target_path
      stats["files_checked"] += 1
      stats["file_bytes_checked"] += record["bytes"]
      return target_path

    for rec in manifest["tensors"]:
      tensor_path = verify_file(rec)
      count = int(np.prod(rec["shape"], dtype=np.int64))
      expected = _expected_byte_count(rec["dtype"], count)
      assert tensor_path.stat().st_size == expected, (
          rec["name"],
          expected,
          tensor_path.stat().st_size,
      )
      if "quantization" not in rec:
        arr = np.memmap(tensor_path, mode="r", dtype="<f4")
        assert np.isfinite(arr).all()
        src_info = rec["sources"][0]
        if src_info.get("field") == "quantization.scale":
          src_arr = get_source_tensor(src_info).Quantization().ScaleAsNumpy()
        else:
          src_arr = get_source_data(src_info).view("<f4")
        assert np.array_equal(
            arr.view("u4"), src_arr.view("u4").reshape(-1)
        ), rec["name"]
        stats["float_coefficients_or_activation_scales_checked"] += len(arr)
        continue

      quant = rec["quantization"]
      scale_path = root_dir / quant["scales_file"]
      assert scale_path.stat().st_size == quant["scales_bytes"]
      assert _digest_file(scale_path) == quant["scales_sha256"]
      stats["files_checked"] += 1
      stats["file_bytes_checked"] += quant["scales_bytes"]
      scales = np.memmap(
          scale_path, mode="r", dtype="<f4", shape=tuple(quant["scales_shape"])
      )
      assert np.isfinite(scales).all() and np.all(scales > 0)
      assert quant["zero_points"] == [0] and quant["quantized_dimension"] == 0
      if quant["kind"] == "per_channel":
        src_info = rec["sources"][0]
        src_bytes = get_source_data(src_info)
        tensor_obj = get_source_tensor(src_info)
        assert np.array_equal(
            scales.view("u4"),
            tensor_obj.Quantization().ScaleAsNumpy().view("u4"),
        )
        assert np.all(tensor_obj.Quantization().ZeroPointAsNumpy() == 0)
        out_bytes = np.memmap(tensor_path, mode="r", dtype="u1")
        if src_info["dtype"] == "INT2":
          # Decode source and destination independently into signed int8 values.
          for begin in range(0, len(src_bytes), _BLOCK_BYTES):
            sb = src_bytes[begin : begin + _BLOCK_BYTES]
            ob = out_bytes[begin * 2 : (begin + len(sb)) * 2]
            sx = np.stack(
                [((sb >> shift) & 3).astype(np.int8) for shift in [0, 2, 4, 6]],
                axis=1,
            ).reshape(-1)
            sx[sx >= 2] -= 4
            ox = np.stack(
                [(ob & 15).astype(np.int8), (ob >> 4).astype(np.int8)], axis=1
            ).reshape(-1)
            ox[ox >= 8] -= 16
            assert np.array_equal(sx, ox), rec["name"]
          stats["independently_decoded_int2_codes"] += count
        else:
          assert np.array_equal(out_bytes, src_bytes), rec["name"]
        assert (
            hashlib.sha256(src_bytes).hexdigest()
            == src_info["source_data_sha256"]
        )
      else:
        assert (
            quant["kind"] == "blockwise"
            and quant["block_size"] == 256
            and rec["shape"] == [262144, 8960]
            and len(rec["sources"]) == 35
        )
        out_bytes = np.memmap(
            tensor_path, mode="r", dtype="u1", shape=(262144, 35, 128)
        )
        parts = []
        for layer, src_info in enumerate(rec["sources"]):
          assert src_info["destination_layer_partition"] == layer
          src_bytes = get_source_data(src_info)
          assert (
              hashlib.sha256(src_bytes).hexdigest()
              == src_info["source_data_sha256"]
          )
          parts.append(src_bytes.reshape(262144, 128))
          assert np.array_equal(
              scales[:, layer].view("u4"),
              get_source_tensor(src_info)
              .Quantization()
              .ScaleAsNumpy()
              .view("u4"),
          )
        for begin in range(0, 262144, 4096):
          block = np.asarray(out_bytes[begin : begin + 4096])
          for layer, part_arr in enumerate(parts):
            assert np.array_equal(
                block[:, layer, :], part_arr[begin : begin + 4096]
            ), layer
        stats["combined_embedding_packed_codes_checked"] += count
      stats["weight_codes_checked"] += count
      stats["weight_scales_checked"] += scales.size

    for const_rec in manifest["constants"]:
      const_path = verify_file(const_rec)
      out_bytes = np.fromfile(const_path, dtype="u1")
      assert np.array_equal(out_bytes, get_source_data(const_rec["sources"][0]))
      stats["fixed_constants_checked"] += 1
    assert (
        stats["weight_codes_checked"] == 5030936576
        and stats["independently_decoded_int2_codes"] == 1937768448
    )

    cross_checks = []
    for graph in trace["graphs"]:
      counter = 0
      for op in graph["operators"]:
        if op["type"] != "FULLY_CONNECTED":
          continue
        weight_in = op["inputs"][1]
        fc_name = _name_for_fc(weight_in["name"])
        rec = by_name[f"{fc_name}.weight"]
        src_ref = {
            "section_index": 10,
            "subgraph_index": graph["index"],
            "tensor_index": weight_in["index"],
        }
        raw_data = get_source_data(src_ref)
        assert (
            hashlib.sha256(raw_data).hexdigest()
            == rec["sources"][0]["source_data_sha256"]
        ), (graph["name"], fc_name)
        assert np.array_equal(
            get_source_tensor(src_ref).Quantization().ScaleAsNumpy().view("u4"),
            np.fromfile(
                root_dir / rec["quantization"]["scales_file"], dtype="<u4"
            ),
        )
        for role, t_info in [
            ("input", op["inputs"][0]),
            ("output", op["outputs"][0]),
        ]:
          if "scales" in t_info:
            scale_file = root_dir / by_name[f"{fc_name}.{role}_scale"]["file"]
            assert np.array_equal(
                np.fromfile(scale_file, dtype="<f4"),
                np.array(t_info["scales"], dtype="<f4"),
            )
          else:
            assert f"{fc_name}.{role}_scale" not in by_name
        counter += 1
      cross_checks.append({
          "signature": graph["name"],
          "fc_weight_and_scale_arrays_exact": counter,
      })

    kv_updates = json.loads(pathlib.Path(_KV.value).read_text())["graphs"][0][
        "cache_updates"
    ]
    assert len(kv_updates) == 15
    for spec_a, spec_b in zip(manifest["kv_cache_specs"], kv_updates):
      assert (
          spec_a["owner"] == spec_b["owner"]
          and spec_a["key_scale"] == spec_b["key_scale"]
          and spec_a["value_scale"] == spec_b["value_scale"]
          and spec_a["zero_point"] == 0
      )
    result = {
        "status": "pass",
        "manifest_sha256": _digest_file(root_dir / "manifest.json"),
        "validator_sha256": _digest_file(pathlib.Path(__file__)),
        "counters": dict(stats),
        "cross_signature_fc_validation": cross_checks,
        "kv_owner_specs_checked": len(kv_updates),
        "validation_seconds": time.time() - start_time,
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print("VALIDATION", result["status"], result["counters"], flush=True)


if __name__ == "__main__":
  app.run(main)
