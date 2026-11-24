#!/usr/bin/env python3
# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Teacher-forced comparison of two backends' logits for the same tokens.

A greedy text is a poor test of a backend whose arithmetic is not the
reference's bit for bit: the two texts part at the first near-tie and are
incomparable after it. Fed the same token stream (LiteRT-LM's CLI does that
with --score_target_text: the prompt is prefilled, then every decode step reads
the target's next token whatever the model picked), the two backends' logits
compare at every step. LiteRT-LM's debugger (--enable_debugger) captures them
on the CPU; it is off for the NPU executor, whose logits the NVIDIA dispatch
writes instead (LITERT_NVIDIA_DISPATCH_DUMP_OUTPUTS=DIR:STEPS:MB).

  teacher_forced.py collect [--tokenizer FILE --target FILE] CAPTURE_DIR
  OUT_PREFIX
      The debugger's captures of one run (CAPTURE_DIR = the cache directory's
      litert_lm_debugger) -> OUT_PREFIX.logits.npy (float32 [steps, vocab]) and
      OUT_PREFIX.steps.json (each decode step's position and, given the
      model's tokenizer and the target text, the token the step scores: step
      i scores the target's token i). The tokenizer is the model's
      SentencePiece model, or its Hugging Face tokenizer.json (*.json, read
      with the `tokenizers` package).
  teacher_forced.py collect-dispatch [--tokenizer FILE --target FILE]
                                     [--vocab N] DUMP_DIR OUT_PREFIX
      The same from the NVIDIA dispatch's output dumps (DUMP_DIR/index.jsonl):
      every output as wide as the vocabulary (--vocab, default Gemma's
      262144; Qwen 3.8's is 248320), in the order the steps produced them, of
      the function that wrote the most of them (the decode step's, not a
      prefill's last row). They carry no positions: `compare` takes the
      reference's.
  teacher_forced.py replay-manifest --model FILE.tflite [--signature decode]
                            --function F [--steps N,...] DUMP_DIR LOG OUT_DIR
      A CPU reference where LiteRT-LM's CPU run does not finish (a long
      context): the steps of an NVIDIA run that recorded its inputs
      (LITERT_NVIDIA_DISPATCH_DUMP_IO=dir:DUMP_DIR; F is the signature's
      dispatch function, e.g. tensorrt_partition_0; LOG is the run's log,
      whose "NVIDIA dispatch io in F=input_I_tensor_T" lines give each input's
      tensor) -> OUT_DIR/manifest.txt for cpu_replay, which runs each step on
      the CPU as LiteRT-LM would. Inputs that the run sized by its context (the
      KV caches, the mask) are padded to the model's length with zeros, which
      changes no logit. Every step reads the NVIDIA run's state: the
      comparison measures the steps, not what the NVIDIA prefill and earlier
      steps wrote. Needs the `tflite` package.
  teacher_forced.py collect-replay [--tokenizer SPM --target FILE] OUT_DIR
  OUT_PREFIX
      cpu_replay's logits of every step of OUT_DIR/manifest.txt, with the
      steps' positions -> OUT_PREFIX.logits.npy and OUT_PREFIX.steps.json.
  teacher_forced.py compare [--json FILE] [--bins B,...] [--positions P,...]
                            REF_PREFIX TEST_PREFIX [REF_PREFIX TEST_PREFIX ...]
      Per step: the reference's top token and its margin over the runner-up,
      whether the test picks the same top token, how far the test's margin for
      the reference's two candidates is from the reference's, KL(ref || test),
      and the negative log-likelihood of the token fed next under both. Prints
      the agreement by the reference's margin, the margin error's
      distribution, and every statistic by position; --json writes the steps.
  teacher_forced.py gate [--json FILE] [--positions P,...] [--cutoff C]
                         [--limit L] [--factor F] [--nll N]
                         REF_PREFIX TEST_PREFIX [REF_PREFIX TEST_PREFIX ...]
      `compare`, then the pass bar of gate(), whose verdict is the last line
      and the exit status (0 pass, 1 fail).

No GPU and no LiteRT are needed: numpy only.
"""

import argparse
import json
import mmap
from pathlib import Path
import re
import struct
import sys

import numpy as np

# Optional packages: only --tokenizer needs sentencepiece (or tokenizers, for
# a tokenizer.json) and only replay-manifest needs tflite. Every other command
# runs with numpy alone.
try:
  import sentencepiece  # pylint: disable=g-import-not-at-top
except ImportError:
  sentencepiece = None
try:
  import tokenizers  # pylint: disable=g-import-not-at-top
except ImportError:
  tokenizers = None
try:
  import tflite  # pylint: disable=g-import-not-at-top
except ImportError:
  tflite = None

_DTYPES = {
    "F32": np.float32,
    "F16": np.float16,
    "I32": np.int32,
    "I64": np.int64,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def read_safetensors(path):
  """Reads the tensors of a safetensors file into {name: array}."""
  with open(path, "rb") as f:
    size = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(size))
    data = f.read()
  tensors = {}
  for name, info in header.items():
    if name == "__metadata__":
      continue
    begin, end = info["data_offsets"]
    array = np.frombuffer(data[begin:end], dtype=_DTYPES[info["dtype"]])
    tensors[name] = array.reshape(info["shape"])
  return tensors


def _single(path):
  tensors = read_safetensors(path)
  if len(tensors) != 1:
    raise ValueError(f"{path}: {len(tensors)} tensors, one expected")
  return next(iter(tensors.values()))


def target_tokens(tokenizer_path, target_path):
  """Tokenizes the target as LiteRT-LM's scoring does (no BOS)."""
  text = Path(target_path).read_text()
  if str(tokenizer_path).endswith(".json"):
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    return tokenizer.encode(text, add_special_tokens=False).ids
  processor = sentencepiece.SentencePieceProcessor(model_file=tokenizer_path)
  return processor.encode(text)


def collect(capture_dir, out_prefix, scored=None):
  """Collects one run's decode steps from the debugger's captures."""
  sessions = [p for p in Path(capture_dir).iterdir() if p.is_dir()]
  if len(sessions) != 1:
    raise ValueError(f"{capture_dir}: {len(sessions)} sessions, one expected")
  session = sessions[0]
  pattern = re.compile(r"decode_post_logits_step_(\d+)\.safetensors")
  steps = sorted(
      int(m.group(1))
      for m in (pattern.fullmatch(p.name) for p in session.iterdir())
      if m
  )
  if not steps:
    raise ValueError(f"{session}: no decode logits captured")
  logits = np.stack([
      _single(session / f"decode_post_logits_step_{s}.safetensors")
      .astype(np.float32)
      .reshape(-1)
      for s in steps
  ])
  if scored is not None and len(scored) != len(steps):
    raise ValueError(
        f"{session}: {len(steps)} decode steps for a target of "
        f"{len(scored)} tokens"
    )
  records = []
  for i, s in enumerate(steps):
    record = {"step": s}
    path = session / f"decode_pre_input_pos_step_{s}.safetensors"
    if path.exists():
      record["position"] = int(_single(path).reshape(-1)[0])
    if scored is not None:
      record["scored"] = int(scored[i])
    records.append(record)
  np.save(f"{out_prefix}.logits.npy", logits)
  Path(f"{out_prefix}.steps.json").write_text(json.dumps(records) + "\n")
  print(
      f"{out_prefix}: {len(steps)} decode steps, vocabulary {logits.shape[1]}"
  )


def collect_dispatch(dump_dir, out_prefix, scored=None, vocab=262144):
  """Collects one run's decode steps from the NVIDIA dispatch's output dumps."""
  entries = [json.loads(line) for line in open(Path(dump_dir) / "index.jsonl")]
  entries = sorted(
      (e for e in entries if e["dims"] and e["dims"][-1] == vocab),
      key=lambda e: e["seq"],
  )
  if not entries:
    raise ValueError(f"{dump_dir}: no output of {vocab} columns dumped")
  # A prefill that also writes logits (its last row) does so once a chunk of
  # the prompt; the decode step once a step.
  counts = {}
  for e in entries:
    counts[e["function"]] = counts.get(e["function"], 0) + 1
  decode = max(counts, key=counts.get)
  entries = [e for e in entries if e["function"] == decode]
  rows = []
  for e in entries:
    if e["skipped"]:
      raise ValueError(
          f"{dump_dir}: step {e['step']} of {e['function']} was not "
          "written (raise the dump's size cap)"
      )
    raw = (Path(dump_dir) / e["file"]).read_bytes()
    dtype = np.float16 if len(raw) == 2 * vocab else np.float32
    rows.append(np.frombuffer(raw, dtype=dtype).astype(np.float32).reshape(-1))
  logits = np.stack(rows)
  if scored is not None and len(scored) != len(rows):
    raise ValueError(
        f"{dump_dir}: {len(rows)} decode steps for a target of "
        f"{len(scored)} tokens"
    )
  records = [{"step": e["step"]} for e in entries]
  if scored is not None:
    for record, token in zip(records, scored):
      record["scored"] = int(token)
  np.save(f"{out_prefix}.logits.npy", logits)
  Path(f"{out_prefix}.steps.json").write_text(json.dumps(records) + "\n")
  print(f"{out_prefix}: {len(rows)} decode steps, vocabulary {logits.shape[1]}")


_DISPATCH_INPUT = re.compile(
    r"NVIDIA dispatch io in (\S+?)=input_(\d+)_tensor_(\d+) "
)

# Bits per element of TFLite's tensor types.
_TFLITE_BITS = {
    0: 32,
    1: 16,
    2: 32,
    3: 8,
    4: 64,
    6: 8,
    7: 16,
    9: 8,
    10: 64,
    17: 4,
}


def find_signature(model, signature):
  """Returns the SignatureDef of MODEL whose key is SIGNATURE, or None."""
  for i in range(model.SignatureDefsLength()):
    definition = model.SignatureDefs(i)
    if definition.SignatureKey().decode() == signature:
      return definition
  return None


def signature_inputs(model_path, signature):
  """Returns {tensor index: (name, shape, bits)} of a signature's inputs."""
  with open(model_path, "rb") as f:
    buffer = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
  model = tflite.Model.GetRootAsModel(buffer, 0)
  definition = find_signature(model, signature)
  if definition is None:
    raise ValueError(f"{model_path}: no signature {signature}")
  subgraph = model.Subgraphs(definition.SubgraphIndex())
  inputs = {}
  for j in range(definition.InputsLength()):
    index = definition.Inputs(j).TensorIndex()
    tensor = subgraph.Tensors(index)
    inputs[index] = (
        definition.Inputs(j).Name().decode(),
        tuple(int(d) for d in tensor.ShapeAsNumpy()),
        _TFLITE_BITS[tensor.Type()],
    )
  return inputs


def dispatch_input_tensors(log_path, function):
  """Reads FUNCTION's input tensor indices from the log, in dispatch order."""
  tensors = []
  with open(log_path, errors="replace") as log:
    for line in log:
      m = _DISPATCH_INPUT.search(line)
      if not m or m.group(1) != function:
        continue
      if int(m.group(2)) < len(tensors):
        break
      tensors.append(int(m.group(3)))
  if not tensors:
    raise ValueError(f"{log_path}: no inputs of {function} logged")
  return tensors


def pad_to_shape(raw, shape, bits):
  """Pads a tensor recorded with a shorter sequence axis with zeros to SHAPE.

  Args:
    raw: The recorded tensor's bytes.
    shape: The tensor's shape in the model. Its longest axis is the sequence
      axis.
    bits: The bits per element.

  Returns:
    The padded bytes and the recorded length of the sequence axis.
  """
  axis = int(np.argmax(shape))
  others = int(np.prod(shape)) // shape[axis]
  count = len(raw) * 8 // bits
  length = count // others
  if length * others != count or length > shape[axis]:
    raise ValueError(f"{len(raw)} bytes do not fit {shape}")
  recorded = list(shape)
  recorded[axis] = length
  if bits == 4:
    packed = np.frombuffer(raw, dtype=np.uint8)
    values = np.empty(packed.size * 2, dtype=np.uint8)
    values[0::2] = packed & 0x0F
    values[1::2] = packed >> 4
  else:
    values = np.frombuffer(raw, dtype=f"u{bits // 8}")
  padded = np.zeros(shape, dtype=values.dtype)
  padded[tuple(slice(0, d) for d in recorded)] = values[:count].reshape(
      recorded
  )
  flat = padded.reshape(-1)
  if bits == 4:
    flat = np.append(flat, np.uint8(0)) if flat.size % 2 else flat
    return (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8).tobytes(), length
  return flat.tobytes(), length


def replay_manifest(
    dump_dir,
    log_path,
    out_dir,
    function,
    model_path,
    signature="decode",
    only=(),
):
  """Writes cpu_replay's manifest for FUNCTION's recorded steps (or ONLY)."""
  inputs = signature_inputs(model_path, signature)
  tensors = dispatch_input_tensors(log_path, function)
  if sorted(tensors) != sorted(inputs):
    raise ValueError(
        f"{function} has the inputs {tensors}, {signature} {sorted(inputs)}"
    )
  root = Path(dump_dir).resolve() / function
  steps = sorted(
      int(p.name[4:])
      for p in root.iterdir()
      if re.fullmatch(r"step\d+", p.name)
      and (not only or int(p.name[4:]) in only)
  )
  out = Path(out_dir).resolve()
  lines = []
  lengths = set()
  for step in steps:
    files = []
    for i, tensor in enumerate(tensors):
      name, shape, bits = inputs[tensor]
      path = root / f"step{step}" / f"input_{i}.raw"
      if path.stat().st_size * 8 != int(np.prod(shape)) * bits:
        raw, length = pad_to_shape(path.read_bytes(), shape, bits)
        lengths.add(length)
        path = out / "inputs" / f"step{step}" / f"{name}.raw"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
      files.append(f"{name}={path}")
    lines.append(" ".join([str(out / f"step{step}")] + files))
  out.mkdir(parents=True, exist_ok=True)
  (out / "manifest.txt").write_text("".join(line + "\n" for line in lines))
  padded = f", padded from {sorted(lengths)}" if lengths else ""
  print(
      f"{out / 'manifest.txt'}: {len(steps)} steps of {len(tensors)} "
      f"inputs{padded}"
  )


def collect_replay(out_dir, out_prefix, scored=None):
  """Collects cpu_replay's logits and positions of the manifest's steps."""
  rows = []
  records = []
  for line in (Path(out_dir) / "manifest.txt").read_text().splitlines():
    fields = line.split()
    prefix, files = fields[0], dict(f.split("=", 1) for f in fields[1:])
    rows.append(np.fromfile(f"{prefix}.logits.raw", dtype=np.float32))
    records.append({
        "step": int(Path(prefix).name[4:]),
        "position": int(np.fromfile(files["input_pos"], dtype=np.int32)[0]),
    })
  if scored is not None and len(scored) != len(rows):
    raise ValueError(
        f"{out_dir}: {len(rows)} steps for a target of {len(scored)} tokens"
    )
  if scored is not None:
    for record, token in zip(records, scored):
      record["scored"] = int(token)
  logits = np.stack(rows)
  np.save(f"{out_prefix}.logits.npy", logits)
  Path(f"{out_prefix}.steps.json").write_text(json.dumps(records) + "\n")
  print(
      f"{out_prefix}: {len(rows)} replayed steps, vocabulary {logits.shape[1]}"
  )


def _log_softmax(logits):
  top = logits.max(axis=-1, keepdims=True)
  shifted = logits.astype(np.float64) - top
  return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def compare_steps(ref, test, scored, positions):
  """Computes per-step statistics of TEST against REF.

  Args:
    ref: The reference's logits, [steps, vocab].
    test: The test's logits, [steps, vocab].
    scored: scored[i] is the token step i is scored on (the one fed to the next
      step), or None.
    positions: positions[i] is the position of step i, or None.

  Returns:
    One dict of statistics per step.
  """
  if ref.shape != test.shape:
    raise ValueError(f"shapes differ: {ref.shape} and {test.shape}")
  ref_lp = _log_softmax(ref)
  test_lp = _log_softmax(test)
  rows = []
  for i in range(ref.shape[0]):
    order = np.argsort(-ref[i], kind="stable")[:2]
    first, second = int(order[0]), int(order[1])
    test_first = int(np.argmax(test[i]))
    ref_margin = float(ref[i, first] - ref[i, second])
    test_margin = float(test[i, first] - test[i, second])
    p = np.exp(ref_lp[i])
    row = {
        "step": i,
        "position": positions[i],
        "ref_top": first,
        "ref_second": second,
        "test_top": test_first,
        "agree": test_first == first,
        "ref_margin": ref_margin,
        "test_margin": test_margin,
        "margin_error": test_margin - ref_margin,
        # The test's own margin of its pick over the reference's (positive when
        # they disagree): how far the flip went.
        "flip_margin": float(test[i, test_first] - test[i, first]),
        "top_logit_error": float(test[i, first] - ref[i, first]),
        "kl": float(np.sum(p * (ref_lp[i] - test_lp[i]))),
    }
    if scored[i] is not None:
      row["scored"] = scored[i]
      row["ref_nll"] = float(-ref_lp[i, scored[i]])
      row["test_nll"] = float(-test_lp[i, scored[i]])
    rows.append(row)
  return rows


def _load_run(prefix):
  logits = np.load(f"{prefix}.logits.npy")
  steps = json.loads(Path(f"{prefix}.steps.json").read_text())
  return logits, steps


def _quantiles(values):
  values = np.asarray(values, dtype=np.float64)
  if values.size == 0:
    return "-"
  return (
      f"median {np.median(values):.3f}, 90% {np.quantile(values, 0.9):.3f}, "
      f"99% {np.quantile(values, 0.99):.3f}, max {values.max():.3f}"
  )


def summarize(rows, bins, position_edges, out=sys.stdout):
  """Prints the tables of a comparison over all rows."""
  rows = list(rows)
  print(
      f"steps: {len(rows)}, top-1 agreement"
      f" {np.mean([r['agree'] for r in rows]):.4f}",
      file=out,
  )
  print("by the reference's top-2 margin:", file=out)
  edges = list(bins) + [float("inf")]
  for lo, hi in zip([0.0] + list(bins), edges):
    part = [r for r in rows if lo <= r["ref_margin"] < hi]
    flips = [r for r in part if not r["agree"]]
    worst = max((r["ref_margin"] for r in flips), default=None)
    print(
        f"  [{lo:g}, {hi:g}): {len(part)} steps, {len(flips)} disagree"
        + (
            f" (largest reference margin among them {worst:.3f})"
            if flips
            else ""
        ),
        file=out,
    )
  errors = [abs(r["margin_error"]) for r in rows]
  print(
      "|margin error| (test - reference, the reference's two candidates):"
      f" {_quantiles(errors)}",
      file=out,
  )
  print(
      f"KL(reference || test): mean {np.mean([r['kl'] for r in rows]):.5f}, "
      f"{_quantiles([r['kl'] for r in rows])}",
      file=out,
  )
  scored = [r for r in rows if "ref_nll" in r]
  if scored:
    ref_nll = np.mean([r["ref_nll"] for r in scored])
    test_nll = np.mean([r["test_nll"] for r in scored])
    print(
        f"mean NLL of the fed tokens over {len(scored)} steps: reference"
        f" {ref_nll:.4f}, test {test_nll:.4f}, test - reference"
        f" {test_nll - ref_nll:+.4f}",
        file=out,
    )
  if position_edges:
    print("by position:", file=out)
    edges = list(position_edges) + [float("inf")]
    for lo, hi in zip([0] + list(position_edges), edges):
      part = [
          r
          for r in rows
          if r["position"] is not None and lo <= r["position"] < hi
      ]
      if not part:
        continue
      part_scored = [r for r in part if "ref_nll" in r]
      nll = ""
      if part_scored:
        nll = (
            f", NLL ref {np.mean([r['ref_nll'] for r in part_scored]):.4f} "
            f"test {np.mean([r['test_nll'] for r in part_scored]):.4f}"
        )
      print(
          f"  [{lo:g}, {hi:g}): {len(part)} steps, agree"
          f" {np.mean([r['agree'] for r in part]):.4f}, |margin error| median"
          f" {np.median([abs(r['margin_error']) for r in part]):.3f} 90%"
          f" {np.quantile([abs(r['margin_error']) for r in part], 0.9):.3f} 99%"
          f" {np.quantile([abs(r['margin_error']) for r in part], 0.99):.3f}"
          f" max {max(abs(r['margin_error']) for r in part):.3f}, KL mean"
          f" {np.mean([r['kl'] for r in part]):.5f}{nll}",
          file=out,
      )
  flips = [r for r in rows if not r["agree"]]
  if flips:
    print(
        "disagreements (reference margin, test's pick over the reference's"
        " top):",
        file=out,
    )
    for r in sorted(flips, key=lambda r: -r["ref_margin"]):
      print(
          f"  {r.get('run', '')} step {r['step']} position {r['position']}:"
          f" reference {r['ref_top']} by {r['ref_margin']:.3f}, test"
          f" {r['test_top']} by {r['flip_margin']:.3f}",
          file=out,
      )


def compare(pairs, json_path=None, bins=(), position_edges=()):
  rows = []
  for ref_prefix, test_prefix in pairs:
    ref, ref_steps = _load_run(ref_prefix)
    test, test_steps = _load_run(test_prefix)
    if len(ref_steps) != len(test_steps):
      raise ValueError(
          f"{ref_prefix} has {len(ref_steps)} steps, {test_prefix}"
          f" {len(test_steps)}"
      )
    scored = [s.get("scored") for s in ref_steps]
    positions = [s.get("position") for s in ref_steps]
    test_positions = [
        s.get("position", p) for s, p in zip(test_steps, positions)
    ]
    if (
        scored != [s.get("scored") for s in test_steps]
        or positions != test_positions
    ):
      raise ValueError(f"{ref_prefix} and {test_prefix} were fed other tokens")
    for row in compare_steps(ref, test, scored, positions):
      row["run"] = Path(test_prefix).name
      rows.append(row)
  summarize(rows, bins, position_edges)
  if json_path:
    with open(json_path, "w") as f:
      for row in rows:
        f.write(json.dumps(row) + "\n")
  return rows


def _spread(rows):
  """Returns the |margin error|'s median, 90% and 99%, and the mean KL."""
  errors = np.abs([r["margin_error"] for r in rows])
  return (
      float(np.median(errors)),
      float(np.quantile(errors, 0.9)),
      float(np.quantile(errors, 0.99)),
      float(np.mean([r["kl"] for r in rows])),
  )


def gate(
    rows,
    position_edges=(),
    cutoff=2.0,
    limit=3.5,
    factor=1.5,
    nll_limit=0.03,
    min_bucket=30,
    out=sys.stdout,
):
  """Judges a comparison's rows by a pass bar and prints the verdict.

  The bar's numbers are those measured on a statically quantized 3B model's
  verification set against the CPU (689 steps, positions up to 1,063), where
  the margin error's 99th percentile was 1.5 to 1.9 logits and its maximum
  2.8 to 3.4 in every build:

  1. The noise cutoff: where the reference leads by more than CUTOFF logits,
     the test picks the reference's top token. Each disagreement above the
     cutoff is listed to be explained; one where the reference's margin or the
     step's |margin error| exceeds LIMIT fails the bar.
  2. No growth with position: in every position bucket after the first that
     holds MIN_BUCKET steps or more, the |margin error|'s median, 90% and 99%
     and the mean KL are at most FACTOR times the first bucket's.
  3. The mean NLL of the fed tokens under the test is at most NLL_LIMIT nats
     a token above the reference's.

  Args:
    rows: The steps' statistics (compare_steps), with their positions.
    position_edges: The positions that separate the buckets of criterion 2.
    cutoff: The reference's margin above which a disagreement is listed.
    limit: The reference's margin, or |margin error|, above which one fails.
    factor: How much a later bucket's spread may exceed the first bucket's.
    nll_limit: How much the test's mean NLL may exceed the reference's.
    min_bucket: The fewest steps of a bucket that is judged.
    out: Where the verdict is printed.

  Returns:
    Whether the bar is passed.
  """
  rows = list(rows)
  failed = []
  above = [r for r in rows if not r["agree"] and r["ref_margin"] > cutoff]
  beyond = [
      r
      for r in above
      if r["ref_margin"] > limit or abs(r["margin_error"]) > limit
  ]
  print(
      f"1. noise cutoff {cutoff:g}: {len(above)} disagreements above it,"
      f" {len(beyond)} beyond {limit:g}",
      file=out,
  )
  for r in sorted(above, key=lambda r: -r["ref_margin"]):
    print(
        f"   {r.get('run', '')} step {r['step']} position {r['position']}:"
        f" reference margin {r['ref_margin']:.3f}, margin error"
        f" {r['margin_error']:+.3f}, KL {r['kl']:.5f}",
        file=out,
    )
  if beyond:
    failed.append("1")
  edges = [0] + list(position_edges) + [float("inf")]
  buckets = []
  for lo, hi in zip(edges[:-1], edges[1:]):
    part = [
        r
        for r in rows
        if r["position"] is not None and lo <= r["position"] < hi
    ]
    if part:
      buckets.append((lo, hi, part))
  if len(buckets) < 2:
    print(
        "2. growth with position: one position bucket, not measured", file=out
    )
  else:
    first = _spread(buckets[0][2])
    print(
        f"2. growth with position, at most x{factor:g} of [{buckets[0][0]:g},"
        f" {buckets[0][1]:g}) ({len(buckets[0][2])} steps, |margin error|"
        f" median {first[0]:.3f}, 90% {first[1]:.3f}, 99% {first[2]:.3f}, KL"
        f" mean {first[3]:.5f}):",
        file=out,
    )
    for lo, hi, part in buckets[1:]:
      if len(part) < min_bucket:
        print(
            f"   [{lo:g}, {hi:g}): {len(part)} steps, too few to judge",
            file=out,
        )
        continue
      ratios = [b / max(a, 1e-9) for a, b in zip(first, _spread(part))]
      grew = max(ratios) > factor
      print(
          f"   [{lo:g}, {hi:g}): {len(part)} steps, median x{ratios[0]:.2f},"
          f" 90% x{ratios[1]:.2f}, 99% x{ratios[2]:.2f}, KL"
          f" x{ratios[3]:.2f}{', grew' if grew else ''}",
          file=out,
      )
      if grew and "2" not in failed:
        failed.append("2")
  scored = [r for r in rows if "ref_nll" in r]
  if scored:
    excess = np.mean([r["test_nll"] for r in scored]) - np.mean(
        [r["ref_nll"] for r in scored]
    )
    print(
        f"3. NLL of the fed tokens: test - reference {excess:+.4f} nats a"
        f" token over {len(scored)} steps (at most {nll_limit:+g})",
        file=out,
    )
    if excess > nll_limit:
      failed.append("3")
  else:
    print("3. NLL of the fed tokens: no scored tokens, not measured", file=out)
    failed.append("3")
  if failed:
    print(f"gate: FAIL (criteria {', '.join(failed)})", file=out)
  elif above:
    print(
        f"gate: PASS, {len(above)} disagreements above the cutoff to explain",
        file=out,
    )
  else:
    print("gate: PASS", file=out)
  return not failed


def main(argv):
  # The flags are parsed with argparse on purpose. The tool has no build rule
  # and runs by path with a python3 that has numpy, so it does not depend on
  # absl-py, and argparse provides its subcommands.
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  sub = parser.add_subparsers(dest="command", required=True)
  c = sub.add_parser("collect")
  c.add_argument("--tokenizer")
  c.add_argument("--target")
  c.add_argument("capture_dir")
  c.add_argument("out_prefix")
  d = sub.add_parser("collect-dispatch")
  d.add_argument("--tokenizer")
  d.add_argument("--target")
  d.add_argument("--vocab", type=int, default=262144)
  d.add_argument("dump_dir")
  d.add_argument("out_prefix")
  r = sub.add_parser("replay-manifest")
  r.add_argument("--model", required=True)
  r.add_argument("--signature", default="decode")
  r.add_argument("--function", required=True)
  r.add_argument("--steps", default="")
  r.add_argument("dump_dir")
  r.add_argument("log")
  r.add_argument("out_dir")
  y = sub.add_parser("collect-replay")
  y.add_argument("--tokenizer")
  y.add_argument("--target")
  y.add_argument("out_dir")
  y.add_argument("out_prefix")
  for name in ("compare", "gate"):
    m = sub.add_parser(name)
    m.add_argument("--json")
    m.add_argument("--bins", default="0.25,0.5,1,1.5,2,3")
    m.add_argument("--positions", default="")
    m.add_argument("prefixes", nargs="+")
    if name == "gate":
      m.add_argument("--cutoff", type=float, default=2.0)
      m.add_argument("--limit", type=float, default=3.5)
      m.add_argument("--factor", type=float, default=1.5)
      m.add_argument("--nll", type=float, default=0.03)
  args = parser.parse_args(argv)
  if args.command == "collect":
    scored = None
    if args.tokenizer and args.target:
      scored = target_tokens(args.tokenizer, args.target)
    collect(args.capture_dir, args.out_prefix, scored)
    return 0
  if args.command == "collect-dispatch":
    scored = None
    if args.tokenizer and args.target:
      scored = target_tokens(args.tokenizer, args.target)
    collect_dispatch(args.dump_dir, args.out_prefix, scored, args.vocab)
    return 0
  if args.command == "replay-manifest":
    replay_manifest(
        args.dump_dir,
        args.log,
        args.out_dir,
        args.function,
        args.model,
        args.signature,
        [int(n) for n in args.steps.split(",") if n],
    )
    return 0
  if args.command == "collect-replay":
    scored = None
    if args.tokenizer and args.target:
      scored = target_tokens(args.tokenizer, args.target)
    collect_replay(args.out_dir, args.out_prefix, scored)
    return 0
  if len(args.prefixes) % 2:
    parser.error("prefixes come in pairs: REF TEST")
  pairs = list(zip(args.prefixes[0::2], args.prefixes[1::2]))
  bins = [float(b) for b in args.bins.split(",") if b]
  positions = [int(p) for p in args.positions.split(",") if p]
  rows = compare(pairs, args.json, bins, positions)
  if args.command == "gate":
    passed = gate(
        rows, positions, args.cutoff, args.limit, args.factor, args.nll
    )
    return 0 if passed else 1
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
