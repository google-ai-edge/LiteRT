#!/usr/bin/env python3
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
"""Repack a mobile-QAT GGUF's TQ2_0 tensors into Q2_0, losslessly, for llama.cpp's CUDA backend.

The E2B and E4B comparison models are Google's mobile-QAT checkpoints as Unsloth's GGUFs
(gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf, gemma-4-E4B-it-qat-UD-Q2_K_XL.gguf), whose 2-bit tensors
are TQ2_0. llama.cpp v0.8.0's CUDA backend has no TQ2_0 kernels: it reports full layer offload
and computes those tensors on the CPU. Q2_0 decodes each code the same way ((code - 1) * scale)
in a layout the CUDA backend runs:

  TQ2_0: 256 codes, bit-interleaved, one FP16 scale at the end (66 bytes)
  Q2_0:   64 sequential codes, one FP16 scale in front (18 bytes)

so one TQ2_0 block becomes four Q2_0 blocks carrying copies of its scale; every dequantized
value is identical, including negative scales and signed zeros. The file grows from 2.0625 to
2.25 bits per 2-bit weight. Other tensors are copied byte for byte.

  python3 llama/repack_mobile_qat.tool SOURCE.gguf DEST.gguf [--llama-bin DIR]   (needs numpy)

DEST is written beside the tensors' new offsets and then read back: with --llama-bin (the
directory of ggml-base.dll / libggml-base.so, LLAMA_BIN when unset) every converted block is
dequantized by the release's own TQ2_0 and Q2_0 decoders and compared bit for bit; without
it the decoding is done here. Prints one JSON line with the result; the runners record a
repacked file by its name (…-cuda-Q2_0-repacked.gguf, models.json's llama.files)."""
import argparse
import ctypes
import hashlib
import json
import math
import os
import pathlib
import struct
import sys

import numpy as np

TQ2_0, Q2_0 = 35, 42  # ggml_type ids (ggml.h), as llama.cpp v0.8.0 names them
TQ2_0_BYTES, TQ2_0_BLOCK = 66, 256
Q2_0_BYTES, Q2_0_BLOCK = 18, 64
SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def read_header(path):
  """The GGUF header: version, metadata (tokenizer arrays skipped), the tensor descriptors
  with the file positions of their type and offset fields, and the data section's start."""
  with path.open('rb') as f:
    def number(fmt):
      return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]

    def string(keep=True):
      n = number('Q')
      if keep:
        return f.read(n).decode('utf-8')
      f.seek(n, 1)

    def value(kind, keep=True):
      if kind == 8:
        return string(keep)
      if kind == 9:
        subtype, count = number('I'), number('Q')
        if not keep and subtype in SCALARS:
          f.seek(count * struct.calcsize('<' + SCALARS[subtype]), 1)
          return None
        values = [value(subtype, keep) for _ in range(count)]
        return values if keep else None
      if kind not in SCALARS:
        raise ValueError(f'unknown GGUF value type {kind}')
      v = number(SCALARS[kind])
      return v if keep else None

    if f.read(4) != b'GGUF':
      raise SystemExit(f'{path} is not a GGUF file')
    version, tensor_count, kv_count = number('I'), number('Q'), number('Q')
    if version not in (2, 3):
      raise SystemExit(f'{path}: GGUF version {version} is not supported')
    metadata = {}
    for _ in range(kv_count):
      key, kind = string(), number('I')
      keep = not key.startswith('tokenizer.')
      v = value(kind, keep)
      if keep:
        metadata[key] = v
    tensors = []
    for _ in range(tensor_count):
      name, dimensions = string(), number('I')
      shape = [number('Q') for _ in range(dimensions)]
      type_position = f.tell()
      kind, offset = number('I'), number('Q')
      tensors.append({'name': name, 'shape': shape, 'elements': math.prod(shape), 'type': kind, 'offset': offset,
                      'type_position': type_position, 'offset_position': type_position + 4})
    end = f.tell()
  align = metadata.get('general.alignment', 32)
  return {'version': version, 'metadata': metadata, 'tensors': tensors, 'alignment': align,
          'data_offset': (end + align - 1) // align * align}


def tensor_bytes(tensor, kind, type_sizes):
  bytes_per_block, block = type_sizes[kind]
  if tensor['elements'] % block:
    raise SystemExit(f"{tensor['name']}: {tensor['elements']} elements are not whole blocks of {block}")
  return tensor['elements'] // block * bytes_per_block


def pack(data):
  """TQ2_0 blocks -> Q2_0 blocks (four per block, the scale copied in front of each)."""
  blocks = np.frombuffer(data, dtype=np.uint8).reshape(-1, TQ2_0_BYTES)
  quants = blocks[:, :64].reshape(-1, 2, 32)
  codes = ((quants[:, :, None, :] >> np.array([0, 2, 4, 6], dtype=np.uint8)[None, None, :, None]) & 3).reshape(-1, 4, 16, 4)
  out = np.empty((len(blocks), 4, Q2_0_BYTES), dtype=np.uint8)
  out[:, :, :2] = blocks[:, None, 64:66]
  out[:, :, 2:] = codes[:, :, :, 0] | (codes[:, :, :, 1] << 2) | (codes[:, :, :, 2] << 4) | (codes[:, :, :, 3] << 6)
  return out.tobytes()


def decoders(llama_bin):
  """(dequantize_tq2_0, dequantize_q2_0): the release's own when llama_bin names its ggml-base
  library, else this file's reference decoding."""
  if llama_bin:
    for name in ('ggml-base.dll', 'libggml-base.so', 'libggml-base.dylib'):
      lib_path = pathlib.Path(llama_bin) / name
      if lib_path.exists():
        break
    else:
      raise SystemExit(f'no ggml-base library in {llama_bin}')
    if os.name == 'nt':
      os.add_dll_directory(str(pathlib.Path(llama_bin).resolve()))
    lib = ctypes.CDLL(str(lib_path))
    for fn in (lib.dequantize_row_tq2_0, lib.dequantize_row_q2_0):
      fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64], None
    lib.ggml_type_size.argtypes, lib.ggml_type_size.restype = [ctypes.c_int], ctypes.c_size_t
    lib.ggml_blck_size.argtypes, lib.ggml_blck_size.restype = [ctypes.c_int], ctypes.c_int64
    if (lib.ggml_type_size(TQ2_0), lib.ggml_blck_size(TQ2_0)) != (TQ2_0_BYTES, TQ2_0_BLOCK) or \
       (lib.ggml_type_size(Q2_0), lib.ggml_blck_size(Q2_0)) != (Q2_0_BYTES, Q2_0_BLOCK):
      raise SystemExit(f"{lib_path}'s TQ2_0/Q2_0 block layouts are not v0.8.0's")

    def with_lib(fn, in_bytes, block):
      def decode(raw):
        count = len(raw) // in_bytes
        out = np.empty(count * block, dtype=np.float32)
        fn(np.frombuffer(raw, dtype=np.uint8).ctypes.data, out.ctypes.data, out.size)
        return out
      return decode
    return with_lib(lib.dequantize_row_tq2_0, TQ2_0_BYTES, TQ2_0_BLOCK), with_lib(lib.dequantize_row_q2_0, Q2_0_BYTES, Q2_0_BLOCK), str(lib_path)

  def tq2_0(raw):
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, TQ2_0_BYTES)
    scale = blocks[:, 64:66].copy().view(np.float16).astype(np.float32)
    quants = blocks[:, :64].reshape(-1, 2, 32)
    codes = ((quants[:, :, None, :] >> np.array([0, 2, 4, 6], dtype=np.uint8)[None, None, :, None]) & 3).reshape(len(blocks), -1)
    return ((codes.astype(np.float32) - 1) * scale).reshape(-1)

  def q2_0(raw):
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, Q2_0_BYTES)
    scale = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    codes = ((blocks[:, None, 2:] >> np.array([0, 2, 4, 6], dtype=np.uint8)[None, :, None]) & 3).reshape(len(blocks), -1)
    return ((codes.astype(np.float32) - 1) * scale).reshape(-1)
  return tq2_0, q2_0, None


def sha256(path):
  with path.open('rb') as f:
    return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('source', type=pathlib.Path)
  parser.add_argument('dest', type=pathlib.Path)
  parser.add_argument('--llama-bin', default=os.environ.get('LLAMA_BIN'), help='the release binaries, for its own decoders')
  parser.add_argument('--force', action='store_true', help='overwrite DEST')
  args = parser.parse_args()
  if args.dest.exists() and not args.force:
    raise SystemExit(f'{args.dest} exists; --force overwrites it')
  info = read_header(args.source)
  type_sizes = {TQ2_0: (TQ2_0_BYTES, TQ2_0_BLOCK), Q2_0: (Q2_0_BYTES, Q2_0_BLOCK)}
  converted = [t for t in info['tensors'] if t['type'] == TQ2_0]
  if not converted:
    raise SystemExit(f'{args.source} has no TQ2_0 tensors; nothing to repack')
  align, data_offset = info['alignment'], info['data_offset']
  with args.source.open('rb') as f:
    header = bytearray(f.read(data_offset))
  # New types and offsets; tensors keep their order in the data section.
  tensors = sorted(info['tensors'], key=lambda t: t['offset'])
  for i, t in enumerate(tensors):
    t['size'] = (tensors[i + 1]['offset'] if i + 1 < len(tensors) else args.source.stat().st_size - data_offset) - t['offset']
  offset = 0
  for t in tensors:
    t['new_type'] = Q2_0 if t['type'] == TQ2_0 else t['type']
    t['new_offset'] = offset
    t['new_size'] = tensor_bytes(t, Q2_0, type_sizes) if t['type'] == TQ2_0 else t['size']
    if t['type'] == TQ2_0 and tensor_bytes(t, TQ2_0, type_sizes) > t['size']:
      raise SystemExit(f"{t['name']}: the file holds fewer bytes than its TQ2_0 blocks need")
    struct.pack_into('<I', header, t['type_position'], t['new_type'])
    struct.pack_into('<Q', header, t['offset_position'], offset)
    offset = (offset + t['new_size'] + align - 1) // align * align
  args.dest.parent.mkdir(parents=True, exist_ok=True)
  with args.source.open('rb') as src, args.dest.open('wb') as dst:
    dst.write(header)
    for t in tensors:
      src.seek(data_offset + t['offset'])
      assert dst.tell() == data_offset + t['new_offset'], t['name']
      remaining = tensor_bytes(t, TQ2_0, type_sizes) if t['type'] == TQ2_0 else t['size']
      while remaining:
        take = min(16384 * TQ2_0_BYTES if t['type'] == TQ2_0 else 4 << 20, remaining)
        data = src.read(take)
        if len(data) != take:
          raise SystemExit(f"{t['name']}: short read")
        dst.write(pack(data) if t['type'] == TQ2_0 else data)
        remaining -= take
      dst.write(b'\0' * ((-dst.tell()) % align))
  # Read back: converted tensors decode identically, the others are the same bytes.
  decode_tq2_0, decode_q2_0, library = decoders(args.llama_bin)
  back = {t['name']: t for t in read_header(args.dest)['tensors']}
  back_offset = read_header(args.dest)['data_offset']
  with args.source.open('rb') as src, args.dest.open('rb') as dst:
    for t in tensors:
      d = back[t['name']]
      if d['shape'] != t['shape'] or d['type'] != t['new_type']:
        raise SystemExit(f"{t['name']}: descriptor mismatch after the repack")
      src.seek(data_offset + t['offset'])
      dst.seek(back_offset + d['offset'])
      if t['type'] == TQ2_0:
        blocks = t['elements'] // TQ2_0_BLOCK
        while blocks:
          count = min(16384, blocks)
          a = decode_tq2_0(src.read(count * TQ2_0_BYTES))
          b = decode_q2_0(dst.read(count * 4 * Q2_0_BYTES))
          if not np.array_equal(a.view(np.uint32), b.view(np.uint32)):
            raise SystemExit(f"{t['name']}: the repacked values differ")
          blocks -= count
      else:
        remaining = t['size']
        while remaining:
          take = min(4 << 20, remaining)
          if src.read(take) != dst.read(take):
            raise SystemExit(f"{t['name']}: copied bytes differ")
          remaining -= take
  print(json.dumps({'source': str(args.source), 'dest': str(args.dest), 'bytes': args.dest.stat().st_size,
                    'sha256': sha256(args.dest), 'converted_tensors': len(converted),
                    'converted_elements': sum(t['elements'] for t in converted), 'unchanged_tensors': len(tensors) - len(converted),
                    'verified_with': library or 'the reference decoding in this file'}))


if __name__ == '__main__':
  sys.exit(main())
