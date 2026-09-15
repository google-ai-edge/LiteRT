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
# ==============================================================================

"""Run an independent FP32 Transformers reference for Gemma 4 mobile CT weights.

The checkpoint stays local. Packed embeddings decode only requested rows, while
active linear weights are dequantized to FP32 in bounded chunks. Static INT8
input/output quantization from config.json is applied around the corresponding
linear modules. This is a float execution reference, not an integer-kernel
benchmark: the unquantized LM-head input stays FP32, and BF16 weight scales are
upcast before multiplication. No compressed-tensors package is required.

Requires PyTorch, safetensors, NumPy, and Transformers with Gemma4ForCausalLM.
Example:
  python reference_mobile.py --model-dir /path/to/mobile-ct \
      --output-prefix /tmp/gemma4-reference --decode-steps 3
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import re
import resource
import sys
import time

from safetensors import safe_open
import torch
from torch import nn
import transformers
from transformers import Gemma4ForCausalLM, Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding


def unpack_rows(packed, bits, columns):
    """Decode CT's low-bit-first, offset-binary INT32 representation."""
    if packed.dtype != torch.int32 or bits not in (2, 4, 8):
        raise ValueError("Expected INT32 packed weights with 2, 4, or 8 bits")
    slots = 32 // bits
    result = torch.empty((packed.shape[0], packed.shape[1] * slots), dtype=torch.int8)
    for slot in range(slots):
        values = ((packed >> (bits * slot)) & ((1 << bits) - 1)) - (1 << (bits - 1))
        result[:, slot::slots] = values.to(torch.int8)
    return result[:, :columns]


class Checkpoint:
    def __init__(self, directory):
        self.directory = directory
        self.config = json.loads((directory / "config.json").read_text())
        quant = self.config.get("quantization_config", {})
        if quant.get("quant_method") != "compressed-tensors":
            raise ValueError("Expected a compressed-tensors checkpoint")
        if quant.get("kv_cache_scheme") is not None:
            raise ValueError("This reference does not implement quantized KV caches")
        self.groups = list(quant.get("config_groups", {}).values())
        self.stack = ExitStack()
        self.files = {}
        for path in sorted(directory.glob("*.safetensors")):
            handle = self.stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            for key in handle.keys():
                if key in self.files:
                    raise ValueError(f"Duplicate tensor: {key}")
                self.files[key] = handle
        if not self.files:
            raise ValueError(f"No safetensors found in {directory}")

    def close(self):
        self.stack.close()

    def tensor(self, name):
        return self.files[name].get_tensor(name)

    def rows(self, name, begin, end):
        return self.files[name].get_slice(name)[begin:end]

    def scheme(self, name):
        matches = []
        for group in self.groups:
            if any(re.fullmatch(target[3:], name) if target.startswith("re:")
                   else target == name for target in group.get("targets", [])):
                matches.append(group)
        if len(matches) > 1:
            raise ValueError(f"Multiple quantization schemes for {name}")
        return matches[0] if matches else None

    def weight_shape(self, name):
        if name + ".weight_packed" in self.files:
            return tuple(self.tensor(name + ".weight_shape").tolist())
        return tuple(self.files[name + ".weight"].get_slice(name + ".weight").get_shape())

    def weight_rows(self, name, begin, end):
        scheme = self.scheme(name)
        if scheme is None:
            return self.rows(name + ".weight", begin, end).float()
        args = scheme["weights"]
        if not args["symmetric"] or args.get("actorder") is not None:
            raise ValueError(f"Unsupported asymmetric or reordered weights: {name}")
        columns = self.weight_shape(name)[1]
        if name + ".weight_packed" in self.files:
            values = unpack_rows(self.rows(name + ".weight_packed", begin, end),
                                 args["num_bits"], columns)
        else:
            values = self.rows(name + ".weight", begin, end)
            if values.dtype != torch.int8 or args["num_bits"] != 8:
                raise ValueError(f"Unsupported integer weight representation: {name}")
        scales = self.rows(name + ".weight_scale", begin, end).float()
        if args["strategy"] == "channel":
            if scales.shape != (end - begin, 1):
                raise ValueError(f"Invalid channel scale shape: {name}")
            return values.float() * scales
        if args["strategy"] == "group":
            size = args["group_size"]
            if columns % size or scales.shape != (end - begin, columns // size):
                raise ValueError(f"Invalid group scale shape: {name}")
            return (values.float().reshape(end - begin, -1, size) *
                    scales.unsqueeze(-1)).reshape(end - begin, columns)
        raise ValueError(f"Unsupported weight strategy: {name}")

    def weight(self, name):
        shape = self.weight_shape(name)
        if len(shape) != 2:
            return self.tensor(name + ".weight").float()
        result = torch.empty(shape, dtype=torch.float32)
        for begin in range(0, shape[0], 1024):
            end = min(begin + 1024, shape[0])
            result[begin:end].copy_(self.weight_rows(name, begin, end))
        return result


class LazyEmbedding(nn.Module):
    def __init__(self, checkpoint, name, scale):
        super().__init__()
        self.checkpoint = checkpoint
        self.name = name
        self.scale = scale
        self.num_embeddings, self.embedding_dim = checkpoint.weight_shape(name)

    def forward(self, input_ids):
        flat_ids = input_ids.reshape(-1).tolist()
        rows = {}
        for token in set(flat_ids):
            if not 0 <= token < self.num_embeddings:
                raise ValueError(f"Token ID out of range: {token}")
            rows[token] = self.checkpoint.weight_rows(self.name, token, token + 1)[0]
        result = torch.stack([rows[token] for token in flat_ids])
        return result.reshape(*input_ids.shape, self.embedding_dim) * self.scale


def source_name(name):
    return "model.language_model." + name[6:] if name.startswith("model.") else name


def fake_quantize(value, scale):
    return torch.round(value / scale).clamp(-128, 127) * scale


def activation_scale(checkpoint, name, scheme, field):
    args = scheme.get(field + "_activations")
    if args is None:
        return None
    if (args["num_bits"] != 8 or not args["symmetric"] or args["dynamic"]
            or args["strategy"] != "tensor" or args["type"] != "int"):
        raise ValueError(f"Unsupported {field} activation scheme for {name}")
    scale = checkpoint.tensor(name + "." + field + "_scale").float()
    if scale.numel() != 1 or not torch.isfinite(scale).all() or scale.item() <= 0:
        raise ValueError(f"Invalid {field} scale for {name}")
    return scale


def build_model(checkpoint):
    config = Gemma4TextConfig(**checkpoint.config["text_config"])
    config._attn_implementation = "eager"
    config.dtype = torch.float32
    with torch.device("meta"):
        model = Gemma4ForCausalLM(config)
    model.model.embed_tokens = LazyEmbedding(
        checkpoint, "model.language_model.embed_tokens", config.hidden_size ** 0.5)
    model.model.embed_tokens_per_layer = LazyEmbedding(
        checkpoint, "model.language_model.embed_tokens_per_layer",
        config.hidden_size_per_layer_input ** 0.5)
    model.model.rotary_emb = Gemma4TextRotaryEmbedding(config, device="cpu")

    loaded_bytes = 0
    for index, (name, parameter) in enumerate(list(model.named_parameters())):
        source = source_name(name)
        value = (checkpoint.weight(source[:-7]) if source.endswith(".weight")
                 else checkpoint.tensor(source).float())
        if tuple(value.shape) != tuple(parameter.shape):
            raise ValueError(f"Shape mismatch for {name}: {value.shape} vs {parameter.shape}")
        owner_name, attribute = name.rsplit(".", 1)
        setattr(model.get_submodule(owner_name), attribute,
                nn.Parameter(value, requires_grad=False))
        loaded_bytes += value.numel() * value.element_size()
        if index % 50 == 0:
            print(f"Loaded {index + 1} parameters; {loaded_bytes / 2**30:.2f} GiB", flush=True)
    for name, value in list(model.named_buffers()):
        if not value.is_meta:
            continue
        source = source_name(name)
        if source not in checkpoint.files:
            raise ValueError(f"Uninitialized buffer: {name}")
        owner_name, attribute = name.rsplit(".", 1)
        setattr(model.get_submodule(owner_name), attribute, checkpoint.tensor(source).float())

    quantized_modules = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        source = source_name(name)
        scheme = checkpoint.scheme(source)
        if scheme is None:
            continue
        input_scale = activation_scale(checkpoint, source, scheme, "input")
        output_scale = activation_scale(checkpoint, source, scheme, "output")
        if input_scale is not None:
            module.register_forward_pre_hook(
                lambda _, inputs, scale=input_scale: (fake_quantize(inputs[0], scale),))
        if output_scale is not None:
            module.register_forward_hook(
                lambda _, inputs, output, scale=output_scale: fake_quantize(output, scale))
        if input_scale is not None or output_scale is not None:
            quantized_modules.append(source)
    model.eval()
    return model, loaded_bytes, quantized_modules


def parse_ids(text):
    try:
        ids = [int(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integer token IDs") from error
    if not ids or any(value < 0 for value in ids):
        raise argparse.ArgumentTypeError("Token IDs must be nonnegative and nonempty")
    return ids


def register_intermediate_dumps(model, output_prefix, state, records):
    """Observe decoder and final norm outputs without replacing their values."""
    def make_hook(name):
        def dump(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Expected tensor output from {name}")
            suffix = state["step"] + "." + name + ".f32"
            path = Path(str(output_prefix) + suffix)
            if path.exists():
                raise ValueError(f"Intermediate output already exists: {path}")
            values = output.detach().float().cpu()
            if not torch.isfinite(values).all():
                raise ValueError(f"Nonfinite intermediate output: {suffix}")
            values.numpy().astype("<f4", copy=False).tofile(path)
            records.append({"suffix": suffix, "shape": list(values.shape),
                            "dtype": "float32-little-endian"})
        return dump

    handles = [layer.register_forward_hook(make_hook(f"layer_{index:03d}"))
               for index, layer in enumerate(model.model.layers)]
    handles.append(model.model.norm.register_forward_hook(make_hook("norm")))
    return handles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--input-ids", type=parse_ids, default=[2, 818, 5279, 529, 7001, 563])
    parser.add_argument("--decode-steps", type=int, default=3,
                        help="Number of cached decode forwards after prefill")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dump-intermediates", action="store_true",
                        help="Dump each decoder layer and final norm as row-major FP32")
    args = parser.parse_args()
    if args.threads < 1 or args.decode_steps < 0:
        parser.error("threads must be positive and decode-steps must be nonnegative")
    suffixes = [".prefill.f32"] + [
        f".decode_{step:04d}.f32" for step in range(1, args.decode_steps + 1)]
    paths = [Path(str(args.output_prefix) + suffix) for suffix in [".json"] + suffixes]
    if any(path.exists() for path in paths):
        parser.error("Output prefix already exists; choose a fresh prefix")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    checkpoint = Checkpoint(args.model_dir.resolve())
    start = time.monotonic()
    try:
        model, weight_bytes, quantized_modules = build_model(checkpoint)
        loading_seconds = time.monotonic() - start
        generated = []
        steps = []
        cache = None
        intermediate_records = []
        dump_state = {}
        if args.dump_intermediates:
            register_intermediate_dumps(model, args.output_prefix, dump_state,
                                        intermediate_records)
        with torch.inference_mode():
            for index, suffix in enumerate(suffixes):
                token_ids = args.input_ids if index == 0 else [generated[-1]]
                step_start = time.monotonic()
                dump_state["step"] = suffix.removesuffix(".f32")
                output = model(input_ids=torch.tensor([token_ids], dtype=torch.long),
                               past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                logits = output.logits[0, -1].float().cpu()
                if not torch.isfinite(logits).all():
                    raise ValueError(f"Nonfinite logits at {suffix}")
                logits.numpy().astype("<f4", copy=False).tofile(str(args.output_prefix) + suffix)
                token = int(logits.argmax())
                generated.append(token)
                top = torch.topk(logits, 10)
                steps.append({"suffix": suffix, "input_token_ids": token_ids,
                              "argmax_token_id": token, "seconds": time.monotonic() - step_start,
                              "top_token_ids": top.indices.tolist(),
                              "top_logits": top.values.tolist()})
                print(f"{suffix}: next token {token}; "
                      f"{steps[-1]['seconds']:.2f} seconds", flush=True)
        report = {
            "model_dir": str(args.model_dir.resolve()), "input_token_ids": args.input_ids,
            "generated_token_ids": generated, "logits_suffixes": suffixes,
            "vocab_size": model.config.vocab_size, "logits_dtype": "float32-little-endian",
            "reference": ("Transformers Gemma4ForCausalLM; FP32 dequantized CT weights; "
                          "static INT8 input/output QDQ"),
            "lm_head_input": ("FP32 (no dynamic quantization; original CT scheme "
                              "has no activation quantization)"),
            "embedding_loading": "Packed checkpoint rows decoded on demand",
            "kv_cache_quantization": None, "static_quantized_linear_count": len(quantized_modules),
            "loaded_fp32_parameter_bytes": weight_bytes, "loading_seconds": loading_seconds,
            "elapsed_seconds": time.monotonic() - start,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "threads": args.threads, "python": sys.version.split()[0],
            "torch": torch.__version__, "transformers": transformers.__version__, "steps": steps,
        }
        if args.dump_intermediates:
            report["intermediates"] = intermediate_records
        Path(str(args.output_prefix) + ".json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {args.output_prefix}.json", flush=True)
    finally:
        checkpoint.close()


if __name__ == "__main__":
    main()
