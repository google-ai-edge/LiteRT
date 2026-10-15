# Agent Instructions for `//tensor/`

<!-- disableFinding(LINE_OVER_80) -->

This directory contains the C++ **LiteRT Tensor API**: a lightweight, backend-agnostic graph construction and execution framework for on-device generative AI and ML models across XNNPACK (CPU), ML Drift (OpenCL/Metal/WebGPU GPU), TFLite FlatBuffer export, and multi-backend `ModelChain` runners.

## 1. Repository Layout

| Path | Purpose |
|---|---|
| `tensor.h`, `arithmetic.h`, `datatypes.h`, `buffer.h` | Core Tensor API graph primitives (`TensorHandle`, `Shape`, `Type`, `Quantization`, arithmetic ops). |
| `ops/` | Core operator definitions (`FullyConnected`, `BatchMatMul`, `DepthwiseConv2D`, `Concatenation`, `Slice`, `Reshape`, `Transpose`, `Softmax`, `Select`, etc.). |
| `examples/ops/transformer/` | Tier-1 reusable transformer building blocks (`RmsNorm`, `RoPE`, `ScaledDotProductAttention`, `GatedMlp`, etc.) templated on backend mixins. |
| `examples/utils/` | Checkpoint and config helpers (`SafetensorLoader`, `LazyTensorMapping`, `TensorLoadingHooks`, `minijson`). |
| `examples/gguf/` | GGUF parser and loader (`GgufParser`, `GgufLoader`) for multi-format quantized GGUF checkpoints (`Q4_0`, `Q4_K_M`, K-quants, I-quants, `Q8_0`). |
| `backends/xnnpack/`, `runners/xnnpack/` | Direct XNNPACK CPU backend and model runner (`XnnpackModelRunner`). |
| `backends/ml_drift/`, `runners/ml_drift/` | Direct ML Drift GPU backend and compiled OpenCL model runner (`CompiledMlDriftModelRunner`). |
| `backends/tflite/`, `backends/multi_backend/` | `.tflite` FlatBuffer serializer (`ModelFactory`) and heterogeneous `CompositeModel` / `ModelChain` partitioner. |
| `examples/gemma3/`, `examples/gemma4/` | Reference Gemma 3 and Gemma 4 C++ graphs and runtimes. |
| `models/<family>/` | End-to-end model bring-up packages (e.g., `models/qwen3_5/`): config, weight map, templated C++ graph, forward pass, runtime, CPU/GPU backends, HF reference goldens, and V0–V3 + benchmark tests. |
| `_agents/skills/` | Modular agent skills for end-to-end model bring-up, verification, quantization, and CPU/GPU performance tuning. |

## 2. Build & Test Conventions

- **Always use `blaze-for-agents`**:
  ```bash
  /google/bin/releases/arca9-local-blaze-cli/blaze-for-agents test -c opt //tensor/...
  ```
- **GPU tests on Forge (`ML Drift` over OpenCL)**:
  Tag `cc_test` targets that require an OpenCL GPU with `tags = ["requires-gpu-sm90"]`.
- **Large checkpoint tests**:
  Do **not** check multi-GB `.safetensors` or `.gguf` weights into Piper. Keep unit tests in TAP fast and hermetic using deterministic LCG synthetic models (`test_model.h`, `CreateSynthetic`, `<model>_perf_smoke_test`) and checked-in small golden files (`testdata/*.txt`, `testdata/*.json`). Mark full-checkpoint tests `tags = ["manual", "local"]` (CPU) or `tags = ["manual", "notap", "requires-gpu-sm90"]` (GPU via local staging `genrule`s).

## 3. End-to-End Model Bring-Up Lifecycle (`V0 -> V1 -> V2 -> V3 -> Perf`)

When bringing up a new Hugging Face or GGUF model family under `models/<family>/`, follow the 5-stage gated workflow and load the corresponding skill from `_agents/skills/` before writing code:

| Stage | Gate & Objective | Skill to Read First |
|---|---|---|
| **Stage 1: Intake & Architecture Card** | Parse `config.json` & checkpoint tensor headers, document the architecture card and weight table in `models/<family>/README.md`, audit Tier-1 ops vs backend capabilities, and select target precision classes. | [`_agents/skills/hf_intake_and_arch_card/SKILL.md`](_agents/skills/hf_intake_and_arch_card/SKILL.md) |
| **Stage 2: C++ Graph & Runtime Authoring** | Author `<model>_config`, `<model>_weights` (`LazyTensorMapping` hooks), `<model>_graph.h` (`template <typename BackendMixin>`), fixed-shape prefill/decode `<model>_forward_pass`, and `<model>_runtime`. | [`_agents/skills/cpp_graph_authoring/SKILL.md`](_agents/skills/cpp_graph_authoring/SKILL.md) |
| **Stage 3: V0–V2 Golden Verification** | Verify `V0` per-op/stage parity (`<model>_op_parity_ml_drift_test`), `V1` synthetic LCG + real-weight FP32 parity against `transformers` (`max |Δ| <= 1e-4`), and `V2` decode-step state handoff (`prefill(T) + decode(1) == prefill(T+1)`) on XNNPACK and ML Drift. | [`_agents/skills/hf_golden_verification/SKILL.md`](_agents/skills/hf_golden_verification/SKILL.md) |
| **Stage 4: V3 Quantization & GGUF** | Enable `compressed-tensors` `W8A8`, load-time `int8_channel` (`--weights=int8`) and `q4_0` (`--weights=q4_0`), per-row INT8 tied `lm_head` (`--int8_head`), and multi-format GGUF (`GgufLoader`), verified against dequantized-weight HF goldens. | [`_agents/skills/quantize_weights_and_gguf/SKILL.md`](_agents/skills/quantize_weights_and_gguf/SKILL.md) |
| **Stage 5: CPU & GPU Runner Tuning & Benchmarking** | Optimize peak RSS (`--low_memory`), XNNPACK `qd8`/`qb4w` threading, ML Drift zero-copy GPU state binding (`ExternalTensorsInfo::immutable_tensors`), async I/O, and `kF32F16` precision; benchmark `pp64`/`tg32` against `llama.cpp` and add `<model>_perf_smoke_test`. | [`_agents/skills/tune_cpu_and_gpu_runners/SKILL.md`](_agents/skills/tune_cpu_and_gpu_runners/SKILL.md) |
