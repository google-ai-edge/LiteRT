# Agent Instructions for `//tensor/`

<!-- disableFinding(LINE_OVER_80) -->

This directory contains the C++ **LiteRT Tensor API**: a lightweight, backend-agnostic graph construction and execution framework for on-device generative AI and ML models across XNNPACK (CPU), YNNPACK (CPU), ML Drift (OpenCL/Metal/WebGPU GPU), TFLite FlatBuffer export, and multi-backend `ModelChain` runners.

## 1. Repository Layout

| Path | Purpose |
|---|---|
| `tensor.h`, `arithmetic.h`, `datatypes.h`, `buffer.h` | Core Tensor API graph primitives (`TensorHandle`, `Shape`, `Type`, `Quantization`, lazy/owning `Buffer`, arithmetic ops). |
| `ops/` | Core operator definitions (`FullyConnected`, `BatchMatMul`, `DepthwiseConv2D`, `Concatenation`, `Slice`, `Reshape`, `Transpose`, `Softmax`, `Select`, etc.). |
| `examples/ops/transformer/` | Tier-1 reusable transformer building blocks (`RmsNorm`, `RoPE`, `ScaledDotProductAttention`, `GatedMlp`, etc.) templated on backend mixins. |
| `examples/utils/`, `utils/` | Checkpoint/config helpers (`SafetensorLoader`, `LazyTensorMapping`, `TensorLoadingHooks`, `minijson`) and shared parallel row-loop utilities (`utils/parallel.h`). |
| `examples/gguf/` | GGUF parser and loader (`GgufParser`, `GgufLoader`) with lazy mmap-backed matrix materialization (`SetLazyQuantizedMatrices`) and lossless multi-format unpacking (`Q4_0`, `Q4_1`, `Q8_0`, `Q2_K`–`Q6_K`, `Q5_0`/`Q5_1`, `IQ4_NL`/`IQ4_XS`, `SetNativeSubblockQuantization`). |
| `backends/xnnpack/`, `runners/xnnpack/` | Direct XNNPACK CPU backend and model runner (`XnnpackModelRunner`, `qd8-f32-qc8w`/`qb4w` GEMMs, `.xnwc` packed weight cache). |
| `backends/ynnpack/`, `runners/ynnpack/` | Direct YNNPACK CPU backend and model runner (`YnnpackRunner`, native `Type::kI2`/`kI4`/`kI8` affine & sub-block `block_size = 16, 32` quantization, row-only gather, composite transformer lowerings, worker-thread keep-alive). |
| `backends/ml_drift/`, `runners/ml_drift/` | Direct ML Drift GPU backend and compiled OpenCL & Apple Metal model runners (`CompiledMlDriftModelRunner`, `MlDriftClBackend`, `MlDriftMetalBackend`, `CompiledMlDriftMetalModelRunner`, native `Q4_0`/`Q4_1`/`Q4_K`/`Q5_K`/`Q6_K`/`Q8_0` GPU weight lowering and persistent weight cache). |
| `backends/tflite/`, `backends/multi_backend/` | `.tflite` FlatBuffer serializer (`ModelFactory`) and heterogeneous `CompositeModel` / `ModelChain` partitioner. |
| `examples/gemma3/`, `examples/gemma4/` | Standalone reference Gemma 3 and Gemma 4 C++ graphs and runtimes. |
| `models/common/` | Model-agnostic bring-up scaffold: checkpoint weight hooks & lazy load-time quantization (`weight_hooks`), declarative `WeightMap` (`SyntheticLazyBuffer`), `HfConfigReader`, standalone `.gguf` config/tokenizer/chat metadata synthesis (`gguf_metadata`), `DecoderLayout` / `BuildDecoderForwardGraph` / `RuntimeBackend` (supporting dual local/global RoPE, sliding-window KV ring buffers, KV-shared layers, and per-layer inputs), reusable composite blocks (`transformer_blocks.h`: `GqaAttention`, `GatedMlp`, `SparseMoeGatedMlp`, `PerLayerInputPrep`), `TextDecoderRuntime<ModelTraits>`, `XnnpackRuntimeBackend`, `YnnpackRuntimeBackend`, `MlDriftRuntimeBackend` (OpenCL & Metal), `TensorApiDecoder` / `DecoderRegistry` (LiteRT-LM `tensor_api_cpu`/`tensor_api_gpu` integration), and shared `TYPED_TEST_P` suites in `common/testing/`. |
| `models/<family>/` | End-to-end model packages built on `models/common/`: `models/gemma4/` (`E2B`, `E4B`, `26B-A4B` MoE, `31B`), `models/qwen3/` (dense `0.6B`, `1.7B`, `4B`, `8B`), `models/qwen3_5/` (`0.8B`–`27B` hybrid GatedDeltaNet + GQA and dense presets), and `models/lfm2/` (`LFM2` & `LFM2-MoE`). |
| `_agents/skills/` | Modular agent skills for end-to-end model bring-up, verification, GGUF/weight quantization, and CPU/GPU performance tuning. |

## 2. Build & Test Conventions

- **Always use `blaze-for-agents`**:
  ```bash
  /google/bin/releases/arca9-local-blaze-cli/blaze-for-agents test -c opt //tensor/...
  ```
- **GPU tests on Forge (`ML Drift` over OpenCL)**:
  Tag `cc_test` targets that require an OpenCL GPU with `tags = ["requires-gpu-sm90"]`.
- **Large checkpoint tests**:
  Do **not** check multi-GB `.safetensors` or `.gguf` weights into Piper. Keep unit tests in TAP fast and hermetic using deterministic LCG synthetic models (`WeightMap::SyntheticWeights`, `CreateSynthetic`, `<model>_perf_smoke_test`) and checked-in small golden files (`testdata/*.txt`, `testdata/*.json`). Mark full-checkpoint tests `tags = ["manual", "local"]` (CPU) or `tags = ["manual", "notap", "requires-gpu-sm90"]` (GPU via local staging `genrule`s).

## 3. End-to-End Model Bring-Up Lifecycle (`V0 -> V1 -> V2 -> V3 -> Perf`)

When bringing up a new Hugging Face or GGUF model family under `models/<family>/`, follow the 5-stage gated workflow and load the corresponding skill from `_agents/skills/` before writing code:

| Stage | Gate & Objective | Skill to Read First |
|---|---|---|
| **Stage 1: Intake & Architecture Card** | Parse `config.json` & `.safetensors`/`.gguf` headers (plus standalone `.gguf` synthesis in `gguf_metadata.cc`), document the architecture card and HF/GGUF weight table in `models/<family>/README.md`, audit ops across XNNPACK, YNNPACK, and ML Drift, and author `<model>_config`. | [`_agents/skills/hf_intake_and_arch_card/SKILL.md`](_agents/skills/hf_intake_and_arch_card/SKILL.md) |
| **Stage 2: C++ Graph, Runtime & Decoder Authoring** | Author `<model>_weights` (`DeclareWeights` + HF/GGUF name predicates & hooks), `<model>_graph.h` (`template <class... Mixins>` over `common::GqaAttention`/`GatedMlp`/`SparseMoeGatedMlp`), `<model>_forward_pass` (`DecoderLayoutOf` + `ModelTraits`), `<model>_runtime` (`RuntimeTraits` over `TextDecoderRuntime`), and `<model>_decoder` (`LITERT_TENSOR_REGISTER_DECODER` for LiteRT-LM). | [`_agents/skills/cpp_graph_authoring/SKILL.md`](_agents/skills/cpp_graph_authoring/SKILL.md) |
| **Stage 3: V0–V3 Golden Verification** | Verify `V0` per-op/stage parity (`<model>_op_parity_ml_drift_test`), `V1` synthetic LCG + real-weight FP32 parity against `transformers`, `V2` stateful decode handoff (`prefill(T) + decode(1) == prefill(T+1)`, including ring-buffer local KV caches), and `V3` quantized/GGUF parity across XNNPACK, YNNPACK, and ML Drift. | [`_agents/skills/hf_golden_verification/SKILL.md`](_agents/skills/hf_golden_verification/SKILL.md) |
| **Stage 4: Weight Quantization & Lossless GGUF** | Enable `compressed-tensors` `W8A8`, lazy load-time `int8_channel` / `q4_0` RTN quantization, quantized tied `lm_head`, standalone `.gguf` metadata synthesis, and lossless GGUF unpacking (`Q4_0`, `Q4_1`, `Q8_0`, `Q2_K`–`Q6_K`, `Q5_0`/`Q5_1`, `IQ4_NL`/`IQ4_XS`) across XNNPACK, YNNPACK, and ML Drift. | [`_agents/skills/quantize_weights_and_gguf/SKILL.md`](_agents/skills/quantize_weights_and_gguf/SKILL.md) |
| **Stage 5: CPU & GPU Runner Tuning & Benchmarking** | Optimize compile latency & peak RSS (lazy `Q4_0`/GGUF buffers, `ReleaseMaterializedPayloadOnly`, CPU/GPU weight caches), XNNPACK & YNNPACK CPU kernels, and ML Drift OpenCL/Metal GPU kernels (zero-copy state, residual epilogue fusion, `head_dim=128/256/512` SDPA, 4-warp MoE); instantiate `<model>_perf_smoke_test` and benchmark against `llama.cpp` and LiteRT-LM. | [`_agents/skills/tune_cpu_and_gpu_runners/SKILL.md`](_agents/skills/tune_cpu_and_gpu_runners/SKILL.md) |
