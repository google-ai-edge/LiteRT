/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"  // IWYU pragma: keep
#include "tensor/backends/ml_drift/ml_drift_buffer.h"
#include "tensor/backends/ml_drift/ml_drift_helpers.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma3/benchmark_utils.h"
#include "tensor/examples/gemma3/gemma3_graph.h"
#include "tensor/examples/gemma3/model_config.h"
#include "tensor/examples/gemma3/tflite_loader.h"
#include "tensor/examples/gemma3/tokenizer.h"
#include "tensor/internal/graph.h"
#include "tensor/runners/ml_drift/ml_drift_cl_model_runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

ABSL_FLAG(std::string, weights_path, "",
          "Path to the TFLite weights file (.tflite)");
ABSL_FLAG(std::string, tokenizer_path, "",
          "Path to the tokenizer.model file (SentencePiece format)");
ABSL_FLAG(std::string, prompt, "Hello, world!",
          "Input prompt for text generation");
ABSL_FLAG(int, max_tokens, 16, "Maximum number of tokens to generate");
ABSL_FLAG(std::string, model_variant, "1b", "Gemma model variant: 270m, 1b");
ABSL_FLAG(std::string, weight_mode, "fp32",
          "Weight loading mode: fp32 or quantized");
ABSL_FLAG(bool, verbose, false, "Enable verbose logging");
ABSL_FLAG(bool, profile, false, "Enable profiling");
ABSL_FLAG(bool, benchmark, false,
          "Run benchmark mode (fixed prefill/decode sizes, no token output)");
ABSL_FLAG(int, benchmark_prefill_tokens, 512, "Benchmark prefill token count");
ABSL_FLAG(int, benchmark_decode_tokens, 128, "Benchmark decode token count");
ABSL_FLAG(int, benchmark_seed, 123, "Benchmark RNG seed");
ABSL_FLAG(bool, bypass_embedding, false, "Bypass embedding lookup on GPU");
ABSL_FLAG(
    int, decode_batch_size, 16,
    "Number of tokens to enqueue before waiting (only in benchmark mode)");
ABSL_FLAG(bool, use_async_param_update, true,
          "Use GPU-GPU token copy and async param upload");
ABSL_FLAG(bool, use_enqueue_wait, true,
          "Use async enqueue and wait instead of blocking Run()");
ABSL_FLAG(bool, bypass_lm_head, false, "Bypass lm_head on GPU and run on CPU");
ABSL_FLAG(bool, use_gpu_param_update, false,
          "Use GPU shader to perform parameter update");
ABSL_FLAG(int, readback_period, 1,
          "Read back output token every N steps. 0 means no readback.");
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift

namespace litert::tensor::examples {

namespace {

class OpenClParamUpdater {
 public:
  OpenClParamUpdater() = default;
  ~OpenClParamUpdater() {
    if (kernel_) {
      ml_drift::cl::clReleaseKernel(kernel_);
    }
    if (program_) {
      ml_drift::cl::clReleaseProgram(program_);
    }
  }

  absl::Status Init(ml_drift::cl::Environment* env) {
    cl_context context = env->context().context();
    cl_device_id device = env->device().id();

    const char* source = R"kernel(
      __kernel void UpdateCacheParams(__global int* cache_params, int cache_len, int active_tokens, int top_k) {
          if (get_global_id(0) == 0) {
              cache_params[0] = cache_len;
              cache_params[1] = active_tokens;
              cache_params[2] = top_k;
          }
      }
    )kernel";

    cl_int err;
    program_ = ml_drift::cl::clCreateProgramWithSource(context, 1, &source,
                                                       nullptr, &err);
    if (err != CL_SUCCESS) {
      return absl::InternalError("Failed clCreateProgramWithSource");
    }

    err = ml_drift::cl::clBuildProgram(program_, 1, &device, nullptr, nullptr,
                                       nullptr);
    if (err != CL_SUCCESS) {
      size_t log_size;
      ml_drift::cl::clGetProgramBuildInfo(
          program_, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
      std::vector<char> log(log_size);
      ml_drift::cl::clGetProgramBuildInfo(program_, device,
                                          CL_PROGRAM_BUILD_LOG, log_size,
                                          log.data(), nullptr);
      return absl::InternalError(
          absl::StrCat("Failed clBuildProgram: ", log.data()));
    }

    kernel_ = ml_drift::cl::clCreateKernel(program_, "UpdateCacheParams", &err);
    if (err != CL_SUCCESS) {
      return absl::InternalError("Failed clCreateKernel");
    }

    return absl::OkStatus();
  }

  absl::Status Update(ml_drift::cl::Environment* env, cl_mem buffer,
                      int cache_len, int active_tokens, int top_k) {
    cl_int err;
    err = ml_drift::cl::clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
    if (err != CL_SUCCESS)
      return absl::InternalError(
          absl::StrCat("Failed clSetKernelArg 0: error=", err));
    err = ml_drift::cl::clSetKernelArg(kernel_, 1, sizeof(int), &cache_len);
    if (err != CL_SUCCESS)
      return absl::InternalError("Failed clSetKernelArg 1");
    err = ml_drift::cl::clSetKernelArg(kernel_, 2, sizeof(int), &active_tokens);
    if (err != CL_SUCCESS)
      return absl::InternalError("Failed clSetKernelArg 2");
    err = ml_drift::cl::clSetKernelArg(kernel_, 3, sizeof(int), &top_k);
    if (err != CL_SUCCESS)
      return absl::InternalError("Failed clSetKernelArg 3");

    size_t global_work_size = 1;
    err = ml_drift::cl::clEnqueueNDRangeKernel(env->queue()->queue(), kernel_,
                                               1, nullptr, &global_work_size,
                                               nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      return absl::InternalError("Failed clEnqueueNDRangeKernel");
    }
    return absl::OkStatus();
  }

 private:
  cl_program program_ = nullptr;
  cl_kernel kernel_ = nullptr;
};

using TensorMLD = Tensor<MlDriftMixinTag>;

absl::StatusOr<TfliteLoader::QuantizedLoadMode> ParseWeightMode(
    absl::string_view mode) {
  if (mode == "fp32" || mode == "float") {
    return TfliteLoader::QuantizedLoadMode::kDequantizeToFp32;
  }
  if (mode == "quantized" || mode == "preserve") {
    return TfliteLoader::QuantizedLoadMode::kPreserveQuantized;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported --weight_mode: ", mode,
                   ". Expected one of: fp32, quantized"));
}

struct WeightView {
  std::string name;
  Type type = Type::kUnknown;
  std::vector<int32_t> shape;
  const float* fp32_data = nullptr;
  const int8_t* int8_data = nullptr;
  size_t elements = 0;
  std::shared_ptr<Quantization> quantization;
  std::shared_ptr<Buffer> buffer;
};

std::pair<std::vector<float>, std::vector<float>> ComputeRopeCosSinForPosition(
    int position, int head_dim, float rope_base) {
  std::vector<float> cos_values(static_cast<size_t>(head_dim));
  std::vector<float> sin_values(static_cast<size_t>(head_dim));
  const int half_dim = head_dim / 2;
  for (int i = 0; i < half_dim; ++i) {
    const float freq = 1.0f / std::pow(rope_base, 2.0f * i / head_dim);
    const float angle = static_cast<float>(position) * freq;
    const float cos_val = std::cos(angle);
    const float sin_val = std::sin(angle);
    cos_values[static_cast<size_t>(i)] = cos_val;
    cos_values[static_cast<size_t>(half_dim + i)] = cos_val;
    sin_values[static_cast<size_t>(i)] = sin_val;
    sin_values[static_cast<size_t>(half_dim + i)] = sin_val;
  }
  return {std::move(cos_values), std::move(sin_values)};
}

std::vector<float> PadKvCache(const std::vector<float>& cache, int src_heads,
                              int dst_heads, int cache_len, int max_cache_len,
                              int head_dim) {
  std::vector<float> out(
      static_cast<size_t>(dst_heads) * max_cache_len * head_dim, 0.0f);
  if (cache_len <= 0) {
    return out;
  }
  const int src_stride = cache_len * head_dim;
  const int dst_stride = max_cache_len * head_dim;
  for (int g = 0; g < dst_heads; ++g) {
    const int src_head =
        src_heads > 0 ? std::min(g % std::max(src_heads, 1), src_heads - 1) : 0;
    const float* src = cache.data() + static_cast<size_t>(g) * src_stride;
    if (src_heads > 0) {
      src = cache.data() + static_cast<size_t>(src_head) * src_stride;
    }
    float* dst = out.data() + static_cast<size_t>(g) * dst_stride;
    std::copy(src, src + src_stride, dst);
  }
  return out;
}

absl::StatusOr<std::vector<WeightView>> BuildWeightViews(
    const absl::flat_hash_map<std::string, TensorHandle>& weights) {
  std::vector<WeightView> views;
  views.reserve(weights.size());
  for (const auto& [name, tensor] : weights) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto info, graph::GetInfo(tensor.GetRaw()));
    if (!info.buffer) {
      continue;
    }
    if (info.type != Type::kFP32 && info.type != Type::kI8 &&
        info.type != Type::kI4) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported weight type for OpenCL: ", name, " ",
                       ToString(info.type)));
    }
    WeightView view;
    view.name = name;
    view.type = info.type;
    view.shape = info.shape;
    view.quantization = info.quantization;
    view.buffer = info.buffer;
    if (info.type == Type::kFP32) {
      auto locked = info.buffer->Lock().As<const float>();
      view.fp32_data = locked.data();
      view.elements = locked.size();
    } else {
      // For I8 and I4, we treat them similarly in WeightView for metadata,
      // but we need to be careful about elements count.
      // For I4, elements count in buffer is half of logical elements.
      // However, we just need to keep the pointer to the data.
      auto locked = info.buffer->Lock().As<const int8_t>();
      view.int8_data = locked.data();
      view.elements = locked.size();
      if (view.quantization == nullptr) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Quantized weight missing quantization metadata: ", name));
      }
    }
    views.push_back(std::move(view));
  }
  return views;
}

absl::flat_hash_map<std::string, absl::Span<const float>> BuildFloatWeightMap(
    const std::vector<WeightView>& views) {
  absl::flat_hash_map<std::string, absl::Span<const float>> weight_map;
  weight_map.reserve(views.size() / 2 + 1);
  for (const auto& view : views) {
    if (view.type != Type::kFP32) {
      continue;
    }
    weight_map[view.name] = absl::MakeConstSpan(view.fp32_data, view.elements);
  }
  return weight_map;
}

absl::flat_hash_map<std::string, absl::Span<const int8_t>> BuildInt8WeightMap(
    const std::vector<WeightView>& views) {
  absl::flat_hash_map<std::string, absl::Span<const int8_t>> weight_map;
  weight_map.reserve(views.size() / 2 + 1);
  for (const auto& view : views) {
    if (view.type == Type::kFP32) {
      continue;
    }
    // Both I8 and I4 weights are stored as int8_t data.
    weight_map[view.name] = absl::MakeConstSpan(view.int8_data, view.elements);
  }
  return weight_map;
}

absl::flat_hash_map<std::string, TensorMLD> BuildMlDriftWeightTensors(
    const std::vector<WeightView>& views) {
  absl::flat_hash_map<std::string, TensorMLD> tensors;
  tensors.reserve(views.size());
  for (const auto& view : views) {
    TensorInit init;
    init.name = view.name;
    init.type = view.type;
    init.shape = view.shape;
    init.quantization = view.quantization;
    init.buffer = view.buffer;
    tensors[view.name] = TensorMLD(init);
  }
  return tensors;
}

struct PrefillResult {
  std::vector<int32_t> output_ids;
  std::vector<std::vector<float>> key_caches;
  std::vector<std::vector<float>> value_caches;
  std::vector<float> embedded_input;
  absl::Duration run_time;
};

absl::Status CpuEmbeddingLookup(const std::vector<int32_t>& input_ids,
                                const WeightView& emb_weights,
                                float* output_data) {
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto quant,
      emb_weights.quantization->As<const PerChannelAffineQuantization>());
  const float* scales = quant.scales.data();
  const auto* zero_points = quant.zero_points.data();

  int vocab_size = emb_weights.shape[0];
  int emb_dim = emb_weights.shape[1];

  const int8_t* packed_data = emb_weights.int8_data;

  for (size_t t = 0; t < input_ids.size(); ++t) {
    int32_t id = input_ids[t];
    if (id < 0 || id >= vocab_size) {
      id = 0;
    }
    float scale = scales[id];
    float zp = static_cast<float>(zero_points[id]);

    const int8_t* row_packed = packed_data + id * (emb_dim / 2);
    float* row_out = output_data + t * emb_dim;

    for (int c = 0; c < emb_dim; ++c) {
      int byte_idx = c / 2;
      int bit_shift = (c % 2) * 4;
      int8_t q = (row_packed[byte_idx] >> bit_shift) & 0x0F;
      if (q & 0x08) {
        q |= 0xF0;
      }
      row_out[c] = (static_cast<float>(q) - zp) * scale;
    }
  }
  return absl::OkStatus();
}

int32_t CpuLmHeadAndArgMax(absl::Span<const float> hidden_states,
                           absl::Span<const float> dequantized_emb,
                           int vocab_size, int emb_dim) {
  float max_logit = -std::numeric_limits<float>::infinity();
  int32_t max_idx = 0;

  for (int i = 0; i < vocab_size; ++i) {
    const float* row_data = dequantized_emb.data() + i * emb_dim;
    float dot_product = 0.0f;
    for (int c = 0; c < emb_dim; ++c) {
      dot_product += hidden_states[c] * row_data[c];
    }
    if (dot_product > max_logit) {
      max_logit = dot_product;
      max_idx = i;
    }
  }
  return max_idx;
}

struct OpenClDecodeState {
  int max_cache_len = 0;
  Gemma3_MlDrift_Decode_Inputs inputs;
  Gemma3_MlDrift_Decode_Outputs outputs;
  std::unique_ptr<MlDriftModelRunner<Gemma3_MlDrift_Decode_Model,
                                     Gemma3_MlDrift_Decode_Inputs,
                                     Gemma3_MlDrift_Decode_Outputs>>
      runner;
  std::vector<float> global_cos_table;
  std::vector<float> global_sin_table;
  std::vector<float> local_cos_table;
  std::vector<float> local_sin_table;
  std::unique_ptr<MlDriftBuffer> input_ids_gpu;
  std::unique_ptr<MlDriftBuffer> cache_params_gpu;
  std::unique_ptr<OpenClParamUpdater> param_updater;
  WeightView emb_weights;
  std::vector<float> dequantized_embedding_table;
};

struct DecodeTiming {
  absl::Duration cpu_prep;
  absl::Duration uploads;
  absl::Duration run;
  absl::Duration readback;
  absl::Duration argmax;
};

absl::StatusOr<PrefillResult> RunPrefillOpenCl(
    const Gemma3Config& config, const std::vector<WeightView>& weight_views,
    const absl::flat_hash_map<std::string, absl::Span<const float>>&
        float_weights,
    const absl::flat_hash_map<std::string, absl::Span<const int8_t>>&
        int8_weights,
    const std::vector<int32_t>& input_ids, int seq_len, int max_cache_len,
    absl::Span<const float> dequantized_embedding_table, bool verbose,
    MlDriftClBackend::Environment* shared_env,
    MlDriftBuildContext* shared_build_ctx,
    std::unique_ptr<MlDriftModelRunner<
        Gemma3_MlDrift_Model, Gemma3_MlDrift_Inputs, Gemma3_MlDrift_Outputs>>*
        runner_keep_alive) {
  Gemma3_MlDrift_Inputs inputs;
  inputs.input_ids = TensorMLD(
      {.name = "input_ids", .type = Type::kI32, .shape = {1, seq_len}});
  if (absl::GetFlag(FLAGS_bypass_embedding)) {
    inputs.embedded_input = TensorMLD({.name = "embedded_input",
                                       .type = Type::kFP32,
                                       .shape = {1, seq_len, config.emb_dim}});
  }
  inputs.position_ids = TensorMLD({.name = "position_ids",
                                   .type = Type::kI32,
                                   .shape = {1, 1, seq_len, 1}});
  inputs.slice_index =
      TensorMLD({.name = "slice_index", .type = Type::kI32, .shape = {1}});
  const int cache_heads = config.n_kv_groups;
  inputs.key_caches.reserve(config.n_layers);
  inputs.value_caches.reserve(config.n_layers);
  for (int i = 0; i < config.n_layers; ++i) {
    inputs.key_caches.push_back(
        TensorMLD({.name = absl::StrCat("key_cache_", i),
                   .type = Type::kFP32,
                   .shape = {1, cache_heads, max_cache_len, config.head_dim}}));
    inputs.value_caches.push_back(
        TensorMLD({.name = absl::StrCat("value_cache_", i),
                   .type = Type::kFP32,
                   .shape = {1, cache_heads, max_cache_len, config.head_dim}}));
  }
  inputs.weights = BuildMlDriftWeightTensors(weight_views);

  Gemma3_MlDrift_Outputs outputs;
  Gemma3_MlDrift_Model model(config);

  auto runner_ptr = std::make_unique<MlDriftModelRunner<
      Gemma3_MlDrift_Model, Gemma3_MlDrift_Inputs, Gemma3_MlDrift_Outputs>>(
      model, inputs, outputs, float_weights, int8_weights,
      /*build_gpu_model=*/true, ml_drift::CalculationsPrecision::F16,
      shared_env, shared_build_ctx);
  auto& runner = *runner_ptr;

  if (verbose) {
    ABSL_LOG(INFO) << "Model compilation time: "
                   << absl::ToDoubleMilliseconds(runner.compilation_time())
                   << " ms";
  }

  std::vector<int32_t> position_ids(static_cast<size_t>(seq_len));
  for (int i = 0; i < seq_len; ++i) {
    position_ids[i] = i;
  }
  if (absl::GetFlag(FLAGS_bypass_embedding)) {
    std::vector<float> embedded_input_data(seq_len * config.emb_dim);
    const WeightView* emb_weights = nullptr;
    for (const auto& view : weight_views) {
      if (view.name == "model.embed_tokens.weight") {
        emb_weights = &view;
        break;
      }
    }
    ABSL_CHECK(emb_weights != nullptr) << "Embedding weights not found";
    LRT_TENSOR_RETURN_IF_ERROR(CpuEmbeddingLookup(input_ids, *emb_weights,
                                                  embedded_input_data.data()));
    LRT_TENSOR_RETURN_IF_ERROR(
        runner.SetInput("embedded_input", embedded_input_data));
  }

  LRT_TENSOR_RETURN_IF_ERROR(runner.SetInput("input_ids", input_ids));
  LRT_TENSOR_RETURN_IF_ERROR(runner.SetInput("position_ids", position_ids));
  int32_t slice_idx = seq_len - 1;
  LRT_TENSOR_RETURN_IF_ERROR(
      runner.SetInput("slice_index", std::vector<int32_t>{slice_idx}));
  std::vector<float> zero_cache(static_cast<size_t>(cache_heads) *
                                    static_cast<size_t>(max_cache_len) *
                                    static_cast<size_t>(config.head_dim),
                                0.0f);
  for (int i = 0; i < config.n_layers; ++i) {
    LRT_TENSOR_RETURN_IF_ERROR(
        runner.SetInput(absl::StrCat("key_cache_", i), zero_cache));
    LRT_TENSOR_RETURN_IF_ERROR(
        runner.SetInput(absl::StrCat("value_cache_", i), zero_cache));
  }

  const auto prefill_run_start = absl::Now();
  LRT_TENSOR_RETURN_IF_ERROR(runner.Run());
  const auto prefill_run_duration = absl::Now() - prefill_run_start;
  if (absl::GetFlag(FLAGS_profile)) {
    std::cout << "Profiling Prefill..." << std::endl;
    auto profiling_info_or = runner.Profile();
    if (profiling_info_or.ok()) {
      std::cout << "Prefill Profiling Report:\n"
                << profiling_info_or.value().GetDetailedReport() << std::endl;
    } else {
      std::cerr << "Prefill Profiling failed: "
                << profiling_info_or.status().message() << std::endl;
    }
  }

  PrefillResult result;
  if (config.bypass_lm_head) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto hidden_states,
                                runner.GetFloatOutput("output"));
    int32_t next_token =
        CpuLmHeadAndArgMax(hidden_states, dequantized_embedding_table,
                           config.vocab_size, config.emb_dim);
    result.output_ids = {next_token};
  } else {
    LRT_TENSOR_ASSIGN_OR_RETURN(result.output_ids,
                                runner.GetInt32Output("output"));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(result.embedded_input,
                              runner.GetFloatOutput("embedded_input"));

  {
    auto logits_or = runner.GetFloatOutput("logits");
    if (logits_or.ok()) {
      const auto& logits = logits_or.value();
      int vocab_size = config.vocab_size;
      const float* last_tok_logits = logits.data();
      if (logits.size() == seq_len * vocab_size) {
        last_tok_logits = logits.data() + (seq_len - 1) * vocab_size;
      }
      std::string first_10_logits = "";
      float max_logit = last_tok_logits[0];
      int max_idx = 0;
      for (int i = 0; i < vocab_size; ++i) {
        if (i < 10) {
          absl::StrAppend(&first_10_logits, last_tok_logits[i], " ");
        }
        if (last_tok_logits[i] > max_logit) {
          max_logit = last_tok_logits[i];
          max_idx = i;
        }
      }
      ABSL_LOG(WARNING) << "CPU Prefill logits (first 10): " << first_10_logits;
      ABSL_LOG(WARNING) << "CPU Prefill max logit: " << max_logit
                        << " at index " << max_idx;
    }
  }

  result.key_caches.resize(config.n_layers);
  result.value_caches.resize(config.n_layers);
  for (int i = 0; i < config.n_layers; ++i) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        result.key_caches[i],
        runner.GetFloatOutput(absl::StrCat("output_key_cache_", i)));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        result.value_caches[i],
        runner.GetFloatOutput(absl::StrCat("output_value_cache_", i)));
  }
  result.run_time = prefill_run_duration;
  if (runner_keep_alive != nullptr) {
    *runner_keep_alive = std::move(runner_ptr);
  }
  return result;
}

absl::StatusOr<std::unique_ptr<OpenClDecodeState>> BuildDecodeRunnerOpenCl(
    const Gemma3Config& config, const std::vector<WeightView>& weight_views,
    const absl::flat_hash_map<std::string, absl::Span<const float>>&
        float_weights,
    const absl::flat_hash_map<std::string, absl::Span<const int8_t>>&
        int8_weights,
    int max_cache_len, absl::Span<const float> dequantized_embedding_table,
    bool verbose, MlDriftClBackend::Environment* shared_env,
    MlDriftBuildContext* shared_build_ctx,
    MlDriftModelRunner<Gemma3_MlDrift_Model, Gemma3_MlDrift_Inputs,
                       Gemma3_MlDrift_Outputs>* prefill_runner) {
  auto state = std::make_unique<OpenClDecodeState>();
  state->max_cache_len = max_cache_len;
  state->dequantized_embedding_table.assign(dequantized_embedding_table.begin(),
                                            dequantized_embedding_table.end());

  state->inputs.input_ids =
      TensorMLD({.name = "input_ids", .type = Type::kI32, .shape = {1, 1}});
  if (absl::GetFlag(FLAGS_bypass_embedding)) {
    state->inputs.embedded_input = TensorMLD({.name = "embedded_input",
                                              .type = Type::kFP32,
                                              .shape = {1, 1, config.emb_dim}});
  }
  state->inputs.rope_global_cos =
      TensorMLD({.name = "rope_global_cos",
                 .type = Type::kFP32,
                 .shape = {max_cache_len, config.head_dim}});
  state->inputs.rope_global_sin =
      TensorMLD({.name = "rope_global_sin",
                 .type = Type::kFP32,
                 .shape = {max_cache_len, config.head_dim}});
  state->inputs.rope_local_cos =
      TensorMLD({.name = "rope_local_cos",
                 .type = Type::kFP32,
                 .shape = {max_cache_len, config.head_dim}});
  state->inputs.rope_local_sin =
      TensorMLD({.name = "rope_local_sin",
                 .type = Type::kFP32,
                 .shape = {max_cache_len, config.head_dim}});
  state->inputs.sliding_attention_mask =
      TensorMLD({.name = "sliding_attention_mask",
                 .type = Type::kFP32,
                 .shape = {1, 1, 1, max_cache_len}});
  state->inputs.global_attention_mask =
      TensorMLD({.name = "global_attention_mask",
                 .type = Type::kFP32,
                 .shape = {1, 1, 1, max_cache_len}});
  state->inputs.cache_params =
      TensorMLD({.name = "cache_params", .type = Type::kI32, .shape = {3}});
  state->inputs.key_caches.reserve(config.n_layers);
  state->inputs.value_caches.reserve(config.n_layers);
  const int cache_heads = config.n_kv_groups;
  for (int i = 0; i < config.n_layers; ++i) {
    state->inputs.key_caches.push_back(
        TensorMLD({.name = absl::StrCat("key_cache_", i),
                   .type = Type::kFP32,
                   .shape = {1, cache_heads, max_cache_len, config.head_dim}}));
    state->inputs.value_caches.push_back(
        TensorMLD({.name = absl::StrCat("value_cache_", i),
                   .type = Type::kFP32,
                   .shape = {1, cache_heads, max_cache_len, config.head_dim}}));
  }
  for (const auto& view : weight_views) {
    if (view.name == "model.embed_tokens.weight") {
      state->emb_weights = view;
      break;
    }
  }
  state->inputs.weights = BuildMlDriftWeightTensors(weight_views);

  Gemma3_MlDrift_Decode_Model model(config);
  state->runner =
      std::make_unique<MlDriftModelRunner<Gemma3_MlDrift_Decode_Model,
                                          Gemma3_MlDrift_Decode_Inputs,
                                          Gemma3_MlDrift_Decode_Outputs>>(
          model, state->inputs, state->outputs, float_weights, int8_weights,
          /*build_gpu_model=*/false, ml_drift::CalculationsPrecision::F16,
          shared_env, shared_build_ctx);

  ABSL_LOG(INFO) << "BuildDecodeRunnerOpenCl: Retrieving weight tensors from "
                    "prefill runner";
  auto weight_tensors = prefill_runner->GetWeightTensors();

  ml_drift::ExternalTensorsInfo external_tensors;
  for (const auto& [id, tensor_ptr] : weight_tensors) {
    external_tensors.immutable_tensors[id] = tensor_ptr;
  }

  ABSL_LOG(INFO) << "BuildDecodeRunnerOpenCl: Building model";
  std::vector<TensorMLD> output_tensors;
  for (auto const& [name, tensor_ptr] : state->outputs.tensors()) {
    tensor_ptr->SetName(name);
    output_tensors.push_back(*tensor_ptr);
  }
  LRT_TENSOR_RETURN_IF_ERROR(state->runner->BuildModel(output_tensors));

  ABSL_LOG(INFO)
      << "BuildDecodeRunnerOpenCl: Init model with external weight tensors";
  LRT_TENSOR_RETURN_IF_ERROR(state->runner->Init(external_tensors));

  state->global_cos_table.resize(max_cache_len * config.head_dim);
  state->global_sin_table.resize(max_cache_len * config.head_dim);
  state->local_cos_table.resize(max_cache_len * config.head_dim);
  state->local_sin_table.resize(max_cache_len * config.head_dim);

  for (int p = 0; p < max_cache_len; ++p) {
    auto [g_cos, g_sin] = ComputeRopeCosSinForPosition(p, config.head_dim,
                                                       config.rope_global_base);
    auto [l_cos, l_sin] = ComputeRopeCosSinForPosition(p, config.head_dim,
                                                       config.rope_local_base);
    absl::c_copy(g_cos, state->global_cos_table.begin() + p * config.head_dim);
    absl::c_copy(g_sin, state->global_sin_table.begin() + p * config.head_dim);
    absl::c_copy(l_cos, state->local_cos_table.begin() + p * config.head_dim);
    absl::c_copy(l_sin, state->local_sin_table.begin() + p * config.head_dim);
  }

  LRT_TENSOR_RETURN_IF_ERROR(
      state->runner->SetInput("rope_global_cos", state->global_cos_table));
  LRT_TENSOR_RETURN_IF_ERROR(
      state->runner->SetInput("rope_global_sin", state->global_sin_table));
  LRT_TENSOR_RETURN_IF_ERROR(
      state->runner->SetInput("rope_local_cos", state->local_cos_table));
  LRT_TENSOR_RETURN_IF_ERROR(
      state->runner->SetInput("rope_local_sin", state->local_sin_table));

  if (absl::GetFlag(FLAGS_use_async_param_update) ||
      absl::GetFlag(FLAGS_use_gpu_param_update)) {
    LRT_TENSOR_ASSIGN_OR_RETURN(state->input_ids_gpu,
                                state->runner->GetInputBuffer("input_ids"));
    LRT_TENSOR_ASSIGN_OR_RETURN(state->cache_params_gpu,
                                state->runner->GetInputBuffer("cache_params"));
  }

  if (absl::GetFlag(FLAGS_use_gpu_param_update)) {
    state->param_updater = std::make_unique<OpenClParamUpdater>();
    LRT_TENSOR_RETURN_IF_ERROR(state->param_updater->Init(shared_env));
    auto* tensor = state->cache_params_gpu->context()->GetTensor(
        state->cache_params_gpu->value_id());
    if (tensor) {
      ABSL_LOG(WARNING) << "cache_params storage type: "
                        << (int)tensor->GetStorageType();
    }
  }

  return state;
}

absl::Status InitDecodeCaches(OpenClDecodeState* state,
                              const Gemma3Config& config,
                              const PrefillResult& prefill, int cache_len) {
  if (state == nullptr || state->runner == nullptr) {
    return absl::InvalidArgumentError("Decode state is not initialized.");
  }
  if (cache_len > state->max_cache_len) {
    return absl::InvalidArgumentError("Cache length exceeds max cache size.");
  }
  const int cache_heads = config.n_kv_groups;
  const size_t expected_size = static_cast<size_t>(cache_heads) *
                               static_cast<size_t>(state->max_cache_len) *
                               static_cast<size_t>(config.head_dim);
  for (int i = 0; i < config.n_layers; ++i) {
    if (prefill.key_caches[i].size() == expected_size &&
        prefill.value_caches[i].size() == expected_size) {
      LRT_TENSOR_RETURN_IF_ERROR(state->runner->SetInput(
          absl::StrCat("key_cache_", i), prefill.key_caches[i]));
      LRT_TENSOR_RETURN_IF_ERROR(state->runner->SetInput(
          absl::StrCat("value_cache_", i), prefill.value_caches[i]));
      continue;
    }

    int src_heads = config.n_kv_groups;
    if (cache_len > 0) {
      const int denom = cache_len * config.head_dim;
      if (denom > 0) {
        src_heads = static_cast<int>(prefill.key_caches[i].size() / denom);
      }
    }
    src_heads = std::max(1, src_heads);
    std::vector<float> key_padded =
        PadKvCache(prefill.key_caches[i], src_heads, cache_heads, cache_len,
                   state->max_cache_len, config.head_dim);
    std::vector<float> value_padded =
        PadKvCache(prefill.value_caches[i], src_heads, cache_heads, cache_len,
                   state->max_cache_len, config.head_dim);
    LRT_TENSOR_RETURN_IF_ERROR(
        state->runner->SetInput(absl::StrCat("key_cache_", i), key_padded));
    LRT_TENSOR_RETURN_IF_ERROR(
        state->runner->SetInput(absl::StrCat("value_cache_", i), value_padded));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int32_t>> RunDecodeStepOpenCl(
    OpenClDecodeState* state, const Gemma3Config& config, int32_t current_token,
    int cache_len, bool readback_output, bool verbose,
    MlDriftClBackend::Environment* env, bool wait_for_completion,
    DecodeTiming* timing) {
  if (state == nullptr || state->runner == nullptr) {
    return absl::InvalidArgumentError("Decode state is not initialized.");
  }
  const auto prep_start = absl::Now();

  std::vector<int32_t> cache_params = BuildMlDriftDecodeCacheParams(cache_len);
  const auto prep_end = absl::Now();
  if (timing != nullptr) {
    timing->cpu_prep += (prep_end - prep_start);
  }

  const auto upload_start = absl::Now();
  const bool async_param =
      absl::GetFlag(FLAGS_use_async_param_update) && !config.bypass_lm_head;
  if (async_param) {
    if (absl::GetFlag(FLAGS_bypass_embedding)) {
      std::vector<float> embedded_input_data(config.emb_dim);
      LRT_TENSOR_RETURN_IF_ERROR(CpuEmbeddingLookup(
          {current_token}, state->emb_weights, embedded_input_data.data()));
      LRT_TENSOR_RETURN_IF_ERROR(
          state->runner->SetInput("embedded_input", embedded_input_data));
    }
    if (readback_output) {
      LRT_TENSOR_RETURN_IF_ERROR(state->runner->SetInput(
          "input_ids", std::vector<int32_t>{current_token}));
    }
    if (absl::GetFlag(FLAGS_use_gpu_param_update)) {
      const int active_tokens = cache_len + 1;
      const int active_tokens_aligned = (active_tokens + 31) / 32 * 32;
      LRT_TENSOR_RETURN_IF_ERROR(state->param_updater->Update(
          env, state->cache_params_gpu->GetMemoryForWriting(), cache_len,
          active_tokens, active_tokens_aligned));
    } else {
      LRT_TENSOR_RETURN_IF_ERROR(env->queue()->EnqueueWriteBuffer(
          state->cache_params_gpu->GetMemoryForWriting(), 3 * sizeof(int32_t),
          cache_params.data(), /*async=*/true));
    }
  } else {
    if (absl::GetFlag(FLAGS_bypass_embedding)) {
      std::vector<float> embedded_input_data(config.emb_dim);
      LRT_TENSOR_RETURN_IF_ERROR(CpuEmbeddingLookup(
          {current_token}, state->emb_weights, embedded_input_data.data()));
      LRT_TENSOR_RETURN_IF_ERROR(
          state->runner->SetInput("embedded_input", embedded_input_data));
    }
    LRT_TENSOR_RETURN_IF_ERROR(state->runner->SetInput(
        "input_ids", std::vector<int32_t>{current_token}));
    if (absl::GetFlag(FLAGS_use_gpu_param_update)) {
      const int active_tokens = cache_len + 1;
      const int active_tokens_aligned = (active_tokens + 31) / 32 * 32;
      LRT_TENSOR_RETURN_IF_ERROR(state->param_updater->Update(
          env, state->cache_params_gpu->GetMemoryForWriting(), cache_len,
          active_tokens, active_tokens_aligned));
    } else {
      LRT_TENSOR_RETURN_IF_ERROR(
          state->runner->SetInput("cache_params", cache_params));
    }
  }
  const auto upload_end = absl::Now();
  if (timing != nullptr) {
    timing->uploads += (upload_end - upload_start);
  }

  const auto run_start = absl::Now();
  const bool use_enqueue = absl::GetFlag(FLAGS_use_enqueue_wait);
  if (use_enqueue) {
    LRT_TENSOR_RETURN_IF_ERROR(state->runner->Enqueue());
    if (async_param) {
      LRT_TENSOR_RETURN_IF_ERROR(
          state->runner->GetOutput("output", *state->input_ids_gpu));
    }
    if (wait_for_completion) {
      LRT_TENSOR_RETURN_IF_ERROR(state->runner->Wait());
    } else {
      ml_drift::cl::clFlush(env->queue()->queue());
    }
  } else {
    LRT_TENSOR_RETURN_IF_ERROR(state->runner->Run());
    if (async_param) {
      LRT_TENSOR_RETURN_IF_ERROR(
          state->runner->GetOutput("output", *state->input_ids_gpu));
    }
  }
  const auto run_end = absl::Now();
  if (timing != nullptr) {
    timing->run += (run_end - run_start);
  }

  std::vector<int32_t> output_ids;
  if (readback_output || !async_param) {
    const auto readback_start = absl::Now();
    if (config.bypass_lm_head) {
      if (readback_output) {
        LRT_TENSOR_ASSIGN_OR_RETURN(auto hidden_states,
                                    state->runner->GetFloatOutput("output"));
        int32_t next_token = CpuLmHeadAndArgMax(
            hidden_states, state->dequantized_embedding_table,
            config.vocab_size, config.emb_dim);
        output_ids = {next_token};
      } else {
        output_ids = {0};
      }
    } else {
      LRT_TENSOR_ASSIGN_OR_RETURN(output_ids,
                                  state->runner->GetInt32Output("output"));
    }
    const auto readback_end = absl::Now();
    if (timing != nullptr) {
      timing->readback += (readback_end - readback_start);
    }
    if (verbose) {
      auto logits_or = state->runner->GetFloatOutput("logits");
      if (logits_or.ok()) {
        const auto& logits = logits_or.value();
        float abs_sum = 0.0f;
        for (float v : logits) abs_sum += std::abs(v);
        ABSL_LOG(WARNING) << "Decode step " << cache_len
                          << " logits absolute sum: " << abs_sum;
      }
    }
  }
  return output_ids;
}

absl::Status Run(absl::string_view weights_path,
                 absl::string_view tokenizer_path, absl::string_view prompt,
                 int max_tokens, bool verbose, Gemma3ModelVariant model_variant,
                 TfliteLoader::QuantizedLoadMode quantized_load_mode,
                 const BenchmarkConfig* benchmark_config) {
  // Load SentencePiece tokenizer
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto tokenizer, GemmaTokenizerSP::Load(std::string(tokenizer_path)));

  absl::flat_hash_map<std::string, TensorHandle> weights;

  LRT_TENSOR_ASSIGN_OR_RETURN(auto loader,
                              TfliteLoader::Load(std::string(weights_path)));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto config,
                              ResolveGemma3Config(loader, model_variant));
  config.bypass_lm_head = absl::GetFlag(FLAGS_bypass_lm_head);
  ABSL_LOG(INFO) << "Resolved Gemma3 config"
                 << " variant=" << Gemma3ModelVariantToString(model_variant)
                 << " layers=" << config.n_layers
                 << " emb_dim=" << config.emb_dim
                 << " hidden_dim=" << config.hidden_dim
                 << " head_dim=" << config.head_dim
                 << " n_heads=" << config.n_heads
                 << " n_kv_groups=" << config.n_kv_groups
                 << " vocab_size=" << config.vocab_size;
  std::string types_str = "";
  for (int i = 0; i < config.n_layers; ++i) {
    absl::StrAppend(&types_str, " ", config.GetLayerTypes()[i]);
  }
  ABSL_LOG(INFO) << "Layer types:" << types_str;
  auto weight_mapping = GetGemma3TfliteWeightMapping(config.n_layers);

  LRT_TENSOR_ASSIGN_OR_RETURN(
      weights,
      loader.LoadWeightsWithMapping(weight_mapping, quantized_load_mode));

  LRT_TENSOR_ASSIGN_OR_RETURN(auto weight_views, BuildWeightViews(weights));
  int fp32_count = 0;
  int i8_count = 0;
  int i4_count = 0;
  for (const auto& view : weight_views) {
    if (view.type == Type::kFP32)
      fp32_count++;
    else if (view.type == Type::kI8)
      i8_count++;
    else if (view.type == Type::kI4)
      i4_count++;
  }
  ABSL_LOG(INFO) << "Weights stats: FP32=" << fp32_count << ", I8=" << i8_count
                 << ", I4=" << i4_count;

  for (const auto& view : weight_views) {
    if (view.name == "model.embed_tokens.weight") {
      int dim0 = view.shape.empty() ? -1 : view.shape[0];
      int dim1 = view.shape.size() < 2 ? -1 : view.shape[1];
      ABSL_LOG(INFO) << "Embedding table info: "
                     << "type=" << ToString(view.type) << " shape=" << dim0
                     << "x" << dim1 << " elements=" << view.elements;
      if (view.quantization) {
        auto pc_quant = std::dynamic_pointer_cast<PerChannelAffineQuantization>(
            view.quantization);
        if (pc_quant) {
          ABSL_LOG(INFO) << "Embedding table quantization: "
                         << "scales=" << pc_quant->scales.size()
                         << " zero_points=" << pc_quant->zero_points.size()
                         << " quant_dim=" << pc_quant->quantized_dimension;
          if (!pc_quant->scales.empty()) {
            ABSL_LOG(INFO) << "Embedding table first scale: "
                           << pc_quant->scales[0];
          }
        }
      }
    }
  }
  auto float_weights = BuildFloatWeightMap(weight_views);
  auto int8_weights = BuildInt8WeightMap(weight_views);

  std::vector<int32_t> tokens;
  int decode_tokens = max_tokens;
  if (benchmark_config != nullptr) {
    decode_tokens = benchmark_config->decode_tokens;
    tokens = MakeBenchmarkTokens(*benchmark_config, config.vocab_size);
  } else {
    tokens = tokenizer.Encode(std::string(prompt), /*add_bos=*/true);
  }
  decode_tokens = std::max(0, decode_tokens);

  if (verbose) {
    ABSL_LOG(INFO) << "Prompt tokens: " << tokens.size();
    std::string token_str = "";
    for (size_t i = 0; i < tokens.size(); ++i) {
      absl::StrAppend(&token_str, "tok_", i, "(id=", tokens[i], ") ");
    }
    ABSL_LOG(INFO) << "Prompt token IDs: " << token_str;
  }

  int tokens_generated = 0;

  const int seq_len = static_cast<int>(tokens.size());
  int max_cache_len = ((seq_len + decode_tokens + 1) + 31) / 32 * 32;
  const char* force_cache_len_env = std::getenv("LRT_FORCE_CACHE_LEN");
  if (force_cache_len_env != nullptr) {
    max_cache_len = std::stoi(force_cache_len_env);
  }
  MlDriftClBackend::Environment shared_env;
  LRT_TENSOR_RETURN_IF_ERROR(MlDriftClBackend::InitEnvironment(&shared_env));

  const auto& gpu_info = shared_env.device().GetInfo();
  ABSL_LOG(WARNING) << "GPU Info: Adreno=" << gpu_info.IsAdreno()
                    << " Mali=" << gpu_info.IsMali()
                    << " PowerVR=" << gpu_info.IsPowerVR();
  ABSL_LOG(WARNING) << "Supports cl_qcom_recordable_queues: "
                    << gpu_info.SupportsExtension("cl_qcom_recordable_queues");
  ABSL_LOG(WARNING) << "Supports cl_khr_command_buffer: "
                    << gpu_info.SupportsExtension("cl_khr_command_buffer");

  std::vector<float> dequantized_embedding_table;
  if (config.bypass_lm_head) {
    const WeightView* emb_view = nullptr;
    for (const auto& view : weight_views) {
      if (view.name == "model.embed_tokens.weight") {
        emb_view = &view;
        break;
      }
    }
    ABSL_CHECK(emb_view != nullptr) << "Embedding table weights not found";
    dequantized_embedding_table.resize(config.vocab_size * config.emb_dim);
    std::vector<int32_t> all_tokens(config.vocab_size);
    for (int i = 0; i < config.vocab_size; ++i) all_tokens[i] = i;
    LRT_TENSOR_RETURN_IF_ERROR(CpuEmbeddingLookup(
        all_tokens, *emb_view, dequantized_embedding_table.data()));
  }

  MlDriftBuildContext shared_build_ctx;
  std::unique_ptr<MlDriftModelRunner<
      Gemma3_MlDrift_Model, Gemma3_MlDrift_Inputs, Gemma3_MlDrift_Outputs>>
      prefill_runner_keep_alive;

  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto prefill,
      RunPrefillOpenCl(config, weight_views, float_weights, int8_weights,
                       tokens, seq_len, max_cache_len,
                       dequantized_embedding_table, verbose, &shared_env,
                       &shared_build_ctx, &prefill_runner_keep_alive));
  if (seq_len > 0) {
    const char* label =
        benchmark_config != nullptr ? "Benchmark prefill" : "Prefill";
    ABSL_LOG(WARNING) << label << " " << seq_len << " tokens in "
                      << absl::ToDoubleMilliseconds(prefill.run_time) << " ms";
    ABSL_LOG(WARNING) << label << " average: "
                      << absl::ToDoubleMilliseconds(prefill.run_time) /
                             static_cast<double>(seq_len)
                      << " ms/token";
  }
  if (verbose) {
    std::string prefill_out_str = "";
    for (size_t i = 0; i < prefill.output_ids.size(); ++i) {
      absl::StrAppend(&prefill_out_str, prefill.output_ids[i], " ");
    }
    ABSL_LOG(INFO) << "Prefill output IDs: " << prefill_out_str;
  }

  int32_t current_token = prefill.output_ids.back();
  ABSL_LOG(INFO) << "Prefill predicted token ID: " << current_token;

  if (benchmark_config == nullptr) {
    std::cout << prompt << std::flush;
    std::cout << tokenizer.DecodeToken(current_token) << std::flush;
  }

  if (benchmark_config == nullptr && IsStopToken(current_token)) {
    if (benchmark_config == nullptr) {
      std::cout << std::endl;
    }
    return absl::OkStatus();
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto decode_state,
      BuildDecodeRunnerOpenCl(config, weight_views, float_weights, int8_weights,
                              max_cache_len, dequantized_embedding_table,
                              verbose, &shared_env, &shared_build_ctx,
                              prefill_runner_keep_alive.get()));
  LRT_TENSOR_RETURN_IF_ERROR(
      InitDecodeCaches(decode_state.get(), config, prefill, seq_len));
  int cache_len = seq_len;
  DecodeTiming decode_timing;
  // Initialize input_ids_gpu with the first token.
  if (absl::GetFlag(FLAGS_use_async_param_update)) {
    auto lock = decode_state->input_ids_gpu->LockMutable();
    int32_t* ptr = reinterpret_cast<int32_t*>(lock.data());
    ptr[0] = current_token;
    ptr[1] = 0;
    ptr[2] = 0;
    ptr[3] = 0;
  }

  const auto decode_start = absl::Now();
  int readback_period = absl::GetFlag(FLAGS_readback_period);
  if (benchmark_config != nullptr && readback_period == 1) {
    readback_period = 0;
  }
  const int batch_size = absl::GetFlag(FLAGS_decode_batch_size);
  const int effective_batch_size =
      (readback_period > 0) ? readback_period
                            : ((absl::GetFlag(FLAGS_bypass_embedding) ||
                                !absl::GetFlag(FLAGS_use_enqueue_wait))
                                   ? 1
                                   : batch_size);

  for (int step = 0; step < decode_tokens; ++step) {
    if (benchmark_config == nullptr && IsStopToken(current_token)) {
      break;
    }
    if (cache_len >= max_cache_len) {
      if (verbose) {
        ABSL_LOG(INFO) << "Reached max cache length at step " << step;
      }
      break;
    }

    const bool readback_output =
        (readback_period > 0) &&
        ((step % readback_period == readback_period - 1) ||
         (step == decode_tokens - 1));

    tokens_generated++;

    const bool wait_for_completion =
        readback_output ||
        (step % effective_batch_size == effective_batch_size - 1) ||
        (step == decode_tokens - 1);

    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto output_ids,
        RunDecodeStepOpenCl(decode_state.get(), config, current_token,
                            cache_len, readback_output, verbose, &shared_env,
                            wait_for_completion, &decode_timing));

    if (readback_output) {
      int32_t next_token = output_ids[0];
      const auto argmax_start = absl::Now();
      const auto argmax_end = absl::Now();
      decode_timing.argmax += (argmax_end - argmax_start);
      if (benchmark_config == nullptr && IsStopToken(next_token)) {
        break;
      }
      current_token = next_token;
      if (benchmark_config == nullptr) {
        std::cout << tokenizer.DecodeToken(current_token) << std::flush;
      }
    }
    cache_len += 1;
  }

  const auto decode_end = absl::Now();
  const auto decode_duration = decode_end - decode_start;

  if (absl::GetFlag(FLAGS_profile)) {
    std::cout << "Profiling Decode..." << std::endl;
    auto profiling_info_or = decode_state->runner->Profile();
    if (profiling_info_or.ok()) {
      std::cout << "Decode Profiling Report:\n"
                << profiling_info_or.value().GetDetailedReport(
                       {.add_shapes_info = true})
                << std::endl;
    } else {
      std::cerr << "Decode Profiling failed: "
                << profiling_info_or.status().message() << std::endl;
    }
  }

  if (benchmark_config == nullptr) {
    std::cout << std::endl;
  }
  const char* label =
      benchmark_config != nullptr ? "Benchmark decode" : "Decode";
  ABSL_LOG(WARNING) << label << " generated " << tokens_generated
                    << " tokens in " << absl::ToDoubleSeconds(decode_duration)
                    << " s";
  if (tokens_generated > 0) {
    ABSL_LOG(WARNING) << label << " average: "
                      << absl::ToDoubleMilliseconds(decode_duration) /
                             static_cast<double>(tokens_generated)
                      << " ms/token";
    ABSL_LOG(INFO) << "Decode breakdown (avg ms/token): "
                   << "prep="
                   << absl::ToDoubleMilliseconds(decode_timing.cpu_prep) /
                          static_cast<double>(tokens_generated)
                   << ", upload="
                   << absl::ToDoubleMilliseconds(decode_timing.uploads) /
                          static_cast<double>(tokens_generated)
                   << ", run="
                   << absl::ToDoubleMilliseconds(decode_timing.run) /
                          static_cast<double>(tokens_generated)
                   << ", readback="
                   << absl::ToDoubleMilliseconds(decode_timing.readback) /
                          static_cast<double>(tokens_generated)
                   << ", argmax="
                   << absl::ToDoubleMilliseconds(decode_timing.argmax) /
                          static_cast<double>(tokens_generated);
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace litert::tensor::examples

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string weights_path = absl::GetFlag(FLAGS_weights_path);
  std::string tokenizer_path = absl::GetFlag(FLAGS_tokenizer_path);
  std::string prompt = absl::GetFlag(FLAGS_prompt);
  int max_tokens = absl::GetFlag(FLAGS_max_tokens);
  std::string model_variant_flag = absl::GetFlag(FLAGS_model_variant);
  std::string weight_mode = absl::GetFlag(FLAGS_weight_mode);
  bool verbose = absl::GetFlag(FLAGS_verbose);
  bool benchmark = absl::GetFlag(FLAGS_benchmark);
  int benchmark_prefill_tokens = absl::GetFlag(FLAGS_benchmark_prefill_tokens);
  int benchmark_decode_tokens = absl::GetFlag(FLAGS_benchmark_decode_tokens);
  int benchmark_seed = absl::GetFlag(FLAGS_benchmark_seed);

  if (weights_path.empty()) {
    std::cerr << "Error: --weights_path is required" << std::endl;
    std::cerr << "Usage: " << argv[0] << " --weights_path=/path/to/model.tflite"
              << " --tokenizer_path=/path/to/tokenizer.model" << std::endl;
    return 1;
  }

  if (tokenizer_path.empty()) {
    std::cerr << "Error: --tokenizer_path is required" << std::endl;
    std::cerr << "Usage: " << argv[0] << " --weights_path=/path/to/model.tflite"
              << " --tokenizer_path=/path/to/tokenizer.model" << std::endl;
    return 1;
  }

  litert::tensor::examples::BenchmarkConfig benchmark_config = {
      .prefill_tokens = benchmark_prefill_tokens,
      .decode_tokens = benchmark_decode_tokens,
      .seed = benchmark_seed,
  };

  auto quantized_load_mode_or =
      litert::tensor::examples::ParseWeightMode(weight_mode);
  if (!quantized_load_mode_or.ok()) {
    std::cerr << "Error: " << quantized_load_mode_or.status() << std::endl;
    return 1;
  }
  auto model_variant_or =
      litert::tensor::examples::ParseGemma3ModelVariant(model_variant_flag);
  if (!model_variant_or.ok()) {
    std::cerr << "Error: " << model_variant_or.status() << std::endl;
    return 1;
  }

  absl::Status status = litert::tensor::examples::Run(
      weights_path, tokenizer_path, prompt, max_tokens, verbose,
      *model_variant_or, *quantized_load_mode_or,
      benchmark ? &benchmark_config : nullptr);

  if (!status.ok()) {
    std::cerr << "Error: " << status << std::endl;
    return 1;
  }

  return 0;
}
