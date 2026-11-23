// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/vendors/nvidia/experimental/megakernel/megakernel_plugin.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/compiler/moe_plugin.h"
#include "litert/vendors/nvidia/experimental/megakernel/megakernel.h"
#include "litert/vendors/nvidia/experimental/megakernel/megakernel_steps.h"
#include "litert/vendors/nvidia/experimental/megakernel/program.h"
#include "NvInferRuntime.h"
#include "litert/vendors/nvidia/trtllm/int4_block_gemv.h"

namespace litert::nvidia {
namespace {

constexpr char kPluginName[] = "LiteRtNvidiaMegakernel";
constexpr char kPluginVersion[] = "1";
constexpr char kPluginNamespace[] = "";
constexpr char kDescriptionField[] = "description";
// The serialized description starts with this tag ("MKD1").
constexpr uint32_t kDescriptionTag = 0x31444b4d;

// Plain values in native byte order: the description only travels between
// the compiler and the runtime of one machine, inside its engine.
class Writer {
 public:
  template <class T>
  void Put(const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
    bytes_.insert(bytes_.end(), bytes, bytes + sizeof(T));
  }
  void PutString(const std::string& value) {
    Put<uint64_t>(value.size());
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }
  std::vector<uint8_t> Take() { return std::move(bytes_); }

 private:
  std::vector<uint8_t> bytes_;
};

class Reader {
 public:
  explicit Reader(absl::Span<const uint8_t> bytes) : bytes_(bytes) {}
  template <class T>
  bool Get(T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (bytes_.size() - offset_ < sizeof(T)) return false;
    std::memcpy(&value, bytes_.data() + offset_, sizeof(T));
    offset_ += sizeof(T);
    return true;
  }
  bool GetString(std::string& value) {
    uint64_t size = 0;
    if (!Get(size) || bytes_.size() - offset_ < size) return false;
    value.assign(reinterpret_cast<const char*>(bytes_.data()) + offset_,
                 static_cast<size_t>(size));
    offset_ += static_cast<size_t>(size);
    return true;
  }
  bool AtEnd() const { return offset_ == bytes_.size(); }

 private:
  absl::Span<const uint8_t> bytes_;
  size_t offset_ = 0;
};

void Put(Writer& w, const MegakernelProjectionSource& p) {
  w.Put(p.holder);
  w.Put(p.weights);
  w.Put(p.scales);
  w.Put(p.rows);
  w.Put(p.cols);
}

bool Get(Reader& r, MegakernelProjectionSource& p) {
  return r.Get(p.holder) && r.Get(p.weights) && r.Get(p.scales) &&
         r.Get(p.rows) && r.Get(p.cols);
}

void Put(Writer& w, const MoeExpertSource& s) {
  w.PutString(s.model_path);
  w.Put(s.model_size);
  for (const uint64_t offset : s.offsets) w.Put(offset);
  for (const uint64_t size : s.sizes) w.Put(size);
  w.Put(s.scale_type);
  w.Put(s.router_scale_type);
  w.Put(s.dense_scale_type);
  w.Put(s.dense_hidden);
}

bool Get(Reader& r, MoeExpertSource& s) {
  if (!r.GetString(s.model_path) || !r.Get(s.model_size)) return false;
  for (uint64_t& offset : s.offsets) {
    if (!r.Get(offset)) return false;
  }
  for (uint64_t& size : s.sizes) {
    if (!r.Get(size)) return false;
  }
  return r.Get(s.scale_type) && r.Get(s.router_scale_type) &&
         r.Get(s.dense_scale_type) && r.Get(s.dense_hidden);
}

void Put(Writer& w, const MegakernelLayerSource& l) {
  w.Put(l.shared_kv);
  w.Put(l.q_heads);
  w.Put(l.kv_heads);
  w.Put(l.head_dim);
  w.Put(l.rotary_dim);
  w.Put(l.rope_min_timescale);
  w.Put(l.rope_max_timescale);
  w.Put(l.input_norm);
  Put(w, l.qkv);
  w.Put(l.q_norm);
  w.Put(l.k_norm);
  w.Put(l.k_cache);
  w.Put(l.v_cache);
  w.Put(l.cache_len);
  w.Put(l.ring);
  w.Put(l.mask);
  Put(w, l.o);
  w.Put(l.post_attention_norm);
  Put(w, l.arena);
  w.Put(l.experts);
  w.Put(l.top_k);
  w.Put(l.expert_hidden);
  w.Put(l.expert_gate);
  w.Put(l.dense_gate);
  w.Put(l.per_expert_scale);
  w.Put(l.router_norm);
  w.Put(l.router_multiplier);
  w.Put(l.expert_norm);
  w.Put(l.dense_norm);
  w.Put(l.dense_post);
  w.Put(l.moe_post);
  w.Put(l.sum_post);
  w.Put(l.residual_multiplier);
}

bool Get(Reader& r, MegakernelLayerSource& l) {
  return r.Get(l.shared_kv) && r.Get(l.q_heads) && r.Get(l.kv_heads) &&
         r.Get(l.head_dim) && r.Get(l.rotary_dim) &&
         r.Get(l.rope_min_timescale) && r.Get(l.rope_max_timescale) &&
         r.Get(l.input_norm) && Get(r, l.qkv) && r.Get(l.q_norm) &&
         r.Get(l.k_norm) && r.Get(l.k_cache) && r.Get(l.v_cache) &&
         r.Get(l.cache_len) && r.Get(l.ring) && r.Get(l.mask) && Get(r, l.o) &&
         r.Get(l.post_attention_norm) && Get(r, l.arena) && r.Get(l.experts) &&
         r.Get(l.top_k) && r.Get(l.expert_hidden) && r.Get(l.expert_gate) &&
         r.Get(l.dense_gate) && r.Get(l.per_expert_scale) &&
         r.Get(l.router_norm) && r.Get(l.router_multiplier) &&
         r.Get(l.expert_norm) && r.Get(l.dense_norm) && r.Get(l.dense_post) &&
         r.Get(l.moe_post) && r.Get(l.sum_post) && r.Get(l.residual_multiplier);
}

template <class T>
void PutVector(Writer& w, const std::vector<T>& values) {
  w.Put<uint64_t>(values.size());
  for (const T& value : values) w.Put(value);
}

template <class T>
bool GetVector(Reader& r, std::vector<T>& values, uint64_t limit) {
  uint64_t size = 0;
  if (!r.Get(size) || size > limit) return false;
  values.resize(static_cast<size_t>(size));
  for (T& value : values) {
    if (!r.Get(value)) return false;
  }
  return true;
}

}  // namespace

std::vector<uint8_t> SerializeMegakernelDescription(
    const MegakernelDescription& d) {
  Writer w;
  w.Put(kDescriptionTag);
  PutVector(w, d.input_types);
  w.Put(d.embeddings);
  w.Put(d.positions);
  PutVector(w, d.masks);
  w.Put(d.model_dim);
  w.Put(d.vocab);
  w.Put(d.norm_eps);
  w.Put(d.soft_cap);
  w.Put(d.soft_cap_in);
  w.Put<uint64_t>(d.layers.size());
  for (const auto& layer : d.layers) Put(w, layer);
  w.Put(d.final_norm);
  Put(w, d.head);
  w.Put(d.norm_values);
  w.Put(d.holders);
  w.Put(d.holder_granule);
  return w.Take();
}

std::optional<MegakernelDescription> ParseMegakernelDescription(
    absl::Span<const uint8_t> bytes) {
  Reader r(bytes);
  MegakernelDescription d;
  uint32_t tag = 0;
  uint64_t layers = 0;
  if (!r.Get(tag) || tag != kDescriptionTag ||
      !GetVector(r, d.input_types, 1024) || !r.Get(d.embeddings) ||
      !r.Get(d.positions) || !GetVector(r, d.masks, 1024) ||
      !r.Get(d.model_dim) || !r.Get(d.vocab) || !r.Get(d.norm_eps) ||
      !r.Get(d.soft_cap) || !r.Get(d.soft_cap_in) || !r.Get(layers) ||
      layers > kLiteRtNvidiaMegakernelRowsMaxLayers) {
    return std::nullopt;
  }
  d.layers.resize(static_cast<size_t>(layers));
  for (auto& layer : d.layers) {
    if (!Get(r, layer)) return std::nullopt;
  }
  if (!r.Get(d.final_norm) || !Get(r, d.head) || !r.Get(d.norm_values) ||
      !r.Get(d.holders) || !r.Get(d.holder_granule) || !r.AtEnd()) {
    return std::nullopt;
  }
  return d;
}

namespace {

void LogCudaError(const char* what, cudaError_t status) {
  std::fprintf(stderr, "[LiteRtNvidiaMegakernel] %s: %s (%d): %s\n", what,
               cudaGetErrorName(status), static_cast<int>(status),
               cudaGetErrorString(status));
}

// How a context runs the step, chosen when it attaches:
// LITERT_NVIDIA_MEGAKERNEL=interpreted runs the program as today's per-op
// launches, interpreted on the host (LiteRtNvidiaMegakernelLaunchReference);
// any other value that lowers the body (1) runs it as the megakernel.
bool InterpretedSteps() {
  const char* value = std::getenv("LITERT_NVIDIA_MEGAKERNEL");
  return value != nullptr && std::strcmp(value, "interpreted") == 0;
}

// What an execution context needs besides the engine's buffers: the
// experts' arenas, the program of each set of buffers it has run with, the
// program's workspace, the megakernel's state and the per-op launches' side
// stream. Made when the context attaches, outside any graph capture of its
// stream.
class StepResources {
 public:
  ~StepResources() {
    // A step replayed from a CUDA graph does not reach the plugin: few
    // enqueued steps, some of them captured, are the graph at work.
    if (steps_ > 0) {
      std::fprintf(stderr,
                   "[LiteRtNvidiaMegakernel] %lld steps enqueued, %lld of them "
                   "into a CUDA graph capture\n",
                   static_cast<long long>(steps_),
                   static_cast<long long>(captured_steps_));
    }
    if (device_programs_ != nullptr) cudaFree(device_programs_);
    if (host_programs_ != nullptr) cudaFreeHost(host_programs_);
    if (state_ != nullptr) cudaFree(state_);
    if (workspace_ != nullptr) cudaFree(workspace_);
    if (join_ != nullptr) cudaEventDestroy(join_);
    if (fork_ != nullptr) cudaEventDestroy(fork_);
    if (side_stream_ != nullptr) cudaStreamDestroy(side_stream_);
  }

  // Loads the arenas, builds the program's shapes and makes the rest; false
  // if any of it fails.
  bool Initialize(const MegakernelDescription& d, bool interpreted) {
    interpreted_ = interpreted;
    arenas_.resize(d.layers.size());
    for (size_t l = 0; l < d.layers.size(); ++l) {
      const auto& layer = d.layers[l];
      bool on_device = false;
      if (!LoadMoeArena(layer.arena, layer.experts, d.model_dim,
                        layer.expert_hidden, &arenas_[l], &on_device)) {
        std::fprintf(stderr,
                     "[LiteRtNvidiaMegakernel] the experts of layer %zu did "
                     "not load\n",
                     l);
        return false;
      }
    }
    // The shapes alone size the workspace. The megakernel's grid is as many
    // CTAs as the device keeps resident; the per-op launches read only the
    // program's layers and workspace, which no grid changes, and the largest
    // grid builds without asking the device what it keeps.
    ctas_ = interpreted_ ? kLiteRtNvidiaMegakernelMaxCtas : 0;
    cudaError_t status =
        cudaMallocHost(reinterpret_cast<void**>(&host_programs_),
                       kPrograms * sizeof(LiteRtNvidiaMegakernelProgram));
    if (status != cudaSuccess) {
      LogCudaError("the step's programs", status);
      return false;
    }
    Describe(d, nullptr);
    const LiteRtNvidiaMegakernelProgram& shapes = host_programs_[0];
    status =
        LiteRtNvidiaMegakernelBuildProgram(&model_, ctas_, &host_programs_[0]);
    if (status != cudaSuccess) {
      LogCudaError("the step's program", status);
      return false;
    }
    ctas_ = shapes.ctas;
    if ((status = cudaMalloc(&workspace_, shapes.workspace_bytes)) !=
            cudaSuccess ||
        (status = cudaMemset(workspace_, 0, shapes.workspace_bytes)) !=
            cudaSuccess) {
      LogCudaError("the step's workspace", status);
      return false;
    }
    if (interpreted_) {
      if ((status = cudaStreamCreateWithFlags(
               &side_stream_, cudaStreamNonBlocking)) != cudaSuccess ||
          (status = cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming)) !=
              cudaSuccess ||
          (status = cudaEventCreateWithFlags(&join_, cudaEventDisableTiming)) !=
              cudaSuccess) {
        LogCudaError("the per-op launches' side stream", status);
        return false;
      }
      std::fprintf(stderr,
                   "[LiteRtNvidiaMegakernel] the step as per-op launches, "
                   "workspace %.1f MB\n",
                   shapes.workspace_bytes / 1e6);
    } else {
      const size_t state_bytes = LiteRtNvidiaMegakernelStateBytes();
      if ((status = cudaMalloc(&state_, state_bytes)) != cudaSuccess ||
          (status = cudaMemset(state_, 0, state_bytes)) != cudaSuccess ||
          (status =
               cudaMalloc(reinterpret_cast<void**>(&device_programs_),
                          kPrograms * sizeof(LiteRtNvidiaMegakernelProgram))) !=
              cudaSuccess) {
        LogCudaError("the megakernel's state", status);
        return false;
      }
      std::fprintf(stderr,
                   "[LiteRtNvidiaMegakernel] the step as the megakernel: %d "
                   "instructions, %d CTAs x %d threads, %d B of shared "
                   "memory, workspace %.1f MB\n",
                   shapes.instruction_count, shapes.ctas, shapes.threads,
                   shapes.shared_bytes, shapes.workspace_bytes / 1e6);
    }
    // The memsets ran on the legacy stream; the steps run on the engine's.
    if ((status = cudaStreamSynchronize(nullptr)) != cudaSuccess) {
      LogCudaError("the step's resources", status);
      return false;
    }
    return true;
  }

  // One step from the engine's buffers, then the cache rows for the
  // engine's native cache writes.
  bool Run(const MegakernelDescription& d,
           const nvinfer1::PluginTensorDesc* input_desc,
           const void* const* inputs, void* const* outputs,
           cudaStream_t stream) {
    const size_t num_other = d.input_types.size();
    const size_t first_holder = num_other + 2 * d.layers.size() + 1;
    if (!checked_holders_) {
      if (!HoldersFit(d, input_desc, first_holder)) {
        std::fprintf(stderr,
                     "[LiteRtNvidiaMegakernel] a weight holder input does not "
                     "hold its projections\n");
        return false;
      }
      checked_holders_ = true;
    }
    // The program reads the caches, the norms and the weight holders: the
    // inputs behind the others.
    const int program =
        Program(d, inputs,
                absl::MakeConstSpan(inputs + num_other,
                                    first_holder + d.holders - num_other));
    if (program < 0) return false;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cudaError_t status = cudaStreamIsCapturing(stream, &capture);
    if (status != cudaSuccess) {
      LogCudaError("the stream's capture status", status);
      return false;
    }
    const bool capturing = capture != cudaStreamCaptureStatusNone;
    ++steps_;
    if (capturing) ++captured_steps_;
    LiteRtNvidiaMegakernelStep step{};
    step.embeddings = static_cast<const float*>(inputs[d.embeddings]);
    step.position = static_cast<const int32_t*>(inputs[d.positions]);
    for (size_t m = 0; m < std::min<size_t>(d.masks.size(), 2); ++m) {
      step.mask[m] = static_cast<const bool*>(inputs[d.masks[m]]);
    }
    step.logits = static_cast<float*>(outputs[0]);
    step.activations = static_cast<float*>(outputs[1]);
    if (interpreted_) {
      const LiteRtNvidiaMegakernelReferenceStreams streams = {side_stream_,
                                                              fork_, join_};
      status = LiteRtNvidiaMegakernelLaunchReference(
          &host_programs_[program], &step, workspace_, &streams, stream);
    } else {
      status = DeviceProgram(program, capturing, stream);
      if (status == cudaSuccess) {
        status = LiteRtNvidiaMegakernelLaunch(
            &host_programs_[program], &device_programs_[program], &step,
            workspace_, state_, /*trace=*/nullptr, stream);
      }
    }
    if (status != cudaSuccess) {
      LogCudaError("the step", status);
      return false;
    }
    LiteRtNvidiaMegakernelRowsParams rows{};
    rows.layers = static_cast<int32_t>(d.layers.size());
    rows.position = step.position;
    rows.rows = static_cast<int32_t*>(outputs[2]);
    for (size_t l = 0; l < d.layers.size(); ++l) {
      const auto& layer = d.layers[l];
      auto& out = rows.layer[l];
      out.k_cache = inputs[num_other + layer.k_cache];
      out.v_cache = inputs[num_other + layer.v_cache];
      out.k_row = outputs[3 + layer.k_cache];
      out.v_row = outputs[3 + layer.v_cache];
      out.heads = layer.kv_heads;
      out.len = layer.cache_len;
      out.depth = layer.head_dim;
      out.ring = layer.ring;
    }
    status = LiteRtNvidiaLaunchMegakernelRows(&rows, stream);
    if (status != cudaSuccess) {
      LogCudaError("the cache rows", status);
      return false;
    }
    return true;
  }

 private:
  // The programs of different sets of buffers a context keeps.
  static constexpr int kPrograms = 16;

  // The program's model: the description with the engine's buffers
  // (`inputs`), or its shapes alone (null).
  void Describe(const MegakernelDescription& d, const void* const* inputs) {
    const size_t num_other = d.input_types.size();
    const size_t norms_input = num_other + 2 * d.layers.size();
    const auto* norms = inputs != nullptr
                            ? static_cast<const float*>(inputs[norms_input])
                            : nullptr;
    auto norm = [&](int64_t offset) {
      return norms != nullptr ? norms + offset : nullptr;
    };
    // A projection's weights and scales in its holder's segment, which
    // starts at the first granule boundary in the holder.
    auto projection = [&](const MegakernelProjectionSource& p) {
      LiteRtNvidiaMegakernelProjection out = {nullptr, nullptr, p.rows, p.cols};
      if (inputs != nullptr) {
        const uintptr_t granule = static_cast<uintptr_t>(d.holder_granule);
        const uintptr_t holder =
            reinterpret_cast<uintptr_t>(inputs[norms_input + 1 + p.holder]);
        const auto* segment = reinterpret_cast<const uint8_t*>(
            (holder + granule - 1) & ~(granule - 1));
        out.weights = segment + p.weights;
        out.scales = segment + p.scales;
      }
      return out;
    };
    auto cache = [&](int32_t index) {
      return inputs != nullptr ? const_cast<void*>(inputs[num_other + index])
                               : nullptr;
    };
    layers_.assign(d.layers.size(), LiteRtNvidiaMegakernelLayer{});
    for (size_t l = 0; l < d.layers.size(); ++l) {
      const MegakernelLayerSource& s = d.layers[l];
      LiteRtNvidiaMegakernelLayer& t = layers_[l];
      t.shared_kv = s.shared_kv;
      t.q_heads = s.q_heads;
      t.kv_heads = s.kv_heads;
      t.head_dim = s.head_dim;
      t.rotary_dim = s.rotary_dim;
      t.rope_min_timescale = s.rope_min_timescale;
      t.rope_max_timescale = s.rope_max_timescale;
      t.input_norm = norm(s.input_norm);
      t.qkv = projection(s.qkv);
      t.q_norm = norm(s.q_norm);
      t.k_norm = norm(s.k_norm);
      t.k_cache = cache(s.k_cache);
      t.v_cache = cache(s.v_cache);
      t.cache_len = s.cache_len;
      t.ring = s.ring;
      t.mask = s.mask;
      t.o = projection(s.o);
      t.post_attention_norm = norm(s.post_attention_norm);
      t.arena = arenas_[l];
      t.experts = s.experts;
      t.top_k = s.top_k;
      t.expert_hidden = s.expert_hidden;
      t.expert_gate = s.expert_gate;
      t.dense_hidden = s.arena.dense_hidden;
      t.dense_gate = s.dense_gate;
      t.per_expert_scale = norm(s.per_expert_scale);
      t.router_norm = norm(s.router_norm);
      t.router_multiplier = s.router_multiplier;
      t.expert_norm = norm(s.expert_norm);
      t.dense_norm = norm(s.dense_norm);
      t.dense_post = norm(s.dense_post);
      t.moe_post = norm(s.moe_post);
      t.sum_post = norm(s.sum_post);
      t.residual_multiplier = s.residual_multiplier;
    }
    model_ = {};
    model_.layers = static_cast<int32_t>(d.layers.size());
    model_.model_dim = d.model_dim;
    model_.vocab = d.vocab;
    model_.norm_eps = d.norm_eps;
    model_.layer = layers_.data();
    model_.final_norm = norm(d.final_norm);
    model_.head = projection(d.head);
    model_.soft_cap = d.soft_cap;
    model_.soft_cap_in = d.soft_cap_in;
  }

  // The index of the program of `buffers` in host_programs_, built on its
  // first step; -1 if it cannot be.
  int Program(const MegakernelDescription& d, const void* const* inputs,
              absl::Span<const void* const> buffers) {
    auto built_for = [&](int i) {
      return std::equal(buffers.begin(), buffers.end(),
                        programs_[i].buffers.begin(),
                        programs_[i].buffers.end());
    };
    if (current_ >= 0 && built_for(current_)) return current_;
    for (int i = 0; i < static_cast<int>(programs_.size()); ++i) {
      if (built_for(i)) return current_ = i;
    }
    if (programs_.size() == static_cast<size_t>(kPrograms)) {
      std::fprintf(stderr,
                   "[LiteRtNvidiaMegakernel] more than %d sets of buffers in "
                   "one context\n",
                   kPrograms);
      return -1;
    }
    const int i = static_cast<int>(programs_.size());
    Describe(d, inputs);
    const cudaError_t status =
        LiteRtNvidiaMegakernelBuildProgram(&model_, ctas_, &host_programs_[i]);
    if (status != cudaSuccess) {
      LogCudaError("the step's program", status);
      return -1;
    }
    programs_.push_back(
        {std::vector<const void*>(buffers.begin(), buffers.end()), false});
    return current_ = i;
  }

  // Program `i` in device memory. Each program is copied there from its own
  // pinned bytes and never written with others: a CUDA graph captured with a
  // launch of it reads it at every replay. A copy made while the stream is
  // captured is a node of that graph (made again at every replay), and is
  // made once more on the next step outside a capture, since the capture
  // may have been dropped.
  cudaError_t DeviceProgram(int i, bool capturing, cudaStream_t stream) {
    if (programs_[i].copied) return cudaSuccess;
    const cudaError_t status = cudaMemcpyAsync(
        &device_programs_[i], &host_programs_[i],
        sizeof(LiteRtNvidiaMegakernelProgram), cudaMemcpyHostToDevice, stream);
    programs_[i].copied = status == cudaSuccess && !capturing;
    return status;
  }

  // Whether the weight holder inputs hold every projection.
  static bool HoldersFit(const MegakernelDescription& d,
                         const nvinfer1::PluginTensorDesc* input_desc,
                         size_t first_holder) {
    auto fits = [&](const MegakernelProjectionSource& p) {
      if (p.holder < 0 || p.holder >= d.holders) return false;
      const auto& dims = input_desc[first_holder + p.holder].dims;
      const int64_t bytes = dims.nbDims == 1 ? dims.d[0] * 8 : 0;
      const int64_t scales_end =
          p.scales + static_cast<int64_t>(p.rows) * (p.cols / 32) * 2;
      return p.weights >= 0 &&
             p.weights + static_cast<int64_t>(p.rows) * p.cols / 2 <=
                 p.scales &&
             scales_end <= bytes - d.holder_granule;
    };
    for (const auto& layer : d.layers) {
      if (!fits(layer.qkv) || !fits(layer.o)) return false;
    }
    return fits(d.head);
  }

  struct ProgramBuffers {
    // The caches, the norms and the weight holders it was built for.
    std::vector<const void*> buffers;
    // Copied to device memory outside a capture (DeviceProgram).
    bool copied = false;
  };

  bool interpreted_ = false;
  std::vector<LiteRtNvidiaMoeArena> arenas_;
  std::vector<LiteRtNvidiaMegakernelLayer> layers_;
  LiteRtNvidiaMegakernelModel model_{};
  int32_t ctas_ = 0;
  std::vector<ProgramBuffers> programs_;
  int current_ = -1;
  LiteRtNvidiaMegakernelProgram* host_programs_ = nullptr;    // pinned
  LiteRtNvidiaMegakernelProgram* device_programs_ = nullptr;  // megakernel
  void* workspace_ = nullptr;
  void* state_ = nullptr;               // megakernel
  cudaStream_t side_stream_ = nullptr;  // per-op launches
  cudaEvent_t fork_ = nullptr;
  cudaEvent_t join_ = nullptr;
  bool checked_holders_ = false;
  int64_t steps_ = 0;
  int64_t captured_steps_ = 0;
};

class MegakernelPlugin final : public nvinfer1::IPluginV3,
                               public nvinfer1::IPluginV3OneCore,
                               public nvinfer1::IPluginV3OneBuild,
                               public nvinfer1::IPluginV3OneRuntime {
 public:
  explicit MegakernelPlugin(MegakernelDescription description) noexcept
      : description_(std::move(description)),
        serialized_(SerializeMegakernelDescription(description_)) {
    field_ = {kDescriptionField, serialized_.data(),
              nvinfer1::PluginFieldType::kINT8,
              static_cast<int32_t>(serialized_.size())};
    field_collection_ = {1, &field_};
  }

  nvinfer1::IPluginCapability* getCapabilityInterface(
      nvinfer1::PluginCapabilityType type) noexcept override {
    switch (type) {
      case nvinfer1::PluginCapabilityType::kCORE:
        return static_cast<nvinfer1::IPluginV3OneCore*>(this);
      case nvinfer1::PluginCapabilityType::kBUILD:
        return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
      case nvinfer1::PluginCapabilityType::kRUNTIME:
        return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return nullptr;
  }

  nvinfer1::IPluginV3* clone() noexcept override {
    auto* copy = new (std::nothrow) MegakernelPlugin(description_);
    if (copy != nullptr) copy->resources_ = resources_;
    return copy;
  }

  const char* getPluginName() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }

  int32_t getNbOutputs() const noexcept override { return OutputCount(); }

  int32_t getOutputDataTypes(nvinfer1::DataType* output_types,
                             int32_t num_outputs,
                             const nvinfer1::DataType* input_types,
                             int32_t num_inputs) const noexcept override {
    if (output_types == nullptr || input_types == nullptr ||
        num_inputs != InputCount() || num_outputs != OutputCount()) {
      return 1;
    }
    for (int32_t i = 0; i < num_inputs; ++i) {
      if (input_types[i] != InputType(i)) return 1;
    }
    for (int32_t o = 0; o < num_outputs; ++o) output_types[o] = OutputType(o);
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t num_inputs, int32_t num_outputs) noexcept override {
    if (in_out == nullptr || num_inputs != InputCount() ||
        num_outputs != OutputCount() || position < 0 ||
        position >= num_inputs + num_outputs) {
      return false;
    }
    const nvinfer1::DataType type = position < num_inputs
                                        ? InputType(position)
                                        : OutputType(position - num_inputs);
    return in_out[position].desc.type == type &&
           in_out[position].desc.format == nvinfer1::TensorFormat::kLINEAR;
  }

  int32_t getOutputShapes(
      const nvinfer1::DimsExprs* inputs, int32_t num_inputs,
      const nvinfer1::DimsExprs* shape_inputs, int32_t num_shape_inputs,
      nvinfer1::DimsExprs* outputs, int32_t num_outputs,
      nvinfer1::IExprBuilder& expr_builder) noexcept override {
    static_cast<void>(shape_inputs);
    if (inputs == nullptr || outputs == nullptr || num_inputs != InputCount() ||
        num_shape_inputs != 0 || num_outputs != OutputCount()) {
      return 1;
    }
    const auto& d = description_;
    auto dims = [&](nvinfer1::DimsExprs& out,
                    std::initializer_list<int64_t> values) {
      out.nbDims = static_cast<int32_t>(values.size());
      int32_t i = 0;
      for (const int64_t value : values) {
        out.d[i++] = expr_builder.constant(value);
      }
    };
    dims(outputs[0], {1, 1, d.vocab});
    dims(outputs[1], {1, 1, d.model_dim});
    dims(outputs[2], {static_cast<int64_t>(d.layers.size())});
    for (const auto& layer : d.layers) {
      for (const int32_t cache : {layer.k_cache, layer.v_cache}) {
        dims(outputs[3 + cache], {1, layer.kv_heads, 1, layer.head_dim});
      }
    }
    return 0;
  }

  int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) noexcept override {
    return inputs != nullptr && outputs != nullptr &&
                   num_inputs == InputCount() && num_outputs == OutputCount()
               ? 0
               : 1;
  }

  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                        int32_t num_inputs,
                        const nvinfer1::PluginTensorDesc* outputs,
                        int32_t num_outputs) noexcept override {
    return inputs != nullptr && outputs != nullptr &&
                   num_inputs == InputCount() && num_outputs == OutputCount()
               ? 0
               : 1;
  }

  size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) const noexcept override {
    static_cast<void>(inputs);
    static_cast<void>(num_inputs);
    static_cast<void>(outputs);
    static_cast<void>(num_outputs);
    // The step's workspace is the context's (StepResources).
    return 0;
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  const void* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    static_cast<void>(output_desc);
    static_cast<void>(workspace);
    if (input_desc == nullptr || inputs == nullptr || outputs == nullptr) {
      return 1;
    }
    if (!MoeArenaLoadingEnabled()) {
      // An engine build times the plugin without experts: nothing to do.
      return 0;
    }
    if (resources_ == nullptr) return 1;
    return resources_->Run(description_, input_desc, inputs, outputs, stream)
               ? 0
               : 1;
  }

  nvinfer1::IPluginV3* attachToContext(
      nvinfer1::IPluginResourceContext* context) noexcept override {
    static_cast<void>(context);
    auto* copy = new (std::nothrow) MegakernelPlugin(description_);
    if (copy == nullptr) return nullptr;
    if (MoeArenaLoadingEnabled()) {
      auto resources = std::make_shared<StepResources>();
      if (resources->Initialize(description_, InterpretedSteps())) {
        copy->resources_ = std::move(resources);
      }
    }
    return copy;
  }

  const nvinfer1::PluginFieldCollection* getFieldsToSerialize() noexcept
      override {
    return &field_collection_;
  }

 private:
  int32_t NumOther() const noexcept {
    return static_cast<int32_t>(description_.input_types.size());
  }
  int32_t NumCaches() const noexcept {
    return static_cast<int32_t>(2 * description_.layers.size());
  }
  int32_t InputCount() const noexcept {
    return NumOther() + NumCaches() + 1 + description_.holders;
  }
  int32_t OutputCount() const noexcept { return 3 + NumCaches(); }

  nvinfer1::DataType InputType(int32_t index) const noexcept {
    if (index < NumOther()) {
      return static_cast<nvinfer1::DataType>(description_.input_types[index]);
    }
    if (index < NumOther() + NumCaches()) return nvinfer1::DataType::kHALF;
    if (index == NumOther() + NumCaches()) return nvinfer1::DataType::kFLOAT;
    return nvinfer1::DataType::kINT64;
  }

  static nvinfer1::DataType OutputType(int32_t index) noexcept {
    switch (index) {
      case 0:
      case 1:
        return nvinfer1::DataType::kFLOAT;
      case 2:
        return nvinfer1::DataType::kINT32;
      default:
        return nvinfer1::DataType::kHALF;
    }
  }

  MegakernelDescription description_;
  std::vector<uint8_t> serialized_;
  nvinfer1::PluginField field_;
  nvinfer1::PluginFieldCollection field_collection_;
  std::shared_ptr<StepResources> resources_;
};

class MegakernelPluginCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  MegakernelPluginCreator() {
    field_ = {kDescriptionField, nullptr, nvinfer1::PluginFieldType::kINT8, 0};
    field_collection_ = {1, &field_};
  }

  nvinfer1::IPluginV3* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fields,
      nvinfer1::TensorRTPhase phase) noexcept override {
    static_cast<void>(name);
    static_cast<void>(phase);
    if (fields == nullptr) return nullptr;
    for (int32_t i = 0; i < fields->nbFields; ++i) {
      const auto& field = fields->fields[i];
      if (field.name == nullptr ||
          std::strcmp(field.name, kDescriptionField) != 0 ||
          field.type != nvinfer1::PluginFieldType::kINT8 ||
          field.data == nullptr || field.length <= 0) {
        continue;
      }
      const auto description = ParseMegakernelDescription(
          absl::MakeConstSpan(static_cast<const uint8_t*>(field.data),
                              static_cast<size_t>(field.length)));
      if (!description.has_value()) return nullptr;
      return new (std::nothrow) MegakernelPlugin(*description);
    }
    return nullptr;
  }

  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return &field_collection_;
  }
  const char* getPluginName() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }

 private:
  nvinfer1::PluginField field_;
  nvinfer1::PluginFieldCollection field_collection_;
};

}  // namespace

nvinfer1::IPluginV3* CreateMegakernelPlugin(
    const MegakernelDescription& description) noexcept {
  if (description.layers.empty() ||
      description.layers.size() >
          static_cast<size_t>(kLiteRtNvidiaMegakernelRowsMaxLayers) ||
      description.holder_granule <= 0) {
    return nullptr;
  }
  return new (std::nothrow) MegakernelPlugin(description);
}

void EnsureMegakernelPluginRegistered() noexcept {}

REGISTER_TENSORRT_PLUGIN(MegakernelPluginCreator);

}  // namespace litert::nvidia
