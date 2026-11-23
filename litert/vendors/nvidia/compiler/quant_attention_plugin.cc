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

#include "litert/vendors/nvidia/compiler/quant_attention_plugin.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>

#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "NvInferRuntime.h"
#include "litert/vendors/nvidia/trtllm/decode_attention.h"
#include "litert/vendors/nvidia/trtllm/quant_cache.h"
#include "litert/vendors/nvidia/trtllm/tiled_attention.h"

namespace litert::nvidia {
namespace {

using Mode = QuantAttentionMode;

constexpr char kPluginName[] = "LiteRtNvidiaQuantAttention";
constexpr char kPluginVersion[] = "1";
constexpr char kPluginNamespace[] = "";
constexpr char kModeField[] = "mode";
constexpr char kCacheBitsField[] = "cache_bits";
constexpr char kQueryScaleField[] = "q_scale";
constexpr char kKeyScaleField[] = "k_scale";
constexpr char kValueScaleField[] = "v_scale";
constexpr char kTokenMajorField[] = "v_token_major";
constexpr char kUpdateCacheField[] = "update_cache";
constexpr int kNumFields = 7;
// The scores a mask hides stay out of the softmax.
constexpr float kFill = -std::numeric_limits<float>::infinity();
constexpr int kMaxInputs = 8;
constexpr int kMaxOutputs = 3;
// Query rows per KV head the decode kernel takes; more go to the tensor-core
// kernel.
constexpr int64_t kMaxDecodeRows = 16;
// The device memory an attached layer keeps for the decode kernel's one
// launch (LiteRtNvidiaQuantDecodeAttentionStateBytes): 64 heads of 16 rows.
constexpr size_t kDecodeStateBytes = 4096;

int32_t NumInputs(Mode mode) {
  switch (mode) {
    case Mode::kCacheUpdate:
      return 6;
    case Mode::kAttention:
      return 4;
    case Mode::kFusedAttentionUpdate:
      return 7;
    case Mode::kUpdateAttention:
      return 8;
  }
  return 0;
}

int32_t NumOutputs(Mode mode) {
  switch (mode) {
    case Mode::kCacheUpdate:
      return 2;
    case Mode::kAttention:
      return 1;
    case Mode::kFusedAttentionUpdate:
    case Mode::kUpdateAttention:
      return 3;
  }
  return 0;
}

// The key cache's input; the value cache follows it.
int32_t KeyCacheInput(Mode mode) { return mode == Mode::kCacheUpdate ? 0 : 1; }

// The output that updates the key cache; the value cache's follows it.
int32_t KeyCacheOutput(Mode mode) { return mode == Mode::kCacheUpdate ? 0 : 1; }

bool FloatType(nvinfer1::DataType type) {
  return type == nvinfer1::DataType::kHALF || type == nvinfer1::DataType::kBF16;
}

// What each input and output holds: 'f' a float tensor, 'c' a cache, 'm' the
// mask, 'i' an INT32 tensor.
const char* InputKinds(Mode mode) {
  switch (mode) {
    case Mode::kCacheUpdate:
      return "ccffii";
    case Mode::kAttention:
      return "fccm";
    case Mode::kFusedAttentionUpdate:
      return "fccmffi";
    case Mode::kUpdateAttention:
      return "fccmffii";
  }
  return "";
}

const char* OutputKinds(Mode mode) {
  switch (mode) {
    case Mode::kCacheUpdate:
      return "cc";
    case Mode::kAttention:
      return "f";
    case Mode::kFusedAttentionUpdate:
    case Mode::kUpdateAttention:
      return "fcc";
  }
  return "";
}

bool ValidDimensions(const nvinfer1::Dims& dims) {
  if (dims.nbDims != 4) return false;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0 || dims.d[i] > std::numeric_limits<int32_t>::max()) {
      return false;
    }
  }
  return true;
}

bool SameDimensions(const nvinfer1::Dims& a, const nvinfer1::Dims& b) {
  if (a.nbDims != b.nbDims) return false;
  for (int i = 0; i < a.nbDims; ++i) {
    if (a.d[i] != b.d[i]) return false;
  }
  return true;
}

size_t Volume(const nvinfer1::Dims& dims) {
  size_t volume = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    volume *= static_cast<size_t>(dims.d[i]);
  }
  return volume;
}

// The workspace's parts start on what a 16-byte load needs, and more.
size_t Aligned(size_t bytes) { return (bytes + 255) / 256 * 256; }

// What a layer's tensors say about it.
struct Layout {
  int32_t heads = 0;
  int32_t cache_len = 0;
  int32_t depth = 0;
  // kCacheUpdate: the tokens written. Otherwise the query rows per KV head,
  // the mask's rows and the chunk's keys after the cache's.
  int32_t tokens = 0;
  int32_t rows = 0;
  int32_t mask_rows = 0;
  int32_t new_len = 0;

  LiteRtNvidiaAttentionShape Shape() const {
    return {heads, rows, mask_rows, depth, cache_len, new_len};
  }
  // Few rows of the cache alone run the decode kernel.
  bool Decode() const { return new_len == 0 && rows <= kMaxDecodeRows; }
};

// Keys [1, H, n, D], or values that may be [1, n, H, D]: n.
int64_t RowTokens(const nvinfer1::Dims& rows, const Layout& layout,
                  bool token_major) {
  if (!ValidDimensions(rows) || rows.d[0] != 1 || rows.d[3] != layout.depth ||
      rows.d[token_major ? 2 : 1] != layout.heads) {
    return 0;
  }
  return rows.d[token_major ? 1 : 2];
}

// The layout of a layer with these tensors, validated against the kernels'
// contracts; false when the layer cannot run.
bool ParseLayout(const QuantAttentionOptions& options,
                 const nvinfer1::Dims* inputs, const nvinfer1::Dims* outputs,
                 Layout* layout) {
  const Mode mode = options.mode;
  const int32_t bits = options.quant.cache_bits;
  const auto& k = inputs[KeyCacheInput(mode)];
  const auto& v = inputs[KeyCacheInput(mode) + 1];
  if (!ValidDimensions(k) || !ValidDimensions(v) || k.d[0] != 1 ||
      v.d[0] != 1 || v.d[1] != k.d[1] || v.d[2] != k.d[3] || v.d[3] != k.d[2] ||
      (mode != Mode::kAttention &&
       (!SameDimensions(outputs[KeyCacheOutput(mode)], k) ||
        !SameDimensions(outputs[KeyCacheOutput(mode) + 1], v)))) {
    return false;
  }
  layout->heads = static_cast<int32_t>(k.d[1]);
  layout->cache_len = static_cast<int32_t>(k.d[2]);
  layout->depth = static_cast<int32_t>(k.d[3]) * (8 / bits);
  if (mode == Mode::kCacheUpdate) {
    const int64_t tokens = RowTokens(inputs[2], *layout, false);
    if (tokens < 1 || tokens > layout->cache_len ||
        RowTokens(inputs[3], *layout, options.v_token_major) != tokens ||
        inputs[4].nbDims != 1 || inputs[4].d[0] != 4 || inputs[5].nbDims != 1 ||
        inputs[5].d[0] != 4) {
      return false;
    }
    layout->tokens = static_cast<int32_t>(tokens);
    return true;
  }
  const auto& q = inputs[0];
  const auto& mask = inputs[3];
  if (!ValidDimensions(q) || !ValidDimensions(mask) || q.d[0] != 1 ||
      q.d[1] != layout->heads || q.d[3] != layout->depth ||
      !SameDimensions(outputs[0], q) || mask.d[0] != 1 || mask.d[1] != 1 ||
      q.d[2] % mask.d[2] != 0) {
    return false;
  }
  layout->rows = static_cast<int32_t>(q.d[2]);
  layout->mask_rows = static_cast<int32_t>(mask.d[2]);
  if (mode == Mode::kFusedAttentionUpdate) {
    const auto& params = inputs[6];
    if (bits != 8 || RowTokens(inputs[4], *layout, false) != mask.d[2] ||
        RowTokens(inputs[5], *layout, options.v_token_major) != mask.d[2] ||
        !ValidDimensions(params) || Volume(params) != 7) {
      return false;
    }
    layout->new_len = layout->mask_rows;
  }
  if (mask.d[3] != static_cast<int64_t>(layout->cache_len) + layout->new_len) {
    return false;
  }
  if (mode == Mode::kUpdateAttention) {
    // One token's rows: their elements are in one order whichever of the
    // heads and the token comes first.
    return layout->Decode() && RowTokens(inputs[4], *layout, false) == 1 &&
           RowTokens(inputs[5], *layout, options.v_token_major) == 1 &&
           inputs[6].nbDims == 1 && inputs[6].d[0] == 4 &&
           inputs[7].nbDims == 1 && inputs[7].d[0] == 4 &&
           LiteRtNvidiaQuantDecodeAttentionUpdateSupports(
               layout->rows, layout->depth, bits, layout->cache_len);
  }
  if (layout->Decode()) {
    return LiteRtNvidiaQuantDecodeAttentionSupports(layout->rows, layout->depth,
                                                    bits);
  }
  const LiteRtNvidiaAttentionShape shape = layout->Shape();
  return LiteRtNvidiaTiledQuantAttentionSupports(&shape, bits);
}

bool ValidDescriptors(const QuantAttentionOptions& options,
                      const nvinfer1::PluginTensorDesc* inputs,
                      const nvinfer1::PluginTensorDesc* outputs,
                      Layout* layout) {
  if (inputs == nullptr || outputs == nullptr) return false;
  const Mode mode = options.mode;
  const char* input_kinds = InputKinds(mode);
  const char* output_kinds = OutputKinds(mode);
  std::array<nvinfer1::Dims, kMaxInputs> input_dims;
  std::array<nvinfer1::Dims, kMaxOutputs> output_dims;
  nvinfer1::DataType float_type = nvinfer1::DataType::kFLOAT;
  const auto valid = [&](const nvinfer1::PluginTensorDesc& desc, char kind) {
    if (desc.format != nvinfer1::TensorFormat::kLINEAR) return false;
    switch (kind) {
      case 'c':
        return desc.type == nvinfer1::DataType::kINT8;
      case 'm':
        return desc.type == nvinfer1::DataType::kBOOL;
      case 'i':
        return desc.type == nvinfer1::DataType::kINT32;
      default:
        // The first float tensor decides the type of the others.
        if (float_type == nvinfer1::DataType::kFLOAT) float_type = desc.type;
        return FloatType(desc.type) && desc.type == float_type;
    }
  };
  for (int i = 0; i < NumInputs(mode); ++i) {
    if (!valid(inputs[i], input_kinds[i])) return false;
    input_dims[i] = inputs[i].dims;
  }
  for (int i = 0; i < NumOutputs(mode); ++i) {
    if (!valid(outputs[i], output_kinds[i])) return false;
    output_dims[i] = outputs[i].dims;
  }
  return ParseLayout(options, input_dims.data(), output_dims.data(), layout);
}

// The parts of the workspace of a layer on the tensor-core kernel: the
// dequantized query and the result in FP16, the chunk's quantized keys and
// values, then the kernel's own.
struct TiledWorkspace {
  size_t query = 0;
  size_t result = 0;
  size_t keys = 0;
  size_t values = 0;
  size_t kernel = 0;
  size_t bytes = 0;
};

TiledWorkspace PlanTiledWorkspace(const Layout& layout) {
  const size_t float_bytes = Aligned(static_cast<size_t>(layout.heads) *
                                     layout.rows * layout.depth * 2);
  const size_t row_bytes = Aligned(static_cast<size_t>(layout.heads) *
                                   layout.new_len * layout.depth);
  const LiteRtNvidiaAttentionShape shape = layout.Shape();
  TiledWorkspace plan;
  plan.result = float_bytes;
  plan.keys = plan.result + float_bytes;
  plan.values = plan.keys + row_bytes;
  plan.kernel = plan.values + row_bytes;
  plan.bytes =
      plan.kernel + LiteRtNvidiaTiledQuantAttentionWorkspaceBytes(&shape);
  return plan;
}

class QuantAttentionPlugin final : public nvinfer1::IPluginV3,
                                   public nvinfer1::IPluginV3OneCore,
                                   public nvinfer1::IPluginV3OneBuild,
                                   public nvinfer1::IPluginV3OneRuntime {
 public:
  explicit QuantAttentionPlugin(const QuantAttentionOptions& options) noexcept
      : options_(options),
        mode_(static_cast<int32_t>(options.mode)),
        v_token_major_(options.v_token_major ? 1 : 0),
        update_cache_(options.update_cache ? 1 : 0),
        fields_{{{kModeField, &mode_, nvinfer1::PluginFieldType::kINT32, 1},
                 {kCacheBitsField, &options_.quant.cache_bits,
                  nvinfer1::PluginFieldType::kINT32, 1},
                 {kQueryScaleField, &options_.quant.q_scale,
                  nvinfer1::PluginFieldType::kFLOAT32, 1},
                 {kKeyScaleField, &options_.quant.k_scale,
                  nvinfer1::PluginFieldType::kFLOAT32, 1},
                 {kValueScaleField, &options_.quant.v_scale,
                  nvinfer1::PluginFieldType::kFLOAT32, 1},
                 {kTokenMajorField, &v_token_major_,
                  nvinfer1::PluginFieldType::kINT32, 1},
                 {kUpdateCacheField, &update_cache_,
                  nvinfer1::PluginFieldType::kINT32, 1}}},
        field_collection_{static_cast<int32_t>(fields_.size()),
                          fields_.data()} {}

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

  ~QuantAttentionPlugin() override {
    if (decode_state_ != nullptr) {
      cudaFree(decode_state_);
    }
  }

  nvinfer1::IPluginV3* clone() noexcept override {
    return new (std::nothrow) QuantAttentionPlugin(options_);
  }

  const char* getPluginName() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }

  int32_t getNbOutputs() const noexcept override {
    return NumOutputs(options_.mode);
  }

  int32_t getOutputDataTypes(nvinfer1::DataType* output_types,
                             int32_t num_outputs,
                             const nvinfer1::DataType* input_types,
                             int32_t num_inputs) const noexcept override {
    if (output_types == nullptr || input_types == nullptr ||
        num_inputs != NumInputs(options_.mode) ||
        num_outputs != NumOutputs(options_.mode)) {
      return 1;
    }
    const char* kinds = OutputKinds(options_.mode);
    for (int i = 0; i < num_outputs; ++i) {
      // The attention has the query's type.
      output_types[i] =
          kinds[i] == 'c' ? nvinfer1::DataType::kINT8 : input_types[0];
    }
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t num_inputs, int32_t num_outputs) noexcept override {
    if (in_out == nullptr || num_inputs != NumInputs(options_.mode) ||
        num_outputs != NumOutputs(options_.mode) || position < 0 ||
        position >= num_inputs + num_outputs) {
      return false;
    }
    const auto& desc = in_out[position].desc;
    if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
      return false;
    }
    const char* input_kinds = InputKinds(options_.mode);
    const char kind = position < num_inputs
                          ? input_kinds[position]
                          : OutputKinds(options_.mode)[position - num_inputs];
    switch (kind) {
      case 'c':
        return desc.type == nvinfer1::DataType::kINT8;
      case 'm':
        return desc.type == nvinfer1::DataType::kBOOL;
      case 'i':
        return desc.type == nvinfer1::DataType::kINT32;
      default: {
        // The float tensors of a layer share the type of its first.
        const int32_t first =
            static_cast<int32_t>(std::strchr(input_kinds, 'f') - input_kinds);
        return FloatType(desc.type) &&
               (position == first || desc.type == in_out[first].desc.type);
      }
    }
  }

  int32_t getOutputShapes(
      const nvinfer1::DimsExprs* inputs, int32_t num_inputs,
      const nvinfer1::DimsExprs* shape_inputs, int32_t num_shape_inputs,
      nvinfer1::DimsExprs* outputs, int32_t num_outputs,
      nvinfer1::IExprBuilder& expr_builder) noexcept override {
    static_cast<void>(shape_inputs);
    static_cast<void>(expr_builder);
    if (inputs == nullptr || outputs == nullptr ||
        num_inputs != NumInputs(options_.mode) || num_shape_inputs != 0 ||
        num_outputs != NumOutputs(options_.mode)) {
      return 1;
    }
    const char* kinds = OutputKinds(options_.mode);
    int32_t cache = KeyCacheInput(options_.mode);
    for (int i = 0; i < num_outputs; ++i) {
      outputs[i] = kinds[i] == 'c' ? inputs[cache++] : inputs[0];
    }
    return 0;
  }

  int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) noexcept override {
    return inputs != nullptr && outputs != nullptr &&
                   num_inputs == NumInputs(options_.mode) &&
                   num_outputs == NumOutputs(options_.mode)
               ? 0
               : 1;
  }

  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                        int32_t num_inputs,
                        const nvinfer1::PluginTensorDesc* outputs,
                        int32_t num_outputs) noexcept override {
    Layout layout;
    return num_inputs == NumInputs(options_.mode) &&
                   num_outputs == NumOutputs(options_.mode) &&
                   ValidDescriptors(options_, inputs, outputs, &layout)
               ? 0
               : 1;
  }

  size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) const noexcept override {
    if (inputs == nullptr || outputs == nullptr ||
        num_inputs != NumInputs(options_.mode) ||
        num_outputs != NumOutputs(options_.mode) ||
        options_.mode == Mode::kCacheUpdate) {
      return 0;
    }
    std::array<nvinfer1::Dims, kMaxInputs> input_dims;
    std::array<nvinfer1::Dims, kMaxOutputs> output_dims;
    for (int i = 0; i < num_inputs; ++i) input_dims[i] = inputs[i].max;
    for (int i = 0; i < num_outputs; ++i) output_dims[i] = outputs[i].max;
    Layout layout;
    if (!ParseLayout(options_, input_dims.data(), output_dims.data(),
                     &layout)) {
      return 0;
    }
    return layout.Decode()
               ? LiteRtNvidiaDecodeAttentionWorkspaceBytes(
                     layout.heads, layout.rows, layout.cache_len, layout.depth)
               : PlanTiledWorkspace(layout).bytes;
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  const void* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    Layout layout;
    if (!ValidDescriptors(options_, input_desc, output_desc, &layout) ||
        inputs == nullptr || outputs == nullptr) {
      return 1;
    }
    for (int i = 0; i < NumInputs(options_.mode); ++i) {
      if (inputs[i] == nullptr) return 1;
    }
    for (int i = 0; i < NumOutputs(options_.mode); ++i) {
      if (outputs[i] == nullptr) return 1;
    }
    if (options_.mode != Mode::kCacheUpdate && workspace == nullptr) {
      return 1;
    }
    const bool bf16 =
        input_desc[options_.mode == Mode::kCacheUpdate ? 2 : 0].type ==
        nvinfer1::DataType::kBF16;
    const cudaError_t status =
        options_.mode == Mode::kCacheUpdate
            ? UpdateCaches(layout, bf16, input_desc, inputs, outputs, stream)
            : Attend(layout, bf16, input_desc, inputs, outputs, workspace,
                     stream);
    if (status != cudaSuccess) {
      std::fprintf(stderr,
                   "[LiteRtNvidiaQuantAttention] CUDA launch failed: %s (%d): "
                   "%s\n",
                   cudaGetErrorName(status), static_cast<int>(status),
                   cudaGetErrorString(status));
      return 1;
    }
    return 0;
  }

  nvinfer1::IPluginV3* attachToContext(
      nvinfer1::IPluginResourceContext* context) noexcept override {
    static_cast<void>(context);
    auto* attached = new (std::nothrow) QuantAttentionPlugin(options_);
    if (attached != nullptr && (options_.mode == Mode::kAttention ||
                                options_.mode == Mode::kUpdateAttention)) {
      attached->AllocateDecodeState();
    }
    return attached;
  }

  const nvinfer1::PluginFieldCollection* getFieldsToSerialize() noexcept
      override {
    return &field_collection_;
  }

 private:
  // The decode kernel's counters for this layer of one execution context,
  // zero between its launches. They are zeroed on a stream of their own: the
  // default stream is not to be used while another stream is captured.
  // Without them (no memory, no CUDA context) the attention takes its two
  // launches.
  void AllocateDecodeState() noexcept {
    void* state = nullptr;
    cudaStream_t stream = nullptr;
    if (cudaMalloc(&state, kDecodeStateBytes) == cudaSuccess &&
        cudaStreamCreate(&stream) == cudaSuccess &&
        cudaMemsetAsync(state, 0, kDecodeStateBytes, stream) == cudaSuccess &&
        cudaStreamSynchronize(stream) == cudaSuccess) {
      decode_state_ = state;
    } else {
      if (state != nullptr) cudaFree(state);
      static_cast<void>(cudaGetLastError());
    }
    if (stream != nullptr) cudaStreamDestroy(stream);
  }

  void* DecodeState(const Layout& layout) const {
    return LiteRtNvidiaQuantDecodeAttentionStateBytes(
               layout.heads, layout.rows) <= kDecodeStateBytes
               ? decode_state_
               : nullptr;
  }

  // The caches' input bytes into the output buffers, unless the runtime
  // bound both to one buffer.
  cudaError_t CarryCaches(const nvinfer1::PluginTensorDesc* input_desc,
                          const void* const* inputs, void* const* outputs,
                          cudaStream_t stream) const {
    for (int i = 0; i < 2; ++i) {
      const int32_t input = KeyCacheInput(options_.mode) + i;
      void* output = outputs[KeyCacheOutput(options_.mode) + i];
      if (output == inputs[input]) continue;
      const cudaError_t status =
          cudaMemcpyAsync(output, inputs[input], Volume(input_desc[input].dims),
                          cudaMemcpyDeviceToDevice, stream);
      if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
  }

  cudaError_t UpdateCaches(const Layout& layout, bool bf16,
                           const nvinfer1::PluginTensorDesc* input_desc,
                           const void* const* inputs, void* const* outputs,
                           cudaStream_t stream) const {
    const cudaError_t status = CarryCaches(input_desc, inputs, outputs, stream);
    if (status != cudaSuccess) return status;
    // The positions along the key cache's axis 2 and the value cache's axis 3.
    return LiteRtNvidiaLaunchQuantCacheWrite(
        outputs[0], outputs[1], inputs[2], inputs[3], bf16,
        options_.v_token_major, static_cast<const int32_t*>(inputs[4]) + 2,
        static_cast<const int32_t*>(inputs[5]) + 3, options_.quant.cache_bits,
        options_.quant.k_scale, options_.quant.v_scale, layout.heads,
        layout.cache_len, layout.depth, layout.tokens, stream);
  }

  cudaError_t Attend(const Layout& layout, bool bf16,
                     const nvinfer1::PluginTensorDesc* input_desc,
                     const void* const* inputs, void* const* outputs,
                     void* workspace, cudaStream_t stream) const {
    const bool* mask = static_cast<const bool*>(inputs[3]);
    if (options_.mode == Mode::kUpdateAttention) {
      const cudaError_t status =
          CarryCaches(input_desc, inputs, outputs, stream);
      if (status != cudaSuccess) return status;
      // The positions along the key cache's axis 2 and the value cache's
      // axis 3, as kCacheUpdate reads them.
      const LiteRtNvidiaQuantCacheToken token = {
          inputs[4], inputs[5], static_cast<const int32_t*>(inputs[6]) + 2,
          static_cast<const int32_t*>(inputs[7]) + 3};
      return LiteRtNvidiaLaunchQuantDecodeAttentionUpdate(
          inputs[0], bf16, outputs[1], outputs[2], &token, &options_.quant,
          mask, layout.mask_rows, layout.heads, layout.rows, layout.cache_len,
          layout.depth, kFill, outputs[0], workspace, DecodeState(layout),
          stream);
    }
    if (layout.Decode()) {
      // The caches are read only: no token is written.
      return LiteRtNvidiaLaunchQuantDecodeAttentionUpdate(
          inputs[0], bf16, const_cast<void*>(inputs[1]),
          const_cast<void*>(inputs[2]), /*token=*/nullptr, &options_.quant,
          mask, layout.mask_rows, layout.heads, layout.rows, layout.cache_len,
          layout.depth, kFill, outputs[0], workspace, DecodeState(layout),
          stream);
    }
    const TiledWorkspace plan = PlanTiledWorkspace(layout);
    char* base = static_cast<char*>(workspace);
    void* query = base + plan.query;
    void* keys = nullptr;
    void* values = nullptr;
    const size_t count =
        static_cast<size_t>(layout.heads) * layout.rows * layout.depth;
    cudaError_t status = LiteRtNvidiaLaunchQuantizeQuery(
        inputs[0], bf16, count, options_.quant.q_scale, query, stream);
    if (status != cudaSuccess) return status;
    if (layout.new_len > 0) {
      keys = base + plan.keys;
      values = base + plan.values;
      status = LiteRtNvidiaLaunchQuantizeRows(
          inputs[4], inputs[5], bf16, options_.v_token_major, layout.heads,
          layout.new_len, layout.depth, options_.quant.k_scale,
          options_.quant.v_scale, keys, values, stream);
      if (status != cudaSuccess) return status;
    }
    // The kernel's result is FP16.
    void* result = bf16 ? base + plan.result : outputs[0];
    const LiteRtNvidiaAttentionShape shape = layout.Shape();
    status = LiteRtNvidiaLaunchTiledQuantAttention(
        &shape, &options_.quant, query, inputs[1], inputs[2], keys, values,
        mask, kFill, result, base + plan.kernel, stream);
    if (status != cudaSuccess) return status;
    if (bf16) {
      status = LiteRtNvidiaLaunchHalfToBf16(result, count, outputs[0], stream);
      if (status != cudaSuccess) return status;
    }
    if (options_.mode != Mode::kFusedAttentionUpdate) {
      return cudaSuccess;
    }
    // The attention above read the caches before this update.
    status = CarryCaches(input_desc, inputs, outputs, stream);
    if (status != cudaSuccess || !options_.update_cache) return status;
    return LiteRtNvidiaLaunchQuantRingUpdate(
        outputs[1], outputs[2], keys, values,
        static_cast<const int32_t*>(inputs[6]), layout.heads, layout.cache_len,
        layout.depth, layout.new_len, stream);
  }

  QuantAttentionOptions options_;
  void* decode_state_ = nullptr;
  int32_t mode_;
  int32_t v_token_major_;
  int32_t update_cache_;
  std::array<nvinfer1::PluginField, kNumFields> fields_;
  nvinfer1::PluginFieldCollection field_collection_;
};

class QuantAttentionPluginCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  QuantAttentionPluginCreator() noexcept
      : fields_{
            {{kModeField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
             {kCacheBitsField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
             {kQueryScaleField, nullptr, nvinfer1::PluginFieldType::kFLOAT32,
              1},
             {kKeyScaleField, nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1},
             {kValueScaleField, nullptr, nvinfer1::PluginFieldType::kFLOAT32,
              1},
             {kTokenMajorField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
             {kUpdateCacheField, nullptr, nvinfer1::PluginFieldType::kINT32,
              1}}},
        field_collection_{static_cast<int32_t>(fields_.size()),
                          fields_.data()} {}

  nvinfer1::IPluginV3* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fields,
      nvinfer1::TensorRTPhase phase) noexcept override {
    static_cast<void>(name);
    static_cast<void>(phase);
    if (fields == nullptr || fields->fields == nullptr) {
      return nullptr;
    }
    QuantAttentionOptions options;
    int32_t mode = -1;
    for (int32_t i = 0; i < fields->nbFields; ++i) {
      const auto& field = fields->fields[i];
      if (field.name == nullptr || field.data == nullptr || field.length != 1) {
        continue;
      }
      if (field.type == nvinfer1::PluginFieldType::kFLOAT32) {
        const float value = *static_cast<const float*>(field.data);
        if (std::strcmp(field.name, kQueryScaleField) == 0) {
          options.quant.q_scale = value;
        } else if (std::strcmp(field.name, kKeyScaleField) == 0) {
          options.quant.k_scale = value;
        } else if (std::strcmp(field.name, kValueScaleField) == 0) {
          options.quant.v_scale = value;
        }
      } else if (field.type == nvinfer1::PluginFieldType::kINT32) {
        const int32_t value = *static_cast<const int32_t*>(field.data);
        if (std::strcmp(field.name, kModeField) == 0) {
          mode = value;
        } else if (std::strcmp(field.name, kCacheBitsField) == 0) {
          options.quant.cache_bits = value;
        } else if (std::strcmp(field.name, kTokenMajorField) == 0) {
          options.v_token_major = value != 0;
        } else if (std::strcmp(field.name, kUpdateCacheField) == 0) {
          options.update_cache = value != 0;
        }
      }
    }
    if (mode < 0 || mode > static_cast<int32_t>(Mode::kUpdateAttention)) {
      return nullptr;
    }
    options.mode = static_cast<Mode>(mode);
    return CreateQuantAttentionPlugin(options);
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
  std::array<nvinfer1::PluginField, kNumFields> fields_;
  nvinfer1::PluginFieldCollection field_collection_;
};

}  // namespace

nvinfer1::IPluginV3* CreateQuantAttentionPlugin(
    const QuantAttentionOptions& options) noexcept {
  const auto valid_scale = [](float scale) {
    return std::isfinite(scale) && scale > 0.0f;
  };
  if ((options.quant.cache_bits != 8 && options.quant.cache_bits != 4) ||
      !valid_scale(options.quant.q_scale) ||
      !valid_scale(options.quant.k_scale) ||
      !valid_scale(options.quant.v_scale) ||
      (options.mode == Mode::kFusedAttentionUpdate &&
       options.quant.cache_bits != 8)) {
    return nullptr;
  }
  return new (std::nothrow) QuantAttentionPlugin(options);
}

void EnsureQuantAttentionPluginRegistered() noexcept {}

REGISTER_TENSORRT_PLUGIN(QuantAttentionPluginCreator);

}  // namespace litert::nvidia
