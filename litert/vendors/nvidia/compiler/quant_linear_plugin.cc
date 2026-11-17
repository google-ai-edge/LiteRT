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

#include "litert/vendors/nvidia/compiler/quant_linear_plugin.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>

#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "NvInferRuntime.h"
#include "litert/vendors/nvidia/trtllm/quant_gemm.h"
#include "litert/vendors/nvidia/trtllm/quant_linear.h"

namespace litert::nvidia {
namespace {

constexpr char kPluginName[] = "LiteRtNvidiaQuantLinear";
constexpr char kPluginVersion[] = "1";
constexpr char kPluginNamespace[] = "";
constexpr char kKindField[] = "kind";
constexpr char kParamsField[] = "params";
constexpr char kOutputsField[] = "outputs";
constexpr char kWeightOffsetField[] = "weight_offset";
constexpr char kHolderGranuleField[] = "holder_granule";
constexpr int kFields = 5;

// Which kernel the layer launches.
enum Kind : int32_t {
  kLinear = 0,
  kGlue = 1,
  kQkvRope = 2,
  kGlueLinear = 3,
  kFactorizedMlp = 4,
};

constexpr int kFactorizedMlpStages = 4;

// kFactorizedMlp's parameters as a plan holds them: the stages, where each
// stage's weights are in its weights input (CreateQuantFactorizedMlpPlugin),
// and the glue in front of the MLP when `glue` is 1.
struct FactorizedMlpPlanParams {
  LiteRtNvidiaQuantFactorizedMlpParams stages = {};
  int64_t weight_offsets[kFactorizedMlpStages] = {};
  int32_t glue = 0;
  LiteRtNvidiaQuantGlueParams glue_params = {};
};

// The parameters of the layer's kernel: the member `kind` names; kGlueLinear
// has `linear` and `glue`, which a plan holds as one run of bytes.
struct Params {
  LiteRtNvidiaQuantLinearParams linear = {};
  LiteRtNvidiaQuantGlueParams glue = {};
  LiteRtNvidiaQuantQkvRopeParams rope = {};
  FactorizedMlpPlanParams factorized_mlp = {};
};
static_assert(offsetof(Params, glue) == sizeof(LiteRtNvidiaQuantLinearParams),
              "kGlueLinear serializes linear and glue as one run");

nvinfer1::DataType TensorType(int32_t type) {
  switch (type) {
    case kLiteRtNvidiaQuantFp16:
      return nvinfer1::DataType::kHALF;
    case kLiteRtNvidiaQuantBf16:
      return nvinfer1::DataType::kBF16;
    case kLiteRtNvidiaQuantFp32:
      return nvinfer1::DataType::kFLOAT;
    default:
      return nvinfer1::DataType::kINT8;
  }
}

// The rows [..., width] of a descriptor, or 0.
int64_t Rows(const nvinfer1::Dims& dims, int64_t width) {
  if (dims.nbDims < 1 || dims.d[dims.nbDims - 1] != width) {
    return 0;
  }
  int64_t rows = 1;
  for (int i = 0; i + 1 < dims.nbDims; ++i) {
    if (dims.d[i] <= 0 || dims.d[i] > std::numeric_limits<int32_t>::max() ||
        rows > std::numeric_limits<int32_t>::max() / dims.d[i]) {
      return 0;
    }
    rows *= dims.d[i];
  }
  return rows;
}

int64_t Elements(const nvinfer1::Dims& dims) {
  int64_t count = 1;
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) return -1;
    count *= dims.d[i];
  }
  return count;
}

class QuantLinearPlugin final : public nvinfer1::IPluginV3,
                                public nvinfer1::IPluginV3OneCore,
                                public nvinfer1::IPluginV3OneBuild,
                                public nvinfer1::IPluginV3OneRuntime {
 public:
  QuantLinearPlugin(int32_t kind, const Params& params, int32_t outputs,
                    int64_t weight_offset, int64_t holder_granule) noexcept
      : kind_(kind),
        params_(params),
        outputs_(outputs),
        weight_offset_(weight_offset),
        holder_granule_(holder_granule),
        fields_{
            {{kKindField, &kind_, nvinfer1::PluginFieldType::kINT32, 1},
             {kParamsField, ParamsData(), nvinfer1::PluginFieldType::kINT8,
              ParamsSize(kind)},
             {kOutputsField, &outputs_, nvinfer1::PluginFieldType::kINT32, 1},
             {kWeightOffsetField, &weight_offset_,
              nvinfer1::PluginFieldType::kINT64, 1},
             {kHolderGranuleField, &holder_granule_,
              nvinfer1::PluginFieldType::kINT64, 1}}},
        field_collection_{static_cast<int32_t>(fields_.size()),
                          fields_.data()} {}

  // The bytes of the parameters of `kind` in a plan.
  static int32_t ParamsSize(int32_t kind) noexcept {
    switch (kind) {
      case kLinear:
        return sizeof(LiteRtNvidiaQuantLinearParams);
      case kGlue:
        return sizeof(LiteRtNvidiaQuantGlueParams);
      case kGlueLinear:
        return sizeof(LiteRtNvidiaQuantLinearParams) +
               sizeof(LiteRtNvidiaQuantGlueParams);
      case kFactorizedMlp:
        return sizeof(FactorizedMlpPlanParams);
      default:
        return sizeof(LiteRtNvidiaQuantQkvRopeParams);
    }
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

  ~QuantLinearPlugin() override {
    if (factorized_mlp_state_ != nullptr) {
      cudaFree(factorized_mlp_state_);
    }
  }

  nvinfer1::IPluginV3* clone() noexcept override {
    return new (std::nothrow) QuantLinearPlugin(
        kind_, params_, outputs_, weight_offset_, holder_granule_);
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
    for (int32_t o = 0; o < num_outputs; ++o) {
      output_types[o] = OutputType(o);
    }
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
        num_shape_inputs != 0 || num_outputs != OutputCount() ||
        inputs[0].nbDims < 1) {
      return 1;
    }
    const int last = inputs[0].nbDims - 1;
    if (kind_ == kFactorizedMlp) {
      // x' and z of a glue in front have the shape of x.
      for (int32_t o = 0; o < num_outputs; ++o) {
        outputs[o] = inputs[0];
      }
      outputs[0].d[last] = expr_builder.constant(
          params_.factorized_mlp.stages.stages[kFactorizedMlpStages - 1].rows);
      return outputs[0].d[last] != nullptr ? 0 : 1;
    }
    if (kind_ == kLinear || kind_ == kGlueLinear) {
      // kGlueLinear: x' and z have the shape of x.
      for (int32_t o = 1; o < num_outputs; ++o) {
        outputs[o] = inputs[0];
      }
      outputs[0] = inputs[0];
      outputs[0].d[last] = expr_builder.constant(params_.linear.rows);
      return outputs[0].d[last] != nullptr ? 0 : 1;
    }
    if (kind_ == kGlue) {
      for (int32_t o = 0; o < num_outputs; ++o) {
        outputs[o] = inputs[0];
        if (GlueOutput(o) == kQuantGlueOutputRowScale) {
          outputs[o].d[last] = expr_builder.constant(1);
          if (outputs[o].d[last] == nullptr) return 1;
        }
      }
      return 0;
    }
    // tokens = every leading dim of the qkv input.
    const nvinfer1::IDimensionExpr* tokens = expr_builder.constant(1);
    for (int i = 0; i < last; ++i) {
      tokens = expr_builder.operation(nvinfer1::DimensionOperation::kPROD,
                                      *tokens, *inputs[0].d[i]);
    }
    const auto& p = params_.rope;
    const int32_t heads[3] = {p.q_heads, p.k_heads, p.v_heads};
    int32_t o = 0;
    for (int group = 0; group < 3; ++group) {
      if (heads[group] == 0) continue;
      const auto* count = expr_builder.constant(heads[group]);
      outputs[o].nbDims = 4;
      outputs[o].d[0] = expr_builder.constant(1);
      // q and k are [heads, tokens], v is [tokens, heads].
      outputs[o].d[1] = group == 2 ? tokens : count;
      outputs[o].d[2] = group == 2 ? count : tokens;
      outputs[o].d[3] = expr_builder.constant(p.depth);
      ++o;
    }
    return tokens != nullptr ? 0 : 1;
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

  // The many rows of a prefill chunk run on the tensor cores (quant_gemm.h),
  // which quantize a float activation into the workspace first; with the
  // glue in front (kGlueLinear) the glue's INT8 rows go there.
  size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc*,
                          int32_t) const noexcept override {
    if (inputs == nullptr || num_inputs < 1) {
      return 0;
    }
    if (kind_ == kGlueLinear) {
      const int64_t m = Rows(inputs[0].max, params_.glue.width);
      return m > 1 ? static_cast<size_t>(m) * params_.glue.width : 0;
    }
    if (kind_ == kFactorizedMlp) {
      return LiteRtNvidiaQuantFactorizedMlpWorkspaceBytes(
          &params_.factorized_mlp.stages);
    }
    if (kind_ != kLinear) {
      return 0;
    }
    const auto& p = params_.linear;
    const int64_t m =
        Rows(inputs[0].max, static_cast<int64_t>(p.segments) * p.columns);
    return LiteRtNvidiaQuantGemmWorkspaceBytes(&p, static_cast<int32_t>(m));
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  const void* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    static_cast<void>(output_desc);
    if (input_desc == nullptr || inputs == nullptr || outputs == nullptr) {
      return 1;
    }
    cudaError_t status = cudaErrorInvalidValue;
    if (kind_ == kLinear) {
      const auto& p = params_.linear;
      const int64_t m = Rows(input_desc[0].dims,
                             static_cast<int64_t>(p.segments) * p.columns);
      const uint8_t* weights = Weights(input_desc[1].dims, inputs[1]);
      const bool logits = p.requant == kLiteRtNvidiaQuantRequantLogits;
      if (m >= 1 && weights != nullptr &&
          (!logits || Elements(input_desc[3].dims) == m)) {
        const int8_t* lut =
            p.lut != 0 ? static_cast<const int8_t*>(inputs[3]) : nullptr;
        if (LiteRtNvidiaQuantGemmSupports(&p, static_cast<int32_t>(m))) {
          // Many rows: the tensor cores, with the row kernels' results.
          status = LiteRtNvidiaLaunchQuantGemm(
              &p, static_cast<int32_t>(m), inputs[0], weights,
              static_cast<const float*>(inputs[2]), lut, outputs[0], workspace,
              stream);
        } else {
          status = LiteRtNvidiaLaunchQuantLinear(
              &p, static_cast<int32_t>(m), inputs[0], weights,
              static_cast<const float*>(inputs[2]), lut,
              logits ? static_cast<const float*>(inputs[3]) : nullptr,
              outputs[0], stream);
        }
      }
    } else if (kind_ == kGlueLinear) {
      status =
          EnqueueGlueLinear(input_desc, inputs, outputs, workspace, stream);
    } else if (kind_ == kFactorizedMlp) {
      status =
          EnqueueFactorizedMlp(input_desc, inputs, outputs, workspace, stream);
    } else if (kind_ == kGlue) {
      const auto& p = params_.glue;
      const int64_t m = Rows(input_desc[0].dims, p.width);
      void* by_kind[4] = {nullptr, nullptr, nullptr, nullptr};
      for (int32_t o = 0; o < OutputCount(); ++o) {
        const int32_t output = GlueOutput(o);
        by_kind[output == kQuantGlueOutputX   ? 0
                : output == kQuantGlueOutputZ ? 1
                : output == kQuantGlueOutputQ ? 2
                                              : 3] = outputs[o];
      }
      if (m >= 1) {
        status = LiteRtNvidiaLaunchQuantGlue(
            &p, static_cast<int32_t>(m), inputs[0],
            p.branch != 0 ? static_cast<const int8_t*>(inputs[1]) : nullptr,
            p.branch != 0 ? static_cast<const float*>(inputs[2]) : nullptr,
            p.norm != 0 ? static_cast<const float*>(inputs[InputCount() - 1])
                        : nullptr,
            by_kind[0], by_kind[1], static_cast<int8_t*>(by_kind[2]),
            static_cast<float*>(by_kind[3]), stream);
      }
    } else {
      LiteRtNvidiaQuantQkvRopeParams p = params_.rope;
      const int32_t heads[3] = {p.q_heads, p.k_heads, p.v_heads};
      const int64_t tokens =
          Rows(input_desc[0].dims,
               static_cast<int64_t>(heads[0] + heads[1] + heads[2]) * p.depth);
      // The norm weights and the outputs of the projections that have heads.
      const float* norms[3] = {nullptr, nullptr, nullptr};
      void* results[3] = {nullptr, nullptr, nullptr};
      int32_t present = 0;
      for (int group = 0; group < 3; ++group) {
        if (heads[group] == 0) continue;
        norms[group] = static_cast<const float*>(inputs[2 + present]);
        results[group] = outputs[present];
        ++present;
      }
      if (tokens >= 1 && Elements(input_desc[1].dims) == tokens) {
        p.tokens = static_cast<int32_t>(tokens);
        status = LiteRtNvidiaLaunchQuantQkvRope(
            &p, static_cast<const int8_t*>(inputs[0]),
            static_cast<const int32_t*>(inputs[1]), norms[0], norms[1],
            norms[2], static_cast<const float*>(inputs[InputCount() - 1]),
            results[0], results[1], results[2], stream);
      }
    }
    if (status != cudaSuccess) {
      std::fprintf(stderr,
                   "[LiteRtNvidiaQuantLinear] CUDA launch failed (kind %d): "
                   "%s (%d): %s\n",
                   kind_, cudaGetErrorName(status), static_cast<int>(status),
                   cudaGetErrorString(status));
      return 1;
    }
    return 0;
  }

  nvinfer1::IPluginV3* attachToContext(
      nvinfer1::IPluginResourceContext* context) noexcept override {
    static_cast<void>(context);
    auto* attached = new (std::nothrow) QuantLinearPlugin(
        kind_, params_, outputs_, weight_offset_, holder_granule_);
    if (attached != nullptr && kind_ == kFactorizedMlp) {
      attached->AllocateFactorizedMlpState();
    }
    return attached;
  }

  const nvinfer1::PluginFieldCollection* getFieldsToSerialize() noexcept
      override {
    return &field_collection_;
  }

 private:
  // kGlueLinear: one launch for few rows; many rows run the glue into the
  // workspace and then the tensor cores, as the two layers did.
  cudaError_t EnqueueGlueLinear(const nvinfer1::PluginTensorDesc* input_desc,
                                const void* const* inputs, void* const* outputs,
                                void* workspace, cudaStream_t stream) const {
    const auto& g = params_.glue;
    const auto& p = params_.linear;
    const int64_t m = Rows(input_desc[0].dims, g.width);
    const uint8_t* weights = Weights(input_desc[1].dims, inputs[1]);
    if (m < 1 || weights == nullptr) {
      return cudaErrorInvalidValue;
    }
    const auto* multipliers = static_cast<const float*>(inputs[2]);
    const int8_t* lut =
        p.lut != 0 ? static_cast<const int8_t*>(inputs[3]) : nullptr;
    int32_t next = 3 + (p.lut != 0 ? 1 : 0);
    const int8_t* y = nullptr;
    const float* branch_weights = nullptr;
    if (g.branch != 0) {
      y = static_cast<const int8_t*>(inputs[next++]);
      branch_weights = static_cast<const float*>(inputs[next++]);
    }
    const float* norm_weights =
        g.norm != 0 ? static_cast<const float*>(inputs[next]) : nullptr;
    void* x_out = (outputs_ & kQuantGlueOutputX) != 0 ? outputs[1] : nullptr;
    void* z_out = (outputs_ & kQuantGlueOutputZ) != 0
                      ? outputs[x_out != nullptr ? 2 : 1]
                      : nullptr;
    if (m > 1 && workspace != nullptr &&
        LiteRtNvidiaQuantGemmSupports(&p, static_cast<int32_t>(m))) {
      auto* q = static_cast<int8_t*>(workspace);
      const cudaError_t status = LiteRtNvidiaLaunchQuantGlue(
          &g, static_cast<int32_t>(m), inputs[0], y, branch_weights,
          norm_weights, x_out, z_out, q, nullptr, stream);
      if (status != cudaSuccess) return status;
      return LiteRtNvidiaLaunchQuantGemm(&p, static_cast<int32_t>(m), q,
                                         weights, multipliers, lut, outputs[0],
                                         /*workspace=*/nullptr, stream);
    }
    return LiteRtNvidiaLaunchQuantGlueLinear(
        &g, &p, static_cast<int32_t>(m), inputs[0], y, branch_weights,
        norm_weights, weights, multipliers, lut, x_out, z_out, outputs[0],
        stream);
  }

  // kFactorizedMlp: the one launch for its one row; without a state (no memory,
  // no CUDA context) the launch runs the four products.
  cudaError_t EnqueueFactorizedMlp(const nvinfer1::PluginTensorDesc* input_desc,
                                   const void* const* inputs,
                                   void* const* outputs, void* workspace,
                                   cudaStream_t stream) const {
    const auto& p = params_.factorized_mlp;
    const auto& first = p.stages.stages[0];
    if (Rows(input_desc[0].dims,
             static_cast<int64_t>(first.segments) * first.columns) != 1) {
      return cudaErrorInvalidValue;
    }
    const uint8_t* weights[kFactorizedMlpStages];
    const float* multipliers[kFactorizedMlpStages];
    bool table = false;
    for (int s = 0; s < kFactorizedMlpStages; ++s) {
      const auto& stage = p.stages.stages[s];
      weights[s] = WeightsAt(input_desc[1 + s].dims, inputs[1 + s],
                             static_cast<uint64_t>(stage.rows) * stage.columns *
                                 stage.bit_width / 8,
                             p.weight_offsets[s]);
      if (weights[s] == nullptr) return cudaErrorInvalidValue;
      multipliers[s] =
          static_cast<const float*>(inputs[1 + kFactorizedMlpStages + s]);
      table = table || stage.lut != 0;
    }
    const auto* lut =
        table ? static_cast<const int8_t*>(inputs[1 + 2 * kFactorizedMlpStages])
              : nullptr;
    auto* output = static_cast<int8_t*>(outputs[0]);
    if (p.glue == 0) {
      return LiteRtNvidiaLaunchQuantFactorizedMlp(
          &p.stages, static_cast<const int8_t*>(inputs[0]), weights,
          multipliers, lut, factorized_mlp_state_, workspace, output, stream);
    }
    const auto& g = p.glue_params;
    int32_t next = FactorizedMlpFirstGlueInput();
    const int8_t* y = nullptr;
    const float* branch_weights = nullptr;
    if (g.branch != 0) {
      y = static_cast<const int8_t*>(inputs[next++]);
      branch_weights = static_cast<const float*>(inputs[next++]);
    }
    const float* norm_weights =
        g.norm != 0 ? static_cast<const float*>(inputs[next]) : nullptr;
    void* x_out = (outputs_ & kQuantGlueOutputX) != 0 ? outputs[1] : nullptr;
    void* z_out = (outputs_ & kQuantGlueOutputZ) != 0
                      ? outputs[x_out != nullptr ? 2 : 1]
                      : nullptr;
    return LiteRtNvidiaLaunchQuantGlueFactorizedMlp(
        &g, &p.stages, inputs[0], y, branch_weights, norm_weights, weights,
        multipliers, lut, factorized_mlp_state_, workspace, x_out, z_out,
        output, stream);
  }

  // The one launch's barrier counters for this layer of one execution
  // context, zero between its launches; zeroed on a stream of their own
  // (the default stream is not to be used while another stream is
  // captured).
  void AllocateFactorizedMlpState() noexcept {
    void* state = nullptr;
    cudaStream_t stream = nullptr;
    const size_t bytes = LiteRtNvidiaQuantFactorizedMlpStateBytes();
    if (cudaMalloc(&state, bytes) == cudaSuccess &&
        cudaStreamCreate(&stream) == cudaSuccess &&
        cudaMemsetAsync(state, 0, bytes, stream) == cudaSuccess &&
        cudaStreamSynchronize(stream) == cudaSuccess) {
      factorized_mlp_state_ = state;
    } else {
      if (state != nullptr) cudaFree(state);
      static_cast<void>(cudaGetLastError());
    }
    if (stream != nullptr) cudaStreamDestroy(stream);
  }

  const void* ParamsData() const noexcept {
    switch (kind_) {
      case kLinear:
      case kGlueLinear:
        return &params_.linear;
      case kGlue:
        return &params_.glue;
      case kFactorizedMlp:
        return &params_.factorized_mlp;
      default:
        return &params_.rope;
    }
  }

  int32_t InputCount() const noexcept {
    switch (kind_) {
      case kLinear:
        return 3 + (params_.linear.lut != 0 ? 1 : 0) +
               (params_.linear.requant == kLiteRtNvidiaQuantRequantLogits ? 1
                                                                          : 0);
      case kGlue:
        return 1 + (params_.glue.branch != 0 ? 2 : 0) +
               (params_.glue.norm != 0 ? 1 : 0);
      case kGlueLinear:
        return GlueLinearFirstGlueInput() + (params_.glue.branch != 0 ? 2 : 0) +
               (params_.glue.norm != 0 ? 1 : 0);
      case kFactorizedMlp:
        return FactorizedMlpFirstGlueInput() +
               (params_.factorized_mlp.glue != 0
                    ? (params_.factorized_mlp.glue_params.branch != 0 ? 2 : 0) +
                          (params_.factorized_mlp.glue_params.norm != 0 ? 1 : 0)
                    : 0);
      default:
        return 3 + RopeGroups();
    }
  }

  // kFactorizedMlp: x, the four stages' weights, their multipliers and the
  // table, then the glue's y and branch norm weights, then its norm weights.
  int32_t FactorizedMlpFirstGlueInput() const noexcept {
    return 1 + 2 * kFactorizedMlpStages + (FactorizedMlpHasTable() ? 1 : 0);
  }

  bool FactorizedMlpHasTable() const noexcept {
    for (const auto& stage : params_.factorized_mlp.stages.stages) {
      if (stage.lut != 0) return true;
    }
    return false;
  }

  // kGlueLinear: x, the product's weights, multipliers and table, then y and
  // the branch norm's weights, then the norm's weights.
  int32_t GlueLinearFirstGlueInput() const noexcept {
    return 3 + (params_.linear.lut != 0 ? 1 : 0);
  }

  nvinfer1::DataType InputType(int32_t index) const noexcept {
    if (kind_ == kFactorizedMlp) {
      const auto& p = params_.factorized_mlp;
      if (index == 0) {
        return p.glue != 0 ? TensorType(p.glue_params.x_type)
                           : nvinfer1::DataType::kINT8;
      }
      if (index <= kFactorizedMlpStages) {
        return p.weight_offsets[index - 1] >= 0 ? nvinfer1::DataType::kINT64
                                                : nvinfer1::DataType::kINT8;
      }
      if (index >= FactorizedMlpFirstGlueInput()) {
        return p.glue_params.branch != 0 &&
                       index == FactorizedMlpFirstGlueInput()
                   ? nvinfer1::DataType::kINT8
                   : nvinfer1::DataType::kFLOAT;
      }
      return index <= 2 * kFactorizedMlpStages ? nvinfer1::DataType::kFLOAT
                                               : nvinfer1::DataType::kINT8;
    }
    if (kind_ == kGlueLinear && index >= GlueLinearFirstGlueInput()) {
      return params_.glue.branch != 0 && index == GlueLinearFirstGlueInput()
                 ? nvinfer1::DataType::kINT8
                 : nvinfer1::DataType::kFLOAT;
    }
    if (kind_ == kLinear || kind_ == kGlueLinear) {
      switch (index) {
        case 0:
          return TensorType(kind_ == kLinear ? params_.linear.activation
                                             : params_.glue.x_type);
        case 1:
          return holder_granule_ != 0 ? nvinfer1::DataType::kINT64
                                      : nvinfer1::DataType::kINT8;
        case 2:
          return nvinfer1::DataType::kFLOAT;
        default:
          return params_.linear.lut != 0 ? nvinfer1::DataType::kINT8
                                         : nvinfer1::DataType::kFLOAT;
      }
    }
    if (kind_ == kGlue) {
      if (index == 0) return TensorType(params_.glue.x_type);
      return params_.glue.branch != 0 && index == 1
                 ? nvinfer1::DataType::kINT8
                 : nvinfer1::DataType::kFLOAT;
    }
    return index == 0   ? nvinfer1::DataType::kINT8
           : index == 1 ? nvinfer1::DataType::kINT32
                        : nvinfer1::DataType::kFLOAT;
  }

  int32_t OutputCount() const noexcept {
    switch (kind_) {
      case kLinear:
        return 1;
      case kGlue:
      case kGlueLinear:
      case kFactorizedMlp: {
        // kGlueLinear, kFactorizedMlp: the product's result, then x' and z of
        // the mask.
        int32_t count = kind_ == kGlue ? 0 : 1;
        for (int32_t bit = 1; bit <= kQuantGlueOutputRowScale; bit <<= 1) {
          count += (outputs_ & bit) != 0 ? 1 : 0;
        }
        return count;
      }
      default:
        return RopeGroups();
    }
  }

  // The projections of the rope kernel that have heads.
  int32_t RopeGroups() const noexcept {
    const auto& p = params_.rope;
    return (p.q_heads != 0 ? 1 : 0) + (p.k_heads != 0 ? 1 : 0) +
           (p.v_heads != 0 ? 1 : 0);
  }

  // The glue output (QuantGlueOutput) at position `index`.
  int32_t GlueOutput(int32_t index) const noexcept {
    for (int32_t bit = 1; bit <= kQuantGlueOutputRowScale; bit <<= 1) {
      if ((outputs_ & bit) != 0 && index-- == 0) return bit;
    }
    return 0;
  }

  nvinfer1::DataType OutputType(int32_t index) const noexcept {
    if (kind_ == kFactorizedMlp) {
      if (index == 0) return nvinfer1::DataType::kINT8;
      return GlueOutput(index - 1) == kQuantGlueOutputX
                 ? nvinfer1::DataType::kHALF
                 : TensorType(params_.factorized_mlp.glue_params.z_type);
    }
    if (kind_ == kGlueLinear) {
      if (index == 0) return nvinfer1::DataType::kINT8;
      return GlueOutput(index - 1) == kQuantGlueOutputX
                 ? nvinfer1::DataType::kHALF
                 : TensorType(params_.glue.z_type);
    }
    if (kind_ == kLinear) {
      return params_.linear.requant == kLiteRtNvidiaQuantRequantLogits
                 ? nvinfer1::DataType::kFLOAT
                 : nvinfer1::DataType::kINT8;
    }
    if (kind_ == kGlue) {
      switch (GlueOutput(index)) {
        case kQuantGlueOutputX:
          return nvinfer1::DataType::kHALF;
        case kQuantGlueOutputZ:
          return TensorType(params_.glue.z_type);
        case kQuantGlueOutputQ:
          return nvinfer1::DataType::kINT8;
        default:
          return nvinfer1::DataType::kFLOAT;
      }
    }
    return nvinfer1::DataType::kHALF;
  }

  // The packed weights in the weights input of `dims` at `input`: the input
  // itself, or their place in the segment a holder stands for, which starts
  // at the first granule boundary of the holder (one granule larger than the
  // segment). nullptr if the input is too small for them.
  const uint8_t* Weights(const nvinfer1::Dims& dims,
                         const void* input) const noexcept {
    const auto& p = params_.linear;
    return WeightsAt(
        dims, input,
        static_cast<uint64_t>(p.rows) * p.columns * p.bit_width / 8,
        holder_granule_ == 0 ? -1 : weight_offset_);
  }

  // `bytes` of packed weights in a weights input of `dims` at `input`: the
  // input itself where `offset` is negative, else `offset` bytes into the
  // segment its holder stands for.
  const uint8_t* WeightsAt(const nvinfer1::Dims& dims, const void* input,
                           uint64_t bytes, int64_t offset) const noexcept {
    if (dims.nbDims != 1 || dims.d[0] <= 0 || input == nullptr) {
      return nullptr;
    }
    if (offset < 0) {
      return static_cast<uint64_t>(dims.d[0]) >= bytes
                 ? static_cast<const uint8_t*>(input)
                 : nullptr;
    }
    const uint64_t granule = static_cast<uint64_t>(holder_granule_);
    const uint64_t holder_bytes =
        static_cast<uint64_t>(dims.d[0]) * sizeof(int64_t);
    if (granule == 0 || holder_bytes < granule ||
        static_cast<uint64_t>(offset) + bytes > holder_bytes - granule) {
      return nullptr;
    }
    const uintptr_t address = reinterpret_cast<uintptr_t>(input);
    const uintptr_t segment = (address + granule - 1) & ~(granule - 1);
    return reinterpret_cast<const uint8_t*>(segment) + offset;
  }

  int32_t kind_;
  Params params_;
  int32_t outputs_;  // glue, glue and product: the QuantGlueOutput mask
  int64_t weight_offset_;
  int64_t holder_granule_;
  std::array<nvinfer1::PluginField, kFields> fields_;
  nvinfer1::PluginFieldCollection field_collection_;
  void* factorized_mlp_state_ = nullptr;
};

// A holder's granule: a power of two of at least 16, so that the kernels
// read aligned words.
bool HolderGranule(int64_t granule) {
  return granule >= 16 && (granule & (granule - 1)) == 0;
}

nvinfer1::IPluginV3* CreatePlugin(int32_t kind, const Params& params,
                                  int32_t outputs, int64_t weight_offset,
                                  int64_t holder_granule) noexcept {
  if (kind == kLinear) {
    // Without a holder the weights input is the packed weights. With one,
    // the granule is a power of two and the kernels read aligned words.
    if (!LiteRtNvidiaQuantLinearSupports(&params.linear) ||
        (holder_granule == 0
             ? weight_offset != 0
             : (holder_granule < 16 ||
                (holder_granule & (holder_granule - 1)) != 0 ||
                weight_offset < 0 || weight_offset % 16 != 0))) {
      return nullptr;
    }
  } else if (kind == kGlue) {
    const auto& p = params.glue;
    const bool quantized = p.quantize != kLiteRtNvidiaQuantGlueNone;
    // An output per result the kernel has, and at least one.
    if (!LiteRtNvidiaQuantGlueSupports(&p) || outputs <= 0 ||
        outputs >= 2 * kQuantGlueOutputRowScale ||
        ((outputs & kQuantGlueOutputQ) != 0) != quantized ||
        ((outputs & kQuantGlueOutputRowScale) != 0 &&
         p.quantize != kLiteRtNvidiaQuantGlueDynamic) ||
        weight_offset != 0 || holder_granule != 0) {
      return nullptr;
    }
  } else if (kind == kGlueLinear) {
    // x' and z are the only glue outputs; q stays inside.
    if (!LiteRtNvidiaQuantGlueLinearSupports(&params.glue, &params.linear) ||
        (outputs & ~(kQuantGlueOutputX | kQuantGlueOutputZ)) != 0 ||
        (holder_granule == 0
             ? weight_offset != 0
             : (holder_granule < 16 ||
                (holder_granule & (holder_granule - 1)) != 0 ||
                weight_offset < 0 || weight_offset % 16 != 0))) {
      return nullptr;
    }
  } else if (kind == kFactorizedMlp) {
    // A weights input is the packed weights (-1) or a holder.
    bool holders = false;
    for (const int64_t offset : params.factorized_mlp.weight_offsets) {
      if (offset < -1 || (offset >= 0 && offset % 16 != 0)) return nullptr;
      holders = holders || offset >= 0;
    }
    // x' and z are the only outputs of a glue in front; q stays inside.
    const auto& p = params.factorized_mlp;
    if (!(p.glue == 0 ? LiteRtNvidiaQuantFactorizedMlpSupports(&p.stages) &&
                            outputs == 0
                      : p.glue == 1 &&
                            LiteRtNvidiaQuantGlueFactorizedMlpSupports(
                                &p.glue_params, &p.stages) &&
                            (outputs &
                             ~(kQuantGlueOutputX | kQuantGlueOutputZ)) == 0) ||
        weight_offset != 0 ||
        (holders ? !HolderGranule(holder_granule) : holder_granule != 0)) {
      return nullptr;
    }
  } else if (kind == kQkvRope) {
    LiteRtNvidiaQuantQkvRopeParams p = params.rope;
    p.tokens = 1;  // the input's at run time
    if (!LiteRtNvidiaQuantQkvRopeSupports(&p) || weight_offset != 0 ||
        holder_granule != 0) {
      return nullptr;
    }
  } else {
    return nullptr;
  }
  return new (std::nothrow)
      QuantLinearPlugin(kind, params, outputs, weight_offset, holder_granule);
}

class QuantLinearPluginCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  QuantLinearPluginCreator() noexcept
      : fields_{{{kKindField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
                 {kParamsField, nullptr, nvinfer1::PluginFieldType::kINT8, 0},
                 {kOutputsField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
                 {kWeightOffsetField, nullptr,
                  nvinfer1::PluginFieldType::kINT64, 1},
                 {kHolderGranuleField, nullptr,
                  nvinfer1::PluginFieldType::kINT64, 1}}},
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
    int32_t kind = -1;
    int32_t outputs = 0;
    int64_t weight_offset = 0;
    int64_t holder_granule = 0;
    const void* params_data = nullptr;
    int32_t params_size = 0;
    for (int32_t i = 0; i < fields->nbFields; ++i) {
      const auto& field = fields->fields[i];
      if (field.name == nullptr || field.data == nullptr) {
        continue;
      }
      if (std::strcmp(field.name, kParamsField) == 0 &&
          field.type == nvinfer1::PluginFieldType::kINT8) {
        params_data = field.data;
        params_size = field.length;
      } else if (field.length != 1) {
        continue;
      } else if (field.type == nvinfer1::PluginFieldType::kINT64) {
        const int64_t wide = *static_cast<const int64_t*>(field.data);
        if (std::strcmp(field.name, kWeightOffsetField) == 0) {
          weight_offset = wide;
        } else if (std::strcmp(field.name, kHolderGranuleField) == 0) {
          holder_granule = wide;
        }
      } else if (field.type == nvinfer1::PluginFieldType::kINT32) {
        const int32_t value = *static_cast<const int32_t*>(field.data);
        if (std::strcmp(field.name, kKindField) == 0) {
          kind = value;
        } else if (std::strcmp(field.name, kOutputsField) == 0) {
          outputs = value;
        }
      }
    }
    if (kind < kLinear || kind > kFactorizedMlp || params_data == nullptr ||
        params_size != QuantLinearPlugin::ParamsSize(kind)) {
      return nullptr;
    }
    Params params;
    std::memcpy(kind == kLinear || kind == kGlueLinear
                    ? static_cast<void*>(&params.linear)
                : kind == kGlue ? static_cast<void*>(&params.glue)
                : kind == kFactorizedMlp
                    ? static_cast<void*>(&params.factorized_mlp)
                    : static_cast<void*>(&params.rope),
                params_data, params_size);
    return CreatePlugin(kind, params, outputs, weight_offset, holder_granule);
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
  std::array<nvinfer1::PluginField, kFields> fields_;
  nvinfer1::PluginFieldCollection field_collection_;
};

}  // namespace

nvinfer1::IPluginV3* CreateQuantLinearPlugin(
    const LiteRtNvidiaQuantLinearParams& params, int64_t weight_offset,
    int64_t holder_granule) noexcept {
  Params all;
  all.linear = params;
  return CreatePlugin(kLinear, all, /*outputs=*/0, weight_offset,
                      holder_granule);
}

nvinfer1::IPluginV3* CreateQuantGluePlugin(
    const LiteRtNvidiaQuantGlueParams& params, int32_t outputs) noexcept {
  Params all;
  all.glue = params;
  return CreatePlugin(kGlue, all, outputs, /*weight_offset=*/0,
                      /*holder_granule=*/0);
}

nvinfer1::IPluginV3* CreateQuantGlueLinearPlugin(
    const LiteRtNvidiaQuantGlueParams& glue,
    const LiteRtNvidiaQuantLinearParams& linear, int32_t outputs,
    int64_t weight_offset, int64_t holder_granule) noexcept {
  Params all;
  all.linear = linear;
  all.glue = glue;
  return CreatePlugin(kGlueLinear, all, outputs, weight_offset, holder_granule);
}

nvinfer1::IPluginV3* CreateQuantFactorizedMlpPlugin(
    const LiteRtNvidiaQuantFactorizedMlpParams& params,
    const std::array<int64_t, 4>& weight_offsets, int64_t holder_granule,
    const LiteRtNvidiaQuantGlueParams* glue, int32_t outputs) noexcept {
  Params all;
  all.factorized_mlp.stages = params;
  for (int s = 0; s < kFactorizedMlpStages; ++s) {
    all.factorized_mlp.weight_offsets[s] = weight_offsets[s];
  }
  if (glue != nullptr) {
    all.factorized_mlp.glue = 1;
    all.factorized_mlp.glue_params = *glue;
  }
  return CreatePlugin(kFactorizedMlp, all, outputs, /*weight_offset=*/0,
                      holder_granule);
}

nvinfer1::IPluginV3* CreateQuantQkvRopePlugin(
    const LiteRtNvidiaQuantQkvRopeParams& params) noexcept {
  Params all;
  all.rope = params;
  return CreatePlugin(kQkvRope, all, /*outputs=*/0, /*weight_offset=*/0,
                      /*holder_granule=*/0);
}

void EnsureQuantLinearPluginRegistered() noexcept {}

REGISTER_TENSORRT_PLUGIN(QuantLinearPluginCreator);

}  // namespace litert::nvidia
