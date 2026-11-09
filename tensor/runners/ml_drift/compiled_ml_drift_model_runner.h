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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_COMPILED_ML_DRIFT_MODEL_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_COMPILED_ML_DRIFT_MODEL_RUNNER_H_

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/container/node_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/model_hints.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/profiling_info.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_conversion.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"
#include "tensor/backends/ml_drift/ml_drift_conversion.h"
#include "tensor/backends/ml_drift/ml_drift_helpers.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/graph_traversal.h"
#include "tensor/runners/ml_drift/ml_drift_dummy_weight_creator.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

namespace ml_drift_runner_internal {

// `sum_i` (per output channel, the sum of the signed quantized values over the
// input channels) of the unpacked INT4 FullyConnected weights `values` (OHWI).
inline std::vector<int32_t> ComputeWeightsSumI(
    const ml_drift::OHWI& weights_ohwi, absl::Span<const int8_t> values) {
  const int out_channels = weights_ohwi.o;
  const size_t row =
      static_cast<size_t>(weights_ohwi.h) * weights_ohwi.w * weights_ohwi.i;
  std::vector<int32_t> sum_i(out_channels, 0);
  for (int o = 0; o < out_channels; ++o) {
    int32_t s = 0;
    const int8_t* r = values.data() + static_cast<size_t>(o) * row;
    for (size_t k = 0; k < row; ++k) s += r[k];
    sum_i[o] = s;
  }
  return sum_i;
}

// INT4 weights in the kernel layout `desc` (`elements_count` elements), and
// their `sum_i` when requested.
struct RearrangedInt4Weights {
  std::vector<uint8_t> data;
  std::optional<std::vector<int32_t>> sum_i;
};

// Rearranges the dense signed INT4 weights `packed` (two per byte, OHWI
// `ohwi`) into the kernel layout `desc`. The I4O4 layouts rearrange straight
// from the packed values (and accumulate `sum_i` in the same pass), like
// `GpuModelBuilder` does for packed INT4 FullyConnected weights: one copy of
// the packed bytes instead of three full-size INT8 intermediates (~1.9 GB for
// a 262144 x 2048 tied head). Other layouts unpack to INT8 first.
inline absl::StatusOr<RearrangedInt4Weights> RearrangeInt4Weights(
    absl::Span<const int8_t> packed, const ml_drift::OHWI& ohwi,
    const ml_drift::WeightsDescription& desc, int elements_count,
    bool want_sum_i) {
  if (desc.type != ml_drift::DataType::kUint4 &&
      desc.type != ml_drift::DataType::kInt4) {
    return absl::UnimplementedError(
        "Only INT4 weight rearrangement is supported in pre-rearrange for "
        "now.");
  }
  const size_t num_elements = ohwi.DimensionsProduct();
  const size_t packed_bytes = (num_elements + 1) / 2;
  if (packed.size() < packed_bytes) {
    return absl::InvalidArgumentError(absl::StrCat(
        "INT4 weights buffer has ", packed.size(), " bytes, expected ",
        packed_bytes, " for ", num_elements, " values."));
  }
  RearrangedInt4Weights out;
  out.data.resize((static_cast<size_t>(elements_count) *
                       ml_drift::SizeInBitsOf(desc.type) +
                   7) /
                  8);
  if (desc.layout == ml_drift::WeightsLayout::kOSpatialIOGroupI4O4 ||
      desc.layout == ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kUint8> bytes;
    bytes.shape = ohwi;
    bytes.data.assign(reinterpret_cast<const uint8_t*>(packed.data()),
                      reinterpret_cast<const uint8_t*>(packed.data()) +
                          packed_bytes);
    // The converter always writes `sum_i`, 4 output channels at a time.
    std::vector<int32_t> sum_i((ohwi.o + 3) / 4 * 4, 0);
    LRT_TENSOR_RETURN_IF_ERROR(ml_drift::RearrangeWeightsUInt4Packed(
        bytes, desc, absl::MakeSpan(out.data), absl::MakeSpan(sum_i),
        /*pad_value=*/8, /*swap_dims=*/false));
    if (want_sum_i) {
      sum_i.resize(ohwi.o);
      out.sum_i = std::move(sum_i);
    }
    return out;
  }
  ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kInt8> unpacked;
  unpacked.shape = ohwi;
  unpacked.data.assign(num_elements + 64, 0);
  ml_drift::UnpackDenseInt4IntoInt8(packed.data(), num_elements,
                                    unpacked.data.data());
  if (want_sum_i) {
    out.sum_i = ComputeWeightsSumI(
        ohwi, absl::MakeConstSpan(unpacked.data).first(num_elements));
  }
  ml_drift::RearrangeWeightsInt8AsUint4(unpacked, desc,
                                        absl::MakeSpan(out.data), 8, 8u);
  return out;
}

}  // namespace ml_drift_runner_internal

template <typename Backend>
class CompiledMlDriftModelRunner {
 public:
  using Environment = typename Backend::Environment;
  using InferenceContext = typename Backend::InferenceContext;
  using BackendTensor = typename Backend::Tensor;

  // Cross-signature cache of packed constant weight descriptors and allocated
  // GPU weight tensors. Passing the same `SharedWeightsCache` across prefill
  // and decode `CompiledMlDriftModelRunner` instances sharing an `Environment`
  // avoids re-packing weights on the CPU and avoids allocating duplicate
  // constant weight buffers/textures on the GPU.
  struct SharedWeightsCache {
    absl::flat_hash_map<std::string, ml_drift::TensorDescriptor> descriptors;
    absl::flat_hash_map<std::string, std::unique_ptr<BackendTensor>> tensors;
    size_t cache_hits = 0;
    size_t cache_misses = 0;
    // When true, runners drop the host buffers of fused constant weights
    // (`+`-joined names, e.g. from `TryConcatConstantFcWeights`) once this
    // cache holds their packed descriptor. That mutates the graph passed to
    // the runner, so only set it when every graph is compiled with this cache
    // (e.g. a decoder runtime that builds a fresh graph for every pass).
    bool release_fused_host_buffers = false;
  };

  CompiledMlDriftModelRunner(std::vector<TensorHandle> flat_inputs,
                             std::vector<TensorHandle> flat_outputs,
                             ml_drift::CalculationsPrecision precision =
                                 ml_drift::CalculationsPrecision::kF32,
                             Environment* shared_env = nullptr,
                             SharedWeightsCache* shared_weights = nullptr,
                             bool allow_8bit_convs = false)
      : flat_inputs_(std::move(flat_inputs)),
        flat_outputs_(std::move(flat_outputs)),
        shared_weights_(shared_weights),
        allow_8bit_convs_(allow_8bit_convs || Allow8BitConvsFromEnv()) {
    if (shared_env != nullptr) {
      env_ptr_ = shared_env;
      env_initialized_ = true;
    }
    ABSL_CHECK_OK(EnsureEnvInitialized());

    gpu_model_info_ = {
        .precision = precision,
        .storage_type =
            Backend::GetStorageTypeForBuilder(*env_ptr_, env_initialized_),
    };
    const char* fast_tuning_env = std::getenv("LRT_FAST_TUNING");
    if (fast_tuning_env == nullptr ||
        std::string(fast_tuning_env) != "false") {
      gpu_model_info_.hints.Add(ml_drift::ModelHints::kFastTuning);
    }
    gpu_model_info_.hints.Add(ml_drift::ModelHints::kPreferTextureWeights);
    gpu_model_info_.hints.Add(ml_drift::ModelHints::kEnableHostMappedPointer);
    // `LRT_ALLOW_8BIT_CONVS=1` lets ML Drift run large quantized
    // FullyConnected / convolutions with dynamically INT8-quantized
    // activations and INT8 dot products.
    if (!allow_8bit_convs_) {
      gpu_model_info_.hints.Add(ml_drift::ModelHints::kDisallow8bitConvs);
    }
    gpu_model_info_.hints.Add(ml_drift::ModelHints::kAllowSpecialKernels);
    const char* allow_cb_env = std::getenv("LRT_ALLOW_COMMAND_BUFFER");
    gpu_model_info_.hints.allow_cl_khr_command_buffer =
        (allow_cb_env == nullptr) || (std::string(allow_cb_env) == "true");
    const ml_drift::GpuInfo gpu_info =
        Backend::GetGpuInfoForBuilder(*env_ptr_, env_initialized_);

    ABSL_LOG(INFO) << "Runner constructor: precision="
                   << static_cast<int>(precision)
                   << ", env_initialized=" << env_initialized_
                   << ", storage_type="
                   << static_cast<int>(gpu_model_info_.storage_type);
    ABSL_LOG(INFO) << "GpuInfo: vendor=" << static_cast<int>(gpu_info.vendor)
                   << ", api=" << static_cast<int>(gpu_info.gpu_api)
                   << ", wave_sizes_size="
                   << gpu_info.supported_wave_sizes.size()
                   << ", matmul_ops_size=" << gpu_info.wave_mat_mul_ops.size();
    ABSL_LOG(INFO) << "OpenClInfo: device_name="
                   << gpu_info.opencl_info.device_name << ", extensions_size="
                   << gpu_info.opencl_info.extensions.size();
    ABSL_LOG(INFO) << "Creating model builder";
    model_builder_ = std::make_unique<ml_drift::GpuModelBuilder>(
        gpu_info, gpu_model_info_.hints, gpu_model_info_.precision,
        gpu_model_info_.storage_type);
    conversion_ctx_.gpu_model_builder = model_builder_.get();
    conversion_ctx_.allow_8bit_convs = allow_8bit_convs_;
    conversion_ctx_.hints = gpu_model_info_.hints;
    conversion_ctx_.default_float_type =
        ml_drift::DeduceDataTypeFromPrecision(gpu_model_info_.precision);
    conversion_ctx_.shared_weight_descriptors =
        shared_weights_ != nullptr ? &shared_weights_->descriptors : nullptr;
  }

  void SetSharedWeightsCache(SharedWeightsCache* shared_weights) {
    shared_weights_ = shared_weights;
    conversion_ctx_.shared_weight_descriptors =
        shared_weights_ != nullptr ? &shared_weights_->descriptors : nullptr;
  }

  absl::Status Prepare(
      const absl::flat_hash_set<std::string>& external_weights = {}) {
    if (prepared_) return absl::OkStatus();
    conversion_ctx_.external_weight_names = external_weights;
    name_to_tensor_map_.clear();
    const absl::Time rearrange_start = absl::Now();
    LRT_TENSOR_RETURN_IF_ERROR(RearrangeAndMapConstantWeights());
    const absl::Time append_start = absl::Now();
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto tensor_to_handle,
        AppendGpuModelBuilder(conversion_ctx_, flat_outputs_));
    ABSL_LOG(INFO) << "LRT_COMPILE_TIMING rearrange_weights_ms="
                   << absl::ToDoubleMilliseconds(append_start - rearrange_start)
                   << " convert_ops_ms="
                   << absl::ToDoubleMilliseconds(absl::Now() - append_start);

    conversion_ctx_.graph_tensor_to_handle = std::move(tensor_to_handle);

    for (const auto& handle : flat_outputs_) {
      LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info,
                                  graph::GetInfo(handle.GetRaw()));
      if (!info.name.empty()) {
        name_to_tensor_map_[info.name] = handle.GetRaw();
      }
    }
    for (const auto& handle : flat_inputs_) {
      LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info,
                                  graph::GetInfo(handle.GetRaw()));
      if (!info.name.empty()) {
        name_to_tensor_map_[info.name] = handle.GetRaw();
      }
    }
    prepared_ = true;
    return absl::OkStatus();
  }

  absl::Status BuildModel(
      const ml_drift::ExternalTensorsInfo& external_tensors = {}) {
    const absl::Time compilation_start = absl::Now();

    LRT_TENSOR_RETURN_IF_ERROR(Prepare());

    graph_input_ids_.clear();
    for (const auto& input_handle : flat_inputs_) {
      if (conversion_ctx_.graph_tensor_to_handle.contains(
              input_handle.GetRaw())) {
        graph_input_ids_.push_back(
            conversion_ctx_.graph_tensor_to_handle.at(input_handle.GetRaw())
                .id);
      }
    }

    graph_output_ids_.clear();
    for (const auto& output_handle : flat_outputs_) {
      if (conversion_ctx_.graph_tensor_to_handle.contains(
              output_handle.GetRaw())) {
        graph_output_ids_.push_back(
            conversion_ctx_.graph_tensor_to_handle.at(output_handle.GetRaw())
                .id);
      }
    }

    LRT_TENSOR_RETURN_IF_ERROR(conversion_ctx_.gpu_model_builder->GetGpuModel(
        graph_input_ids_, graph_output_ids_, &gpu_model_));

    ml_drift::ExternalTensorsInfo combined_external_tensors = external_tensors;
    if (shared_weights_ != nullptr) {
      for (const auto& [weight_name, handle] :
           conversion_ctx_.weight_name_to_handle) {
        if (conversion_ctx_.external_weight_names.contains(weight_name)) {
          continue;
        }
        auto const_it = gpu_model_.const_tensors.find(handle.id);
        if (const_it == gpu_model_.const_tensors.end()) {
          continue;
        }
        auto tensor_it = shared_weights_->tensors.find(weight_name);
        if (tensor_it == shared_weights_->tensors.end()) {
          const ml_drift::TensorDescriptor* upload_desc = &const_it->second;
          auto shared_desc_it = shared_weights_->descriptors.find(weight_name);
          if (shared_desc_it != shared_weights_->descriptors.end() &&
              !shared_desc_it->second.GetData().empty()) {
            upload_desc = &shared_desc_it->second;
          }
          auto gpu_tensor = std::make_unique<BackendTensor>();
          LRT_TENSOR_RETURN_IF_ERROR(Backend::CreateConstTensor(
              env_ptr_, *upload_desc, gpu_tensor.get()));
          if (shared_desc_it != shared_weights_->descriptors.end()) {
            shared_desc_it->second =
                CopyDescriptorWithoutData(shared_desc_it->second);
          } else {
            shared_weights_->descriptors.emplace(
                weight_name, CopyDescriptorWithoutData(*upload_desc));
          }
          tensor_it = shared_weights_->tensors
                          .emplace(weight_name, std::move(gpu_tensor))
                          .first;
          ++shared_weights_->cache_misses;
        } else {
          ++shared_weights_->cache_hits;
        }
        gpu_model_.tensors[handle.id] =
            CopyDescriptorWithoutData(const_it->second);
        gpu_model_.const_tensors.erase(const_it);
        combined_external_tensors.immutable_tensors[handle.id] =
            tensor_it->second.get();
      }
    }

    const absl::Time init_start = absl::Now();
    LRT_TENSOR_RETURN_IF_ERROR(Backend::InitInferenceContext(
        &inference_context_, gpu_model_info_, &gpu_model_, env_ptr_,
        combined_external_tensors));

    compilation_time_ = absl::Now() - compilation_start;
    ABSL_LOG(INFO) << "LRT_COMPILE_TIMING build_model_ms="
                   << absl::ToDoubleMilliseconds(compilation_time_)
                   << " init_inference_context_ms="
                   << absl::ToDoubleMilliseconds(absl::Now() - init_start)
                   << " nodes=" << gpu_model_.nodes.size();
    return absl::OkStatus();
  }

  absl::StatusOr<ml_drift::ProfilingInfo> Profile() {
    ml_drift::ProfilingInfo profiling_info;
    LRT_TENSOR_RETURN_IF_ERROR(inference_context_.Profile(
        env_ptr_->profiling_queue(), &profiling_info));
    return profiling_info;
  }

  absl::StatusOr<ml_drift::ValueId> GetWeightValueId(
      const std::string& name) const {
    auto it = conversion_ctx_.weight_name_to_handle.find(name);
    if (it == conversion_ctx_.weight_name_to_handle.end()) {
      return absl::NotFoundError(
          absl::StrCat("Weight tensor handle not found: ", name));
    }
    return it->second.id;
  }

  absl::StatusOr<ml_drift::ValueId> GetTensorValueId(
      const std::string& name) const {
    auto raw_tensor_ptr = FindTensorByName(name);
    if (!raw_tensor_ptr) {
      return absl::NotFoundError(absl::StrCat("Tensor not found: ", name));
    }
    auto it = conversion_ctx_.graph_tensor_to_handle.find(*raw_tensor_ptr);
    if (it == conversion_ctx_.graph_tensor_to_handle.end()) {
      return absl::NotFoundError(
          absl::StrCat("Tensor handle not found: ", name));
    }
    return it->second.id;
  }

  absl::flat_hash_map<ml_drift::ValueId, BackendTensor*> GetWeightTensors() {
    absl::flat_hash_map<ml_drift::ValueId, BackendTensor*> tensors;
    for (const auto& [graph_tensor, handle] :
         conversion_ctx_.graph_tensor_to_handle) {
      auto info = graph::GetInfo(graph_tensor);
      const bool is_weight = (info.ok() && info->buffer != nullptr) ||
                             released_weight_tensors_.contains(graph_tensor);
      if (is_weight) {
        auto* t = Backend::GetTensor(&inference_context_, handle.id);
        if (t != nullptr) {
          tensors[handle.id] = t;
        }
      }
    }
    return tensors;
  }

  absl::Status SetInput(const std::string& name, TensorHandle& arg) {
    auto raw_tensor_ptr = FindTensorByName(name);
    if (!raw_tensor_ptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Input tensor not found: ", name));
    }

    auto it = conversion_ctx_.graph_tensor_to_handle.find(*raw_tensor_ptr);
    if (it == conversion_ctx_.graph_tensor_to_handle.end()) {
      return absl::OkStatus();
    }

    auto buffer_or = arg.GetBuffer();
    if (!buffer_or.ok())
      return absl::InvalidArgumentError("Tensor has no buffer");

    LRT_TENSOR_RETURN_IF_ERROR(EnsureEnvInitialized());

    const auto& th = it->second;
    if (Backend::CopyBufferToTensor(&inference_context_, env_ptr_, th.id,
                                    &buffer_or.value())
            .ok()) {
      return absl::OkStatus();
    }

    // Fallback for CPU buffers
    auto locked = buffer_or.value().Lock();
    if (arg.GetType() == Type::kFP32) {
      auto span = std::move(locked).template As<const float>();
      std::vector<float> data(span.begin(), span.end());

      ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kFloat32> t;
      t.id = th.id;
      t.data.resize(th.tensor_desc.GetBHWCShape().DimensionsProduct());
      t.shape = th.tensor_desc.GetBHWCShape();
      t.data.assign(data.begin(), data.end());
      return Backend::SetInputTensorFloat(&inference_context_, env_ptr_, t.id,
                                          t);
    } else if (arg.GetType() == Type::kI32) {
      auto span = std::move(locked).template As<const int>();
      std::vector<int> data(span.begin(), span.end());

      ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kInt32> t;
      t.id = th.id;
      t.data.resize(th.tensor_desc.GetBHWCShape().DimensionsProduct());
      t.shape = th.tensor_desc.GetBHWCShape();
      t.data.assign(data.begin(), data.end());
      return Backend::SetInputTensorInt32(&inference_context_, env_ptr_, t.id,
                                          t);
    }
    return absl::InvalidArgumentError(absl::StrCat(
        "Unsupported tensor type for SetInput fallback: ", arg.GetType()));
  }

  absl::Status Enqueue() {
    LRT_TENSOR_RETURN_IF_ERROR(EnsureEnvInitialized());
    return Backend::AddToQueue(&inference_context_, env_ptr_);
  }

  absl::Status Wait() { return Backend::WaitForCompletion(env_ptr_); }

  absl::Status Run() {
    LRT_TENSOR_RETURN_IF_ERROR(Enqueue());
    return Wait();
  }
  absl::StatusOr<std::unique_ptr<Buffer>> GetInputBuffer(
      const std::string& name) {
    auto raw_tensor_ptr = FindTensorByName(name);
    if (!raw_tensor_ptr) {
      return absl::NotFoundError(
          absl::StrCat("Input tensor not found: ", name));
    }
    auto it = conversion_ctx_.graph_tensor_to_handle.find(*raw_tensor_ptr);
    if (it == conversion_ctx_.graph_tensor_to_handle.end()) {
      return absl::NotFoundError(
          absl::StrCat("Input tensor handle not found: ", name));
    }
    const auto& th = it->second;
    return Backend::CreateInputBuffer(&inference_context_, env_ptr_, th.id);
  }

  absl::Status GetOutput(const std::string& name, Buffer& buffer) {
    auto raw_tensor_ptr = FindTensorByName(name);
    if (!raw_tensor_ptr) {
      return absl::NotFoundError(
          absl::StrCat("Output tensor not found: ", name));
    }
    auto it = conversion_ctx_.graph_tensor_to_handle.find(*raw_tensor_ptr);
    if (it == conversion_ctx_.graph_tensor_to_handle.end()) {
      return absl::NotFoundError(
          absl::StrCat("Output tensor handle not found: ", name));
    }
    const auto& th = it->second;
    return Backend::CopyTensorToBuffer(&inference_context_, env_ptr_, th.id,
                                       &buffer);
  }

  absl::StatusOr<std::vector<int32_t>> GetInt32Output(const std::string& name) {
    auto raw_tensor_ptr = FindTensorByName(name);
    if (!raw_tensor_ptr) {
      return absl::NotFoundError(
          absl::StrCat("Output tensor not found: ", name));
    }
    auto it = conversion_ctx_.graph_tensor_to_handle.find(*raw_tensor_ptr);
    if (it == conversion_ctx_.graph_tensor_to_handle.end()) {
      return absl::NotFoundError(
          absl::StrCat("Output tensor handle not found: ", name));
    }
    const auto& th = it->second;
    LRT_TENSOR_RETURN_IF_ERROR(EnsureEnvInitialized());

    ml_drift::TensorInt32 t;
    t.id = th.id;
    t.data.resize(th.tensor_desc.GetBHWCShape().DimensionsProduct());
    t.shape = th.tensor_desc.GetBHWCShape();

    LRT_TENSOR_RETURN_IF_ERROR(Backend::GetOutputTensorInt32(
        &inference_context_, env_ptr_, th.id, &t));
    return t.data;
  }

  // Reads a float output in logical BHWC order (the raw device buffer uses
  // the backend's padded storage layout, so `GetOutputBuffer` is not a
  // substitute for this).
  absl::StatusOr<std::vector<float>> GetFloatOutput(const std::string& name) {
    auto raw_tensor_ptr = FindTensorByName(name);
    if (!raw_tensor_ptr) {
      return absl::NotFoundError(
          absl::StrCat("Output tensor not found: ", name));
    }
    auto it = conversion_ctx_.graph_tensor_to_handle.find(*raw_tensor_ptr);
    if (it == conversion_ctx_.graph_tensor_to_handle.end()) {
      return absl::NotFoundError(
          absl::StrCat("Output tensor handle not found: ", name));
    }
    const auto& th = it->second;
    LRT_TENSOR_RETURN_IF_ERROR(EnsureEnvInitialized());

    ml_drift::TensorFloat32 t;
    t.id = th.id;
    t.data.resize(th.tensor_desc.GetBHWCShape().DimensionsProduct());
    t.shape = th.tensor_desc.GetBHWCShape();

    LRT_TENSOR_RETURN_IF_ERROR(Backend::GetOutputTensorFloat(
        &inference_context_, env_ptr_, th.id, &t));
    return t.data;
  }

  absl::StatusOr<std::shared_ptr<Buffer>> GetOutputBuffer(

      const std::string& name) {
    auto raw_tensor_ptr = FindTensorByName(name);
    if (!raw_tensor_ptr) {
      return absl::NotFoundError(
          absl::StrCat("Output tensor not found: ", name));
    }
    auto it = conversion_ctx_.graph_tensor_to_handle.find(*raw_tensor_ptr);
    if (it == conversion_ctx_.graph_tensor_to_handle.end()) {
      return absl::NotFoundError(
          absl::StrCat("Output tensor handle not found: ", name));
    }
    const auto& th = it->second;
    return Backend::CreateOutputBuffer(&inference_context_, env_ptr_, th.id);
  }

  absl::StatusOr<BackendTensor*> GetBackendTensor(const std::string& name) {
    auto raw_tensor_ptr = FindTensorByName(name);
    if (!raw_tensor_ptr) {
      return absl::NotFoundError(absl::StrCat("Tensor not found: ", name));
    }
    auto it = conversion_ctx_.graph_tensor_to_handle.find(*raw_tensor_ptr);
    if (it == conversion_ctx_.graph_tensor_to_handle.end()) {
      return absl::NotFoundError(
          absl::StrCat("Tensor handle not found: ", name));
    }
    const auto& th = it->second;
    BackendTensor* tensor = Backend::GetTensor(&inference_context_, th.id);
    if (tensor == nullptr) {
      return absl::InternalError(
          absl::StrCat("Backend tensor is null for: ", name));
    }
    return tensor;
  }

  // Returns the compilation time.
  absl::Duration compilation_time() const { return compilation_time_; }

 protected:
  Environment& env() { return *env_ptr_; }
  const Environment& env() const { return *env_ptr_; }

  absl::Status EnsureEnvInitialized() {
    if (env_initialized_) return absl::OkStatus();
    LRT_TENSOR_RETURN_IF_ERROR(Backend::InitEnvironment(env_ptr_));
    env_initialized_ = true;
    return absl::OkStatus();
  }

  std::optional<graph::Tensor> FindTensorByName(const std::string& name) const {
    auto it = name_to_tensor_map_.find(name);
    if (it != name_to_tensor_map_.end()) {
      return it->second;
    }
    return std::nullopt;
  }

 private:
  static std::string WeightsSumIName(absl::string_view weights_name) {
    return absl::StrCat(weights_name, ".sum_i");
  }

  // Registers `sum_i` (per output channel, the sum of the signed quantized
  // values over the input channels) of per-channel quantized INT4
  // FullyConnected weights `values` (OHWI), with which ML Drift runs long
  // prefill FullyConnected layers as INT8 convolutions on dynamically
  // INT8-quantized activations. Only when 8-bit convolutions are allowed.
  bool WantsWeightsSumI(const graph::TensorInformation& info) const {
    return allow_8bit_convs_ && info.quantization != nullptr &&
           info.quantization->As<const graph::PerChannelAffineQuantization>()
               .ok();
  }


  absl::Status AddWeightsSumI(const graph::Tensor& weights_tensor,
                              const graph::TensorInformation& info,
                              std::vector<int32_t> sums) {
    ml_drift::Tensor<ml_drift::Linear, ml_drift::DataType::kInt32> sum_i;
    sum_i.shape = ml_drift::Linear(static_cast<int>(sums.size()));
    sum_i.data = std::move(sums);
    ml_drift::TensorDescriptor desc =
        ml_drift::CreateConstantLinearTensorDescriptor(
            model_builder_->gpu_info(), sum_i);
    ml_drift::GpuModelBuilder::TensorHandle handle;
    if (!info.name.empty() &&
        conversion_ctx_.shared_weight_descriptors != nullptr) {
      auto [it, _] = conversion_ctx_.shared_weight_descriptors->emplace(
          WeightsSumIName(info.name), std::move(desc));
      handle = model_builder_->AddConstantTensor(
          CopyDescriptorWithoutData(it->second));
    } else {
      handle = model_builder_->AddConstantTensor(std::move(desc));
    }
    conversion_ctx_.weight_sum_i[weights_tensor] = handle;
    if (!info.name.empty()) {
      conversion_ctx_.weight_name_to_handle.insert(
          {WeightsSumIName(info.name), handle});
    }
    return absl::OkStatus();
  }

  // `MaybeAddWeightsSumI` for weights found in the shared weights cache.
  void ReuseSharedWeightsSumI(const graph::Tensor& weights_tensor,
                              const std::string& weights_name) {
    if (!allow_8bit_convs_) return;
    const std::string name = WeightsSumIName(weights_name);
    auto it = conversion_ctx_.shared_weight_descriptors->find(name);
    if (it == conversion_ctx_.shared_weight_descriptors->end()) return;
    auto handle = model_builder_->AddConstantTensor(
        CopyDescriptorWithoutData(it->second));
    conversion_ctx_.weight_sum_i[weights_tensor] = handle;
    conversion_ctx_.weight_name_to_handle.insert({name, handle});
  }

  // Host-side INT4 weight preparation (unpacking, rearranging into the
  // kernel's packed layout, `sum_i`), computed in parallel ahead of the
  // (serial) GPU tensor registration.
  struct PreparedInt4Weights {
    ml_drift::WeightsDescription desc;
    ml_drift::OHWI ohwi;
    int elements_count = 0;
    std::vector<uint8_t> rearranged;
    std::optional<std::vector<int32_t>> sum_i;
    absl::Status status;
  };

  absl::node_hash_map<graph::Tensor, PreparedInt4Weights> PrepareInt4Weights(
      absl::Span<const graph::Operation* const> execution_plan) {
    // `node_hash_map`: the jobs keep pointers to the entries.
    struct Job {
      graph::Tensor tensor;
      bool fc = false;
      LockedBufferSpan<const int8_t> raw;
      PreparedInt4Weights* out = nullptr;
    };
    absl::node_hash_map<graph::Tensor, PreparedInt4Weights> prepared;
    std::vector<Job> jobs;
    // The index in `jobs` of each prepared tensor, or -1 if it is skipped.
    absl::flat_hash_map<graph::Tensor, int> job_index;
    for (const graph::Operation* op : execution_plan) {
      const bool fc = op->GetName() == "FullyConnected";
      if (!fc && op->GetName() != "EmbeddingLookup") continue;
      graph::Tensor weights_tensor = op->inputs[1];
      if (auto it = job_index.find(weights_tensor); it != job_index.end()) {
        // Weights tied between an EmbeddingLookup and a FullyConnected (e.g.
        // an LM head) need `sum_i` if any FullyConnected consumer does.
        if (fc && it->second >= 0 && !jobs[it->second].fc) {
          const auto info = graph::GetInfo(weights_tensor);
          jobs[it->second].fc = info.ok() && WantsWeightsSumI(*info);
        }
        continue;
      }
      job_index[weights_tensor] = -1;
      if (conversion_ctx_.graph_tensor_to_handle.contains(weights_tensor)) {
        continue;
      }
      const auto info = graph::GetInfo(weights_tensor);
      if (!info.ok()) continue;
      if (info->type != Type::kI4 || info->buffer == nullptr ||
          conversion_ctx_.external_weight_names.contains(info->name)) {
        continue;
      }
      if (!info->name.empty() &&
          conversion_ctx_.shared_weight_descriptors != nullptr &&
          conversion_ctx_.shared_weight_descriptors->contains(info->name)) {
        continue;
      }
      auto bhwc = ShapeToBHWC(info->shape);
      if (!bhwc.ok()) continue;
      PreparedInt4Weights& p = prepared[weights_tensor];
      p.ohwi = ml_drift::OHWI(bhwc->b, bhwc->h, bhwc->w, bhwc->c);
      p.desc = model_builder_->GetFullyConnectedInt4WeightsDesc(p.ohwi);
      p.elements_count =
          ml_drift::GetTotalElementsCountForLayout(p.desc, p.ohwi);
      job_index[weights_tensor] = static_cast<int>(jobs.size());
      jobs.push_back(Job{.tensor = weights_tensor,
                         .fc = fc && WantsWeightsSumI(*info),
                         .raw = info->buffer->Lock().As<const int8_t>(),
                         .out = &p});
    }
    // Largest first, so the big tensors (e.g. the tied head) start early.
    std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
      return a.out->ohwi.DimensionsProduct() > b.out->ohwi.DimensionsProduct();
    });
    std::atomic<size_t> next{0};
    auto worker = [&] {
      for (size_t j = next.fetch_add(1, std::memory_order_relaxed);
           j < jobs.size(); j = next.fetch_add(1, std::memory_order_relaxed)) {
        Job& job = jobs[j];
        PreparedInt4Weights& p = *job.out;
        absl::StatusOr<ml_drift_runner_internal::RearrangedInt4Weights>
            rearranged = ml_drift_runner_internal::RearrangeInt4Weights(
                absl::MakeConstSpan(job.raw.data(), job.raw.size()), p.ohwi,
                p.desc, p.elements_count, /*want_sum_i=*/job.fc);
        if (!rearranged.ok()) {
          p.status = rearranged.status();
          continue;
        }
        p.rearranged = std::move(rearranged->data);
        p.sum_i = std::move(rearranged->sum_i);
      }
    };
    int num_threads = std::max(
        1, std::min<int>(
               4, static_cast<int>(std::thread::hardware_concurrency())));
    if (const char* e = std::getenv("LRT_WEIGHT_PREP_THREADS"); e != nullptr) {
      int parsed = 0;
      if (absl::SimpleAtoi(e, &parsed)) num_threads = std::max(1, parsed);
    }
    std::vector<std::thread> threads;
    for (int t = 1; t < num_threads; ++t) threads.emplace_back(worker);
    worker();
    for (std::thread& t : threads) t.join();
    return prepared;
  }

  // Registers the `sum_i` of FullyConnected weights that are already mapped to
  // a GPU tensor (by an EmbeddingLookup with the same weights): the prepared
  // `sum_i`, or the one of the shared weights cache.
  absl::Status AddSumIOfMappedWeights(
      const graph::Tensor& weights_tensor,
      absl::node_hash_map<graph::Tensor, PreparedInt4Weights>& prepared_int4) {
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info,
                                graph::GetInfo(weights_tensor));
    if (auto it = prepared_int4.find(weights_tensor);
        it != prepared_int4.end() && it->second.sum_i.has_value()) {
      std::vector<int32_t> sum_i = std::move(*it->second.sum_i);
      it->second.sum_i.reset();
      return AddWeightsSumI(weights_tensor, info, std::move(sum_i));
    }
    if (!info.name.empty() &&
        conversion_ctx_.shared_weight_descriptors != nullptr) {
      ReuseSharedWeightsSumI(weights_tensor, info.name);
    }
    return absl::OkStatus();
  }

  absl::Status RearrangeAndMapConstantWeights() {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        std::vector<const graph::Operation*> execution_plan,
        GetExecutionPlan(flat_outputs_));
    absl::node_hash_map<graph::Tensor, PreparedInt4Weights> prepared_int4 =
        PrepareInt4Weights(execution_plan);

    for (const graph::Operation* op : execution_plan) {
      if (op->GetName() == "FullyConnected" ||
          op->GetName() == "EmbeddingLookup") {
        auto weights_tensor = op->inputs[1];
        if (conversion_ctx_.graph_tensor_to_handle.contains(weights_tensor)) {
          // Weights mapped by an earlier EmbeddingLookup can still need the
          // `sum_i` of a tied FullyConnected.
          if (op->GetName() == "FullyConnected" &&
              !conversion_ctx_.weight_sum_i.contains(weights_tensor)) {
            LRT_TENSOR_RETURN_IF_ERROR(
                AddSumIOfMappedWeights(weights_tensor, prepared_int4));
          }
          continue;
        }
        LRT_TENSOR_ASSIGN_OR_RETURN(auto& info,
                                    graph::GetInfo(weights_tensor));
        if (conversion_ctx_.external_weight_names.contains(info.name)) {
          continue;
        }
        // Fused weights are always constants; their host buffer may have been
        // released by an earlier compilation of this graph (see
        // `SharedWeightsCache::release_fused_host_buffers`).
        const bool is_fused_weight = absl::StrContains(info.name, '+');
        // Reuse a packed descriptor of the shared cache before looking at the
        // host buffer, so that recompiling a graph whose fused weights were
        // released still finds them.
        if ((info.buffer != nullptr || is_fused_weight) && !info.name.empty() &&
            conversion_ctx_.shared_weight_descriptors != nullptr) {
          auto shared_it =
              conversion_ctx_.shared_weight_descriptors->find(info.name);
          if (shared_it != conversion_ctx_.shared_weight_descriptors->end()) {
            auto gpu_handle = model_builder_->AddConstantTensor(
                CopyDescriptorWithoutData(shared_it->second));
            conversion_ctx_.graph_tensor_to_handle[weights_tensor] = gpu_handle;
            name_to_tensor_map_[info.name] = weights_tensor;
            conversion_ctx_.weight_name_to_handle.insert(
                {info.name, gpu_handle});
            if (op->GetName() == "FullyConnected") {
              ReuseSharedWeightsSumI(weights_tensor, info.name);
            }
            MaybeReleaseFusedHostBuffer(weights_tensor, info);
            continue;
          }
        }
        if (info.buffer == nullptr) {
          if (is_fused_weight) {
            return absl::FailedPreconditionError(absl::StrCat(
                "The host data of fused weight '", info.name,
                "' was released after an earlier upload; compile this graph "
                "with the SharedWeightsCache that holds its packed weights."));
          }
          continue;
        }

        auto buffer = info.buffer;
        auto lock = buffer->Lock();
        auto raw_span = std::move(lock).As<const int8_t>();

        if (info.type != Type::kI4) {
          // Constant weights consumed by FullyConnected / EmbeddingLookup
          // kernels must be stored in the kernel's packed weights layout.
          // The generic constant path in `GetTensorHandle` creates a plain
          // linear tensor instead, which silently produces wrong results, so
          // pack them here the same way the ML Drift numerical test bridge
          // does. INT4 keeps its dedicated path below.
          MlDriftDummyWeightCreator weight_creator(model_builder_.get());
          Tensor<MlDriftMixinTag> weights_wrapper(weights_tensor);
          ml_drift::GpuModelBuilder::TensorHandle gpu_handle;
          ml_drift::TensorDescriptor* shared_desc_out = nullptr;
          ml_drift::TensorDescriptor local_shared_desc;
          if (!info.name.empty() &&
              conversion_ctx_.shared_weight_descriptors != nullptr) {
            shared_desc_out = &local_shared_desc;
          }
          if (info.type == Type::kFP32) {
            absl::Span<const float> data(
                reinterpret_cast<const float*>(raw_span.data()),
                raw_span.size() / sizeof(float));
            LRT_TENSOR_ASSIGN_OR_RETURN(
                gpu_handle,
                weight_creator.CreateFcWeights<float>(
                    weights_wrapper, data, conversion_ctx_.default_float_type,
                    shared_desc_out));
          } else if (info.type == Type::kI8 || info.type == Type::kI2) {
            LRT_TENSOR_ASSIGN_OR_RETURN(
                gpu_handle,
                weight_creator.CreateFcWeights<int8_t>(
                    weights_wrapper, raw_span,
                    conversion_ctx_.default_float_type, shared_desc_out));
          } else {
            return absl::UnimplementedError(absl::StrCat(
                "Unsupported constant weights type for ", op->GetName(), ": ",
                info.type, " (tensor '", info.name, "')"));
          }
          if (shared_desc_out != nullptr) {
            conversion_ctx_.shared_weight_descriptors->emplace(
                info.name, std::move(local_shared_desc));
          }
          conversion_ctx_.graph_tensor_to_handle[weights_tensor] = gpu_handle;
          name_to_tensor_map_[info.name] = weights_tensor;
          if (!info.name.empty()) {
            conversion_ctx_.weight_name_to_handle.insert(
                {info.name, gpu_handle});
          }
          MaybeReleaseFusedHostBuffer(weights_tensor, info);
          continue;
        }

        auto prepared_it = prepared_int4.find(weights_tensor);
        if (prepared_it == prepared_int4.end()) {
          return absl::InternalError(absl::StrCat("INT4 weights '", info.name,
                                                  "' were not prepared."));
        }
        PreparedInt4Weights& prepared = prepared_it->second;
        LRT_TENSOR_RETURN_IF_ERROR(prepared.status);
        const ml_drift::OHWI weights_ohwi = prepared.ohwi;
        const ml_drift::WeightsDescription desc = prepared.desc;
        const int elements_count = prepared.elements_count;
        std::vector<uint8_t> rearranged_data = std::move(prepared.rearranged);
        if (op->GetName() == "FullyConnected" && prepared.sum_i.has_value()) {
          LRT_TENSOR_RETURN_IF_ERROR(
              AddWeightsSumI(weights_tensor, info, std::move(*prepared.sum_i)));
        }

        ml_drift::TensorDescriptor weights_td;
        bool is_embedding = (op->GetName() == "EmbeddingLookup");
        bool is_supported_texture =
            desc.layout ==
            ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;

        if (desc.IsLinearLayout() || (is_embedding && !is_supported_texture)) {
          weights_td = ml_drift::TensorDescriptor(
              desc.type, ml_drift::TensorStorageType::kBuffer,
              ml_drift::Layout::kLinear);
          weights_td.SetBHWDCShape(ml_drift::BHWDC(1, 1, 1, 1, elements_count));
        } else {
          ml_drift::DataType texture_type = ml_drift::DataType::kUint32;
          if (desc.type == ml_drift::DataType::kUint4) {
            texture_type = ml_drift::DataType::kUint16;
          } else if (desc.type == ml_drift::DataType::kUint2) {
            texture_type = ml_drift::DataType::kUint8;
          }
          weights_td = ml_drift::TensorDescriptor(
              texture_type, ml_drift::TensorStorageType::kTexture2D,
              ml_drift::Layout::kHW);
          ml_drift::uint2 tex_size =
              ml_drift::Get2dResourceSize(desc, weights_ohwi);
          weights_td.SetBHWDCShape(
              ml_drift::BHWDC(1, tex_size.y, tex_size.x / 4, 1, 4));
        }

        weights_td.UploadDataRaw(absl::MakeConstSpan(rearranged_data));

        ml_drift::GpuModelBuilder::TensorHandle gpu_handle;
        if (!info.name.empty() &&
            conversion_ctx_.shared_weight_descriptors != nullptr) {
          auto [shared_it, _] =
              conversion_ctx_.shared_weight_descriptors->emplace(
                  info.name, std::move(weights_td));
          gpu_handle = model_builder_->AddConstantTensor(
              CopyDescriptorWithoutData(shared_it->second));
        } else {
          gpu_handle = model_builder_->AddConstantTensor(std::move(weights_td));
        }
        conversion_ctx_.graph_tensor_to_handle[weights_tensor] = gpu_handle;
        name_to_tensor_map_[info.name] = weights_tensor;
        if (!info.name.empty()) {
          conversion_ctx_.weight_name_to_handle.insert({info.name, gpu_handle});
        }
        MaybeReleaseFusedHostBuffer(weights_tensor, info);
      }
    }
    return absl::OkStatus();
  }

  // Drops the host copy of a fused constant weight once the shared cache holds
  // its packed descriptor, if the cache allows it
  // (`SharedWeightsCache::release_fused_host_buffers`). Released tensors stay
  // visible to `GetWeightTensors()`.
  void MaybeReleaseFusedHostBuffer(const graph::Tensor& weights_tensor,
                                   graph::TensorInformation& info) {
    if (shared_weights_ == nullptr ||
        !shared_weights_->release_fused_host_buffers ||
        !absl::StrContains(info.name, '+') ||
        !shared_weights_->descriptors.contains(info.name)) {
      return;
    }
    info.buffer = nullptr;
    released_weight_tensors_.insert(weights_tensor);
  }

 protected:
  std::vector<TensorHandle> flat_inputs_;
  std::vector<TensorHandle> flat_outputs_;

  absl::flat_hash_map<std::string, graph::Tensor> name_to_tensor_map_;
  // Fused weights whose host buffers `MaybeReleaseFusedHostBuffer` dropped.
  absl::flat_hash_set<graph::Tensor> released_weight_tensors_;

  Environment env_;
  Environment* env_ptr_ = &env_;
  SharedWeightsCache* shared_weights_ = nullptr;
  std::unique_ptr<ml_drift::GpuModelBuilder> model_builder_;
  ml_drift::GpuModel gpu_model_;
  InferenceContext inference_context_;
  MlDriftBuildContext conversion_ctx_;
  // ML Drift may use INT8-activation convolutions (constructor argument or
  // `LRT_ALLOW_8BIT_CONVS=1`).
  bool allow_8bit_convs_ = false;
  ml_drift::CreateGpuModelInfo gpu_model_info_;

  std::vector<ml_drift::ValueId> graph_input_ids_;
  std::vector<ml_drift::ValueId> graph_output_ids_;

  absl::Duration compilation_time_;
  bool env_initialized_ = false;
  bool prepared_ = false;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_COMPILED_ML_DRIFT_MODEL_RUNNER_H_
