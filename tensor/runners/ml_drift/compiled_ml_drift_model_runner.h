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
#include <cstddef>
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
#include "tensor/runners/ml_drift/compiled_ml_drift_model_runner_util.h"
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
  out.data.resize(
      (static_cast<size_t>(elements_count) * ml_drift::SizeInBitsOf(desc.type) +
       7) /
      8);
  if (desc.layout == ml_drift::WeightsLayout::kOSpatialIOGroupI4O4 ||
      desc.layout == ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kUint8> bytes;
    bytes.shape = ohwi;
    bytes.data.assign(
        reinterpret_cast<const uint8_t*>(packed.data()),
        reinterpret_cast<const uint8_t*>(packed.data()) + packed_bytes);
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

  // Host wall time of the compilation phases of `Prepare()` and
  // `BuildModel()`. The GPU weight conversion (`SetConvertWeightsOnGpu`,
  // `RunGpuWeightConversions`) runs between `build_gpu_model` and
  // `upload_shared_weights` and is in neither: it is reported in
  // `weight_pack_stats()["gpu_convert"]` and counted in `compilation_time()`.
  struct CompileTiming {
    // `RearrangeAndMapConstantWeights()`: packing constant FullyConnected /
    // EmbeddingLookup weights into kernel layouts on the CPU (see
    // `weight_pack_stats()` for the split by kind).
    absl::Duration rearrange_weights;
    // `AppendGpuModelBuilder()`: lowering the graph ops to ML Drift operations
    // (including the weight packing of ops that pack their own weights).
    absl::Duration lower_ops;
    // `GpuModelBuilder::GetGpuModel()` alone: merging (fusing) the nodes,
    // generating the kernel source code and resolving the kernel arguments
    // into the `GpuModel`. CPU only, no GPU work.
    absl::Duration build_gpu_model;
    // Creating the GPU tensors of the `SharedWeightsCache` weights that are
    // not in the cache yet (host -> GPU copies), and their bytes. Only these
    // are uploaded here: without a `SharedWeightsCache` (and for the other
    // constants) `InitFromGpuModel` uploads the constants, which is counted
    // in `init_inference_context`.
    absl::Duration upload_shared_weights;
    size_t upload_shared_weights_bytes = 0;
    // `InitFromGpuModel`: memory allocation, upload of the remaining
    // constants, kernel compilation and tuning.
    absl::Duration init_inference_context;
    size_t num_nodes = 0;
    // Nodes that convert weights to another layout on every run (e.g. when a
    // kernel needs another layout than the packed weights).
    size_t num_nodes_to_convert_weights = 0;
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

  // When true and `ml_drift::WeightsManager::IsGpuWeightsPreparationSupported`
  // (OpenCL only for now), constant INT4 / INT8 / INT2 FullyConnected and
  // EmbeddingLookup weights are uploaded as raw OHWI values and rearranged
  // into the kernel layout (and `sum_i`) by a one-shot ML Drift conversion
  // model on the GPU instead of on the CPU. Produces the same bytes as the CPU
  // path. Off by default. Set before `Prepare()`.
  void SetConvertWeightsOnGpu(bool enable) { convert_weights_on_gpu_ = enable; }
  // Tests: converts on the GPU even where ML Drift disables it (e.g. a GPU
  // info faked to select other weight layouts).
  void ForceGpuWeightConversionForTesting() {
    convert_weights_on_gpu_ = true;
    force_gpu_weight_conversion_ = true;
  }
  void SetMaxGpuWeightConversionBatchBytesForTesting(size_t bytes) {
    max_gpu_weight_conversion_batch_bytes_ = bytes;
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
    compile_timing_.rearrange_weights = absl::Now() - rearrange_start;
    const absl::Time lower_start = absl::Now();
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto tensor_to_handle,
        AppendGpuModelBuilder(conversion_ctx_, flat_outputs_));
    compile_timing_.lower_ops = absl::Now() - lower_start;

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

    const absl::Time gpu_model_start = absl::Now();
    LRT_TENSOR_RETURN_IF_ERROR(conversion_ctx_.gpu_model_builder->GetGpuModel(
        graph_input_ids_, graph_output_ids_, &gpu_model_));
    compile_timing_.build_gpu_model = absl::Now() - gpu_model_start;

    compile_timing_.num_nodes_to_convert_weights = 0;
    for (const auto& node : gpu_model_.nodes) {
      if (absl::StartsWith(node.name, "weights_convert")) {
        ++compile_timing_.num_nodes_to_convert_weights;
      }
    }

    ml_drift::ExternalTensorsInfo combined_external_tensors = external_tensors;
    LRT_TENSOR_RETURN_IF_ERROR(
        RunGpuWeightConversions(&combined_external_tensors));

    const absl::Time upload_start = absl::Now();
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
          if (upload_desc->GetData().empty() &&
              upload_desc->GetMemorySizeInBytes() > 0) {
            // Neither the model nor the cache has data for these weights and
            // the cache has no GPU tensor for them. The runners never leave
            // the cache like that (a descriptor loses its data only when its
            // GPU tensor is added, and the weights this runner converts on
            // the GPU are bound above, without publishing dataless
            // descriptors in `Prepare()`), so this only triggers when the
            // caller removed entries of `SharedWeightsCache::tensors` or
            // added `descriptors` without data. Uploading would create an
            // uninitialized tensor.
            return absl::FailedPreconditionError(
                absl::StrCat("No data for the weights '", weight_name, "'"));
          }
          auto gpu_tensor = std::make_unique<BackendTensor>();
          LRT_TENSOR_RETURN_IF_ERROR(Backend::CreateConstTensor(
              env_ptr_, *upload_desc, gpu_tensor.get()));
          compile_timing_.upload_shared_weights_bytes +=
              upload_desc->GetData().size();
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
    compile_timing_.upload_shared_weights = absl::Now() - upload_start;

    const absl::Time init_start = absl::Now();
    LRT_TENSOR_RETURN_IF_ERROR(Backend::InitInferenceContext(
        &inference_context_, gpu_model_info_, &gpu_model_, env_ptr_,
        combined_external_tensors));

    const absl::Time end = absl::Now();
    compilation_time_ = end - compilation_start;
    compile_timing_.init_inference_context = end - init_start;
    compile_timing_.num_nodes = gpu_model_.nodes.size();
    return absl::OkStatus();
  }

  const CompileTiming& compile_timing() const { return compile_timing_; }
  const absl::flat_hash_map<std::string, MlDriftBuildContext::WeightPackStat>&
  weight_pack_stats() const {
    return conversion_ctx_.weight_pack_stats;
  }

  // One-line summary of `compile_timing()` and `weight_pack_stats()`.
  std::string CompileTimingString() const {
    const CompileTiming& t = compile_timing_;
    std::string s = absl::StrCat(
        "rearrange_weights_ms=",
        absl::ToDoubleMilliseconds(t.rearrange_weights),
        " lower_ops_ms=", absl::ToDoubleMilliseconds(t.lower_ops),
        " build_gpu_model_ms=", absl::ToDoubleMilliseconds(t.build_gpu_model),
        " upload_shared_weights_ms=",
        absl::ToDoubleMilliseconds(t.upload_shared_weights),
        " upload_shared_weights_mb=",
        static_cast<double>(t.upload_shared_weights_bytes) / (1 << 20),
        " init_context_ms=",
        absl::ToDoubleMilliseconds(t.init_inference_context),
        " build_model_ms=", absl::ToDoubleMilliseconds(compilation_time_),
        " num_nodes=", t.num_nodes,
        " num_nodes_to_convert_weights=", t.num_nodes_to_convert_weights);
    std::vector<std::string> kinds;
    kinds.reserve(conversion_ctx_.weight_pack_stats.size());
    for (const auto& [kind, stat] : conversion_ctx_.weight_pack_stats) {
      kinds.push_back(kind);
    }
    std::sort(kinds.begin(), kinds.end());
    for (const std::string& kind : kinds) {
      const auto& stat = conversion_ctx_.weight_pack_stats.at(kind);
      absl::StrAppend(&s, " pack_", kind,
                      "=(ms=", absl::ToDoubleMilliseconds(stat.time),
                      ",mb=", static_cast<double>(stat.bytes) / (1 << 20),
                      ",n=", stat.count, ")");
    }
    return s;
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

  // The descriptor (without data) that the graph gives the input or output
  // tensor `name`, e.g. to check that an external tensor matches it. Valid
  // after `Prepare()`.
  absl::StatusOr<ml_drift::TensorDescriptor> GetTensorDescriptor(
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
    return CopyDescriptorWithoutData(it->second.tensor_desc);
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
    if (GpuWeightConversionEnabled()) {
      const absl::Time register_start = absl::Now();
      LRT_TENSOR_RETURN_IF_ERROR(RegisterGpuWeightConversions(execution_plan));
      const absl::Duration register_time = absl::Now() - register_start;
      if (!gpu_weight_conversions_.empty()) {
        // Bytes: the raw weights that `RunGpuWeightConversions` uploads.
        size_t raw_bytes = 0;
        for (const GpuWeightConversion& c : gpu_weight_conversions_) {
          raw_bytes += c.raw_bytes;
        }
        conversion_ctx_.AddWeightPackStat(
            "gpu_register", register_time, raw_bytes,
            static_cast<int>(gpu_weight_conversions_.size()));
      }
    }
    const absl::Time int4_start = absl::Now();
    absl::node_hash_map<graph::Tensor, PreparedInt4Weights> prepared_int4 =
        PrepareInt4Weights(execution_plan);
    const absl::Duration int4_time = absl::Now() - int4_start;
    if (!prepared_int4.empty()) {
      size_t int4_bytes = 0;
      for (const auto& [tensor, p] : prepared_int4) {
        int4_bytes += p.rearranged.size();
        if (p.sum_i.has_value()) {
          int4_bytes += p.sum_i->size() * sizeof(int32_t);
        }
      }
      conversion_ctx_.AddWeightPackStat("int4", int4_time, int4_bytes,
                                        static_cast<int>(prepared_int4.size()));
    }

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
          const absl::Time pack_start = absl::Now();
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
            // `CreateFcWeights<int8_t>` takes the raw bytes (packed for INT2),
            // unpacks INT2 to one int8 per value and rearranges it like INT8.
            // INT4 has its own path (`PrepareInt4Weights`) because INT4
            // weights include the largest tensors (e.g. a tied embedding / LM
            // head): it rearranges straight from the packed bytes, in
            // parallel, without full-size INT8 copies. INT2 could do the same
            // with ML Drift's `RearrangeWeightsUInt2Packed`.
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
          conversion_ctx_.AddWeightPackStat(
              info.type == Type::kFP32 ? "fp"
              : info.type == Type::kI8 ? "int8"
                                       : "int2",
              absl::Now() - pack_start,
              gpu_handle.tensor_desc.GetMemorySizeInBytes());
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
          weights_td.SetBHWDCShape(
              ml_drift::BHWDC(1, 1, 1, 1, static_cast<int>(elements_count)));
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

  bool GpuWeightConversionEnabled() const {
    if (!convert_weights_on_gpu_) return false;
    if (force_gpu_weight_conversion_) return true;
    const ml_drift::GpuInfo& gpu_info = model_builder_->gpu_info();
    // Only exercised on OpenCL so far.
    return gpu_info.IsApiOpenCl() &&
           ml_drift::WeightsManager::IsGpuWeightsPreparationSupported(gpu_info);
  }

  // Registers the constant INT4 / INT8 / INT2 FullyConnected / EmbeddingLookup
  // weights that the GPU can rearrange into exactly the tensors of the CPU
  // paths below (see `GetGpuWeightConversionSpec`; the others keep the CPU
  // paths). Each one gets a dataless constant tensor in the model, plus a
  // `sum_i` tensor when a FullyConnected consumer needs it, and
  // `RunGpuWeightConversions` fills them in `BuildModel()`.
  absl::Status RegisterGpuWeightConversions(
      absl::Span<const graph::Operation* const> execution_plan) {
    const ml_drift_runner_internal::FcAndEmbeddingWeights weights =
        ml_drift_runner_internal::CollectFcAndEmbeddingWeights(execution_plan);
    for (const graph::Tensor& weights_tensor : weights.ordered) {
      if (conversion_ctx_.graph_tensor_to_handle.contains(weights_tensor)) {
        continue;
      }
      absl::StatusOr<const graph::TensorInformation&> tensor_info =
          graph::GetInfo(weights_tensor);
      if (!tensor_info.ok()) continue;
      const graph::TensorInformation& info = *tensor_info;
      // External weights and weights of the shared cache aren't packed here.
      if (conversion_ctx_.external_weight_names.contains(info.name) ||
          (conversion_ctx_.shared_weight_descriptors != nullptr &&
           conversion_ctx_.shared_weight_descriptors->contains(info.name))) {
        continue;
      }
      // Like the CPU path: `sum_i` for weights with a FullyConnected consumer.
      const bool wants_sum_i =
          weights.used_by_fully_connected.contains(weights_tensor) &&
          WantsWeightsSumI(info);
      std::optional<ml_drift_runner_internal::GpuWeightConversionSpec> spec =
          ml_drift_runner_internal::GetGpuWeightConversionSpec(
              *model_builder_, info, wants_sum_i);
      if (!spec.has_value()) continue;

      // Keep the raw values alive until `RunGpuWeightConversions`.
      GpuWeightConversion c;
      c.raw = info.buffer->Lock().As<const int8_t>();
      if (c.raw.size() < spec->raw_bytes) {
        return absl::InvalidArgumentError(
            absl::StrCat("Weights '", info.name, "' have ", c.raw.size(),
                         " bytes, expected ", spec->raw_bytes));
      }
      c.weights_tensor = weights_tensor;
      c.buffer = info.buffer;
      c.name = info.name;
      c.ohwi = spec->ohwi;
      c.desc = spec->desc;
      c.src_type = spec->src_type;
      c.raw_bytes = spec->raw_bytes;

      // Map the weights to a dataless constant tensor of the model.
      c.handle = model_builder_->AddConstantTensor(std::move(spec->dst_desc));
      conversion_ctx_.graph_tensor_to_handle[weights_tensor] = c.handle;
      name_to_tensor_map_[info.name] = weights_tensor;
      conversion_ctx_.weight_name_to_handle.insert({info.name, c.handle});
      if (spec->needs_sum_i) {
        c.sum_i = model_builder_->AddConstantTensor(
            ml_drift_runner_internal::GpuWeightsSumIDescriptor(c.ohwi.o));
        conversion_ctx_.weight_sum_i[weights_tensor] = *c.sum_i;
        conversion_ctx_.weight_name_to_handle.insert(
            {WeightsSumIName(info.name), *c.sum_i});
      }
      gpu_weight_conversions_.push_back(std::move(c));
    }
    return absl::OkStatus();
  }

  // Rearranges the weights of `RegisterGpuWeightConversions` on the GPU into
  // GPU tensors that become immutable external tensors of the model (through
  // the `SharedWeightsCache` if there is one).
  //
  // Weights that another runner sharing the `SharedWeightsCache` converted in
  // the meantime (e.g. both runners ran `Prepare()` before either ran
  // `BuildModel()`) are taken from the cache instead of converted again.
  //
  // The conversions run in batches, one batch after another: a batch uploads
  // the raw weights of at most `max_gpu_weight_conversion_batch_bytes_` (or
  // of one larger tensor), converts them all together with one ML Drift
  // conversion model, and waits for it before freeing the raw GPU copies. That
  // bounds the GPU memory of the raw weights by the batch size instead of the
  // size of all the weights.
  absl::Status RunGpuWeightConversions(
      ml_drift::ExternalTensorsInfo* external_tensors) {
    if (gpu_weight_conversions_.empty()) return absl::OkStatus();
    const absl::Time start = absl::Now();
    // Split off the weights that the cache already has: they are only bound.
    std::vector<GpuWeightConversion> to_convert;
    std::vector<GpuWeightConversion> in_cache;
    for (GpuWeightConversion& c : gpu_weight_conversions_) {
      const bool cached =
          shared_weights_ != nullptr &&
          shared_weights_->tensors.contains(c.name) &&
          (!c.sum_i.has_value() ||
           shared_weights_->tensors.contains(WeightsSumIName(c.name)));
      (cached ? in_cache : to_convert).push_back(std::move(c));
    }
    gpu_weight_conversions_.clear();

    const ml_drift::GpuInfo& gpu_info = model_builder_->gpu_info();
    // The converted tensors, by main model value id.
    absl::flat_hash_map<ml_drift::ValueId, std::unique_ptr<BackendTensor>>
        dst_tensors;
    size_t converted_bytes = 0;
    // Every batch is non-empty: `GpuWeightConversionBatchEnd` > `begin`.
    for (size_t begin = 0; begin < to_convert.size();) {
      const size_t end = ml_drift_runner_internal::GpuWeightConversionBatchEnd(
          to_convert, begin, max_gpu_weight_conversion_batch_bytes_);
      const absl::Span<const GpuWeightConversion> batch =
          absl::MakeConstSpan(to_convert).subspan(begin, end - begin);
      // Allocate the converted tensors of the batch.
      for (const GpuWeightConversion& c : batch) {
        LRT_TENSOR_RETURN_IF_ERROR(
            AllocateConvertedTensor(c.handle, &dst_tensors, &converted_bytes));
        if (c.sum_i.has_value()) {
          LRT_TENSOR_RETURN_IF_ERROR(AllocateConvertedTensor(
              *c.sum_i, &dst_tensors, &converted_bytes));
        }
      }
      // Bind the raw weights (uploaded now) and the converted tensors to the
      // conversion model of the batch.
      LRT_TENSOR_ASSIGN_OR_RETURN(
          ml_drift_runner_internal::GpuWeightConversionModel conversion,
          ml_drift_runner_internal::CreateGpuWeightConversionModel(gpu_info,
                                                                   batch));
      ml_drift::ExternalTensorsInfo conversion_tensors;
      std::vector<std::unique_ptr<BackendTensor>> raw_tensors;
      for (const auto& upload : conversion.uploads) {
        LRT_TENSOR_ASSIGN_OR_RETURN(
            const ml_drift::TensorDescriptor raw_desc,
            ml_drift_runner_internal::RawWeightsDescriptor(conversion.model,
                                                           upload));
        auto tensor = std::make_unique<BackendTensor>();
        LRT_TENSOR_RETURN_IF_ERROR(
            Backend::CreateConstTensor(env_ptr_, raw_desc, tensor.get()));
        conversion_tensors.immutable_tensors[upload.input_id] = tensor.get();
        raw_tensors.push_back(std::move(tensor));
      }
      for (const auto& [conversion_id, main_id] : conversion.io_mapping) {
        auto it = dst_tensors.find(main_id);
        if (it == dst_tensors.end()) {
          return absl::InternalError("Missing converted weights tensor");
        }
        conversion_tensors.immutable_tensors[conversion_id] = it->second.get();
      }
      // Convert, and wait before `raw_tensors` frees the raw weights.
      InferenceContext conversion_context;
      LRT_TENSOR_RETURN_IF_ERROR(Backend::InitInferenceContext(
          &conversion_context, gpu_model_info_, &conversion.model, env_ptr_,
          conversion_tensors));
      LRT_TENSOR_RETURN_IF_ERROR(
          Backend::AddToQueue(&conversion_context, env_ptr_));
      LRT_TENSOR_RETURN_IF_ERROR(Backend::WaitForCompletion(env_ptr_));
      begin = end;
    }
    // Bind the converted (or cached) tensors to the model and drop the host
    // weights. Only the `to_convert` weights have tensors in `dst_tensors`;
    // the `in_cache` ones bind the tensors of the `SharedWeightsCache`.
    const auto bind = [&](GpuWeightConversion& c,
                          bool converted) -> absl::Status {
      const auto take = [&](const ml_drift::GpuModelBuilder::TensorHandle& h)
          -> std::unique_ptr<BackendTensor> {
        if (!converted) return nullptr;
        return std::move(dst_tensors.at(h.id));
      };
      LRT_TENSOR_RETURN_IF_ERROR(BindConvertedWeights(
          c.name, c.handle, take(c.handle), external_tensors));
      if (c.sum_i.has_value()) {
        LRT_TENSOR_RETURN_IF_ERROR(
            BindConvertedWeights(WeightsSumIName(c.name), *c.sum_i,
                                 take(*c.sum_i), external_tensors));
      }
      c.raw = LockedBufferSpan<const int8_t>::Empty();
      c.buffer.reset();
      if (absl::StatusOr<graph::TensorInformation&> info =
              graph::GetInfo(c.weights_tensor);
          info.ok()) {
        MaybeReleaseFusedHostBuffer(c.weights_tensor, *info);
      }
      return absl::OkStatus();
    };
    for (GpuWeightConversion& c : to_convert) {
      LRT_TENSOR_RETURN_IF_ERROR(bind(c, /*converted=*/true));
    }
    for (GpuWeightConversion& c : in_cache) {
      LRT_TENSOR_RETURN_IF_ERROR(bind(c, /*converted=*/false));
    }
    if (!to_convert.empty()) {
      conversion_ctx_.AddWeightPackStat("gpu_convert", absl::Now() - start,
                                        converted_bytes,
                                        static_cast<int>(to_convert.size()));
    }
    return absl::OkStatus();
  }

  // Allocates the GPU tensor (without data) that a conversion model writes the
  // converted weights (or `sum_i`) of the model tensor `handle` into, in
  // `dst_tensors` by main model value id, and adds its size to
  // `converted_bytes`.
  absl::Status AllocateConvertedTensor(
      const ml_drift::GpuModelBuilder::TensorHandle& handle,
      absl::flat_hash_map<ml_drift::ValueId, std::unique_ptr<BackendTensor>>*
          dst_tensors,
      size_t* converted_bytes) {
    auto tensor = std::make_unique<BackendTensor>();
    LRT_TENSOR_RETURN_IF_ERROR(Backend::CreateConstTensor(
        env_ptr_, CopyDescriptorWithoutData(handle.tensor_desc), tensor.get()));
    *converted_bytes += handle.tensor_desc.GetMemorySizeInBytes();
    (*dst_tensors)[handle.id] = std::move(tensor);
    return absl::OkStatus();
  }

  // Makes the GPU-converted weights (or `sum_i`) `name` of the model tensor
  // `handle` an immutable external tensor of the model. With a
  // `SharedWeightsCache`, the cache owns it: the tensor the cache already has
  // under `name` if any (`tensor`, if not null, is then dropped), else
  // `tensor`. Without a cache, this runner owns `tensor`.
  absl::Status BindConvertedWeights(
      const std::string& name,
      const ml_drift::GpuModelBuilder::TensorHandle& handle,
      std::unique_ptr<BackendTensor> tensor,
      ml_drift::ExternalTensorsInfo* external_tensors) {
    // The model must not upload (dataless) constant data for it.
    auto const_it = gpu_model_.const_tensors.find(handle.id);
    if (const_it != gpu_model_.const_tensors.end()) {
      gpu_model_.tensors[handle.id] =
          CopyDescriptorWithoutData(const_it->second);
      gpu_model_.const_tensors.erase(const_it);
    }
    if (shared_weights_ != nullptr) {
      if (!shared_weights_->tensors.contains(name) && tensor == nullptr) {
        return absl::InternalError(
            absl::StrCat("Weights '", name, "' were not converted"));
      }
      // One lookup: inserts `tensor` only if no other runner cached it.
      auto [it, inserted] =
          shared_weights_->tensors.try_emplace(name, std::move(tensor));
      if (inserted) {
        shared_weights_->descriptors.insert_or_assign(
            name, CopyDescriptorWithoutData(handle.tensor_desc));
        ++shared_weights_->cache_misses;
      } else {
        ++shared_weights_->cache_hits;
      }
      external_tensors->immutable_tensors[handle.id] = it->second.get();
      return absl::OkStatus();
    }
    if (tensor == nullptr) {
      return absl::InternalError(
          absl::StrCat("Weights '", name, "' were not converted"));
    }
    external_tensors->immutable_tensors[handle.id] = tensor.get();
    converted_weight_tensors_.push_back(std::move(tensor));
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

  using GpuWeightConversion = ml_drift_runner_internal::GpuWeightConversion;
  std::vector<GpuWeightConversion> gpu_weight_conversions_;
  // GPU-converted weights of runners without a `SharedWeightsCache`.
  std::vector<std::unique_ptr<BackendTensor>> converted_weight_tensors_;
  size_t max_gpu_weight_conversion_batch_bytes_ = size_t{256} << 20;
  bool convert_weights_on_gpu_ = false;
  bool force_gpu_weight_conversion_ = false;

  absl::Duration compilation_time_;
  CompileTiming compile_timing_;
  bool env_initialized_ = false;
  bool prepared_ = false;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_COMPILED_ML_DRIFT_MODEL_RUNNER_H_
