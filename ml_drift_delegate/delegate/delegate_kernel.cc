// Copyright 2025 Google LLC.
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

#include "ml_drift_delegate/delegate/delegate_kernel.h"

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/blocking_counter.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
// copybara:uncomment #include "mediapipe/framework/profiler/perfetto_profiling.h"  // from @mediapipe
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_util.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/model_hints.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/profiling_info.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "third_party/odml/infra/ml_drift_delegate/delegate_data_util.h"
#include "third_party/odml/infra/ml_drift_delegate/ml_drift_delegate.h"
#include "third_party/odml/infra/ml_drift_delegate/util.h"
// clang-format off
#include "third_party/odml/infra/ml_drift_delegate/quantization_util.h"
#include "third_party/odml/infra/ml_drift_delegate/serialization_program_cache/serialization_program_cache.h"
#include "third_party/odml/infra/ml_drift_delegate/serialization_weight_cache/serialization_weight_cache.h"
// clang-format on
#include "third_party/odml/infra/ml_drift_delegate/shared_memory_manager.h"
#include "third_party/odml/infra/ml_drift_delegate/tflite_profile.h"
#include "ml_drift_delegate/delegate/composite/custom_parsers.h"
#include "ml_drift_delegate/delegate/composite/litert_op_selector.h"
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include "ml_drift_delegate/delegate/incrementable_blocking_counter.h"
#include "ml_drift_delegate/tflite/model_builder.h"
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/shared_const_tensor_map.h"
#include "weight_loader/external_weight_loader_litert.h"
#include "tflite/c/c_api_types.h"
#include "tflite/core/c/common.h"
#include "tflite/core/subgraph.h"
#include "tflite/delegates/serialization.h"
#include "tflite/kernels/kernel_util.h"
#include "tflite/profiling/memory_latency_logger.h"  // IWYU pragma: keep

namespace litert::ml_drift {
namespace {

// When Perfetto tracing is enabled, it may be desirable to also force the
// commands to complete (CPU sync) so the time taken by each command on the GPU
// is more accurately recorded. This is not ideal for general performance, but
// it is useful for tracing.
#ifdef LITERT_GPU_FORCE_COMPLETION
constexpr bool kForceCompletion = true;
#else
constexpr bool kForceCompletion = false;
#endif

// The threshold for the total number of tensors in the shared memory
// serialization cache. If the number of tensors in the cache is smaller than
// this threshold AND gpu weight rearrangement is enabled, the serialization
// cache will not be used. Otherwise the gpu weight rearrangement is always
// preferred.
constexpr size_t kSharedMemorySerializationCacheSizeThreshold = 100;

}  // namespace

DelegateKernel::~DelegateKernel() {
  if (delegate_data_ && delegate_data_->weights_conversion_counter) {
    delegate_data_->weights_conversion_counter->Wait();
  }
}

absl::Status DelegateKernel::Dispatch(TfLiteContext* context) {
  // copybara:uncomment MEDIAPIPE_PERFETTO_TRACE_EVENT("Dispatch");

  if (delegate_data_->weights_conversion_counter) {
    delegate_data_->weights_conversion_counter->Wait();
    delegate_data_->weights_conversion_counter.reset();
  }

  if (::ml_drift::IsTfLiteProfilerActive(context)) {
    ::ml_drift::ProfilingInfo profiling_info;
    RETURN_IF_ERROR(ctx_->Profile(profiling_info));
    AddTfLiteProfilerEvents(context, &profiling_info);
  }

  RETURN_IF_ERROR(ctx_->Dispatch());
  if (kForceCompletion) {
    return backend_->WaitForCompletion();
  }

  return absl::OkStatus();
}

// Calls the delegate to get the list of tensors that need to be temporarily
// storage. It's usually needed for quantized tensors. And it's eventually
// stored in TfLiteNode::temporaries.
absl::Status DelegateKernel::GetRequiredTemporaries(
    TfLiteContext* context, TfLiteNode* node,
    TfLiteIntArray** temporaries_array_ptr) {
  if (!HasQuantizedTensors()) return absl::OkStatus();

  std::vector<int> temporary_tensors;
  temporary_tensors.reserve(input_indices_.size() + output_indices_.size());
  for (auto index : input_indices_) {
    if (quant_conversion_map_.find(index) != quant_conversion_map_.end()) {
      temporary_tensors.push_back(index);
    }
  }
  for (auto index : output_indices_) {
    if (quant_conversion_map_.find(index) != quant_conversion_map_.end()) {
      temporary_tensors.push_back(index);
    }
  }
#if defined(_WIN32)
  context->TfLiteIntArrayFree(*temporaries_array_ptr);
  *temporaries_array_ptr =
      context->TfLiteIntArrayCreate(temporary_tensors.size());
#else   // defined(_WIN32)
  TfLiteIntArrayFree(*temporaries_array_ptr);
  *temporaries_array_ptr = TfLiteIntArrayCreate(temporary_tensors.size());
#endif  // defined(_WIN32)
  for (size_t i = 0; i < temporary_tensors.size(); ++i) {
    (*temporaries_array_ptr)->data[i] = temporary_tensors[i];
  }
  return absl::OkStatus();
}

// Dequantizes the input tensors using `quant_conversion_map_`.
absl::Status DelegateKernel::DequantizeInputs(TfLiteContext* context) {
  return ::ml_drift::delegate::DequantizeInputs(context, input_indices_,
                                                quant_conversion_map_);
}

// Quantizes the output tensors using `quant_conversion_map_`.
absl::Status DelegateKernel::QuantizeOutputs(TfLiteContext* context) {
  return ::ml_drift::delegate::QuantizeOutputs(context, output_indices_,
                                               quant_conversion_map_);
}

// Returns the values of the graph that actually have a producer or
// consumers, to avoid creating unnecessary and unused tensors. For example,
// in the case of having a composite op where some inputs are needed for
// decomposition, but not for gpu, this can happen.
std::vector<::ml_drift::Value*> GetValuesUsed(
    const ::ml_drift::GraphFloat32& graph,
    const std::vector<::ml_drift::Value*>& values) {
  std::vector<::ml_drift::Value*> used_values;
  used_values.reserve(values.size());
  for (const auto& value : values) {
    auto producer = graph.FindProducer(value->id);
    auto consumers = graph.FindConsumers(value->id);
    if (producer != nullptr || !consumers.empty()) {
      used_values.push_back(value);
    }
  }
  return used_values;
}

absl::Status DelegateKernel::Initialize(
    TfLiteContext* context, const TfLiteDelegateParams* delegate_params) {
  delegate_data_ =
      reinterpret_cast<MlDriftDelegateData*>(delegate_params->delegate->data_);
  if (!delegate_data_) {
    return absl::InternalError(
        "Could not read MlDriftDelegateData from delegate data.");
  }
  if (delegate_data_->shared_backend) {
    backend_ = delegate_data_->shared_backend.get();
  } else {
    backend_ = delegate_data_->backend.get();
  }
  if (backend_->GetBackendName() == "OpenCL") {
    is_opencl_backend_ = true;
  }

  SharedConstTensorsMap shared_tensors;
  SharedConstTensorsMap* shared_tensors_ptr = nullptr;
  const TensorIndexToBufferIdMap* tensor_to_buffer_id_map = nullptr;
  const TensorIndexToExternalBufferIdMap* tensor_to_external_buffer_id_map =
      nullptr;
  TensorIndexToExternalBufferIdMap canonical_external_buffer_id_map;
  if (delegate_data_->options->enable_constant_tensors_sharing) {
    shared_tensors_ptr = &shared_tensors;
    tensor_to_buffer_id_map =
        &(reinterpret_cast<const tflite::Subgraph*>(context->impl_)
              ->GetTensorBufferIdentifiers());
    const auto& external_buffer_id_map =
        reinterpret_cast<const tflite::Subgraph*>(context->impl_)
            ->GetExternalTensorBufferIdentifiers();
    tensor_to_external_buffer_id_map = &external_buffer_id_map;
    if (delegate_data_->weight_loader != nullptr &&
        !external_buffer_id_map.empty()) {
      canonical_external_buffer_id_map.reserve(external_buffer_id_map.size());
      for (const auto& [tensor_id, external_buffer_id] :
           external_buffer_id_map) {
        canonical_external_buffer_id_map.emplace(
            tensor_id,
            delegate_data_->weight_loader->GetCanonicalExternalBufferId(
                static_cast<uint32_t>(external_buffer_id)));
      }
      tensor_to_external_buffer_id_map = &canonical_external_buffer_id_map;
    }
  }

  litert::ml_drift::ModelBuilderOptions options;
  options.enable_infinite_float_capping =
      delegate_data_->options->enable_infinite_float_capping;
  options.enable_reduced_precision = delegate_data_->calculation_precision ==
                                     ::ml_drift::CalculationsPrecision::F16;
  // Build GraphFloat32.
  ::ml_drift::GraphFloat32 graph;
  CustomOperationParserFactory custom_parser_factory;
  RETURN_IF_ERROR(BuildFinalModel(
      context, delegate_params, options, &graph, &quant_conversion_map_,
      shared_tensors_ptr, tensor_to_buffer_id_map,
      tensor_to_external_buffer_id_map, &custom_parser_factory));

  const TfLiteIntArray* input_tensors = delegate_params->input_tensors;
  const std::vector<::ml_drift::Value*> inputs =
      GetValuesUsed(graph, graph.inputs());
  std::vector<uint32_t> input_refs;
  input_refs.reserve(input_tensors->size);
  for (const ::ml_drift::Value* input : inputs) {
    const TfLiteTensor* tensor = context->tensors + input->tensor.ref;
    if (tflite::IsConstantTensor(tensor)) continue;
    input_ids_.push_back(input->id);
    input_refs.push_back(input->tensor.ref);
  }
  const TfLiteIntArray* output_tensors = delegate_params->output_tensors;
  const std::vector<::ml_drift::Value*> outputs =
      GetValuesUsed(graph, graph.outputs());
  const int output_size =
      std::min(static_cast<int>(graph.outputs().size()), output_tensors->size);
  std::vector<uint32_t> output_refs;
  output_refs.reserve(output_size);
  output_ids_.reserve(output_size);
  for (const ::ml_drift::Value* output : outputs) {
    output_ids_.push_back(output->id);
    output_refs.push_back(output->tensor.ref);
  }

  // Create inference context.
  ::ml_drift::CreateGpuModelInfo create_info;
  create_info.precision = delegate_data_->calculation_precision;
  create_info.storage_type = GetStorageType();
#ifdef __APPLE__
  create_info.hints.use_metal_argument_buffers =
      delegate_data_->options->use_metal_argument_buffers;
#endif  // __APPLE__

  create_info.hints.Add(::ml_drift::ModelHints::kAllowSpecialKernels);
  if (delegate_data_->options->prefer_texture_weights) {
    create_info.hints.Add(::ml_drift::ModelHints::kPreferTextureWeights);
  }
  if (delegate_data_->options->enable_fast_tuning) {
    create_info.hints.Add(::ml_drift::ModelHints::kFastTuning);
  }
  if (!delegate_data_->options->allow_src_quantized_fc_conv_ops) {
    create_info.hints.Add(::ml_drift::ModelHints::kDisallow8bitConvs);
  }
  if (delegate_data_->options->enable_constant_tensors_sharing) {
    RETURN_IF_ERROR(InitializeExternalSharedConstantTensors(
        context, delegate_params, shared_tensors, graph, create_info));
  }

  input_indices_.reserve(input_refs.size());
  output_indices_.reserve(output_refs.size());
  for (const int64_t input_ref : input_refs) {
    input_indices_.push_back(input_ref);
  }
  for (const int64_t output_ref : output_refs) {
    output_indices_.push_back(output_ref);
  }

  external_tensor_ids_.reserve(input_indices_.size() + output_indices_.size());
  RETURN_IF_ERROR(UpdateCreateInfoWithExternalTensors(context, inputs, outputs,
                                                      create_info));
  if (!external_tensor_ids_.empty()) {
    ABSL_LOG(INFO)
        << "Total " << external_tensor_ids_.size()
        << " external tensors are used for delegate inputs and outputs";
  }
  RETURN_IF_ERROR(
      InitInferenceContext(context, delegate_params, create_info, &graph));

  if (delegate_data_->options->enable_op_profiling_detailed_report) {
    std::cout << create_info.external_mutable_tensors.size() << std::endl;
    ::ml_drift::ProfilingInfo profiling_info;
    RETURN_IF_ERROR(ctx_->Profile(profiling_info));
    ::ml_drift::ProfilingInfo::DetailedReportOptions options;
    options.add_shapes_info = false;
    std::cout << profiling_info.GetDetailedReport(options) << std::endl;
    ASSIGN_OR_RETURN(auto runtime_mem_bytes,
                     ctx_->GetSizeOfMemoryAllocatedForIntermediateTensors());
    std::cout << "Memory for intermediate tensors - "
              << runtime_mem_bytes / 1024.0 / 1024.0 << " MB" << std::endl;
    ASSIGN_OR_RETURN(auto const_mem_bytes,
                     ctx_->GetSizeOfMemoryAllocatedForConstantTensors());
    std::cout << "Memory for constant tensors - "
              << const_mem_bytes / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "Total tensors memory(const + intermediate) - "
              << (const_mem_bytes + runtime_mem_bytes) / 1024.0 / 1024.0
              << " MB" << std::endl;
    ASSIGN_OR_RETURN(auto external_bytes,
                     ctx_->GetSizeOfMemoryAllocatedForExternalTensors());
    std::cout << "Memory for external tensors - "
              << external_bytes / 1024.0 / 1024.0 << " MB" << std::endl;
  }

  ctx_->ReportMemoryBenchmarkIfEnabled(create_info).IgnoreError();
  return absl::OkStatus();
}

absl::Status DelegateKernel::InitializeExternalSharedConstantTensors(
    TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
    const SharedConstTensorsMap& shared_tensors,
    ::ml_drift::GraphFloat32& graph,
    ::ml_drift::CreateGpuModelInfo& create_info) {
#ifdef ML_DRIFT_MEM_STATS
  tflite::profiling::memory::MemoryLatencyLogger logger;
  logger.Start();
#endif

  ::ml_drift::SerializationWeightCache* shared_memory_serialization_cache;
#ifdef __EMSCRIPTEN__
  bool prepare_weights_in_batches = false;
#else
  bool prepare_weights_in_batches = true;
#endif
  ASSIGN_OR_RETURN(shared_memory_serialization_cache,
                   TryInitializingExternalTensorsSerialization(
                       context, delegate_params, prepare_weights_in_batches));
  size_t shared_memory_serialization_cache_size =
      (shared_memory_serialization_cache
           ? shared_memory_serialization_cache->GetCurrentSize()
           : 0);

  ASSIGN_OR_RETURN(auto shared_mem_manager,
                   backend_->CreateSharedMemoryManager(
                       create_info, graph, context, *delegate_data_,
                       shared_memory_serialization_cache));

  if (delegate_data_->options->convert_weights_on_gpu &&
      delegate_data_->options->enable_constant_tensors_sharing) {
    ASSIGN_OR_RETURN(auto gpu_info, backend_->GetInfo());
    if (!::ml_drift::WeightsManager::IsGpuWeightsPreparationSupported(
            gpu_info) ||
        shared_memory_serialization_cache_size >
            kSharedMemorySerializationCacheSizeThreshold) {
      delegate_data_->options->convert_weights_on_gpu = false;
    } else {
      ASSIGN_OR_RETURN(auto weights_manager, backend_->CreateWeightsManager());
      shared_mem_manager->SetWeightsManager(weights_manager);
    }
  }

  // Iterate over shared tensors in sorted order to ensure that the order of
  // the external tensor allocations are consistent so their storage types
  // are consistent across runs. Allocate the largest tensors first to reduce
  // the peak memory usage.
  std::vector<::ml_drift::ValueId> shared_tensor_ids_ordered_by_size;
  for (const auto& [shared_tensor_id, _] : shared_tensors) {
    shared_tensor_ids_ordered_by_size.push_back(shared_tensor_id);
  }
  auto get_tensor = [&](::ml_drift::ValueId shared_tensor_id) -> TfLiteTensor& {
    return context->tensors[shared_tensors.find(shared_tensor_id)
                                ->second.tflite_tensor_id];
  };
  std::sort(shared_tensor_ids_ordered_by_size.begin(),
            shared_tensor_ids_ordered_by_size.end(),
            [&](::ml_drift::ValueId a, ::ml_drift::ValueId b) {
              // Sort by size first.
              if (get_tensor(a).bytes < get_tensor(b).bytes) {
                return true;
              }
              if (get_tensor(a).bytes > get_tensor(b).bytes) {
                return false;
              }
              // If the sizes are the same, sort by id to ensure a consistent
              // ordering.
              return a < b;
            });
  // The map contains the weight tensors from the current DelegateKernel
  // instance that will be registered and shared with other DelegateKernel
  // instances, in the form of ML Drift's GpuSpatialTensor, via the
  // SharedMemoryManager instance.
  //  * Key: the tensor's ID in the GraphFloat32 of the current DelegateKernel
  //  instance and it's only valid for the current DelegateKernel instance.
  //  * Value: the tensor's global ID that is valid for all DelegateKernel
  //  instances of the TFLite model, and it will be used to index the tensor's
  //  GpuSpatialTensor object.
  absl::flat_hash_map<::ml_drift::ValueId,
                      ::ml_drift::SharedMemoryManager::GlobalId>
      local_to_global_id_map;
  // Register the constant weight tensors to be shared with other
  // DelegateKernel instances.
  for (auto it = shared_tensor_ids_ordered_by_size.rbegin();
       it != shared_tensor_ids_ordered_by_size.rend(); ++it) {
    ::ml_drift::ValueId shared_tensor_id = *it;
    auto& shared_tflite_tensor = shared_tensors.find(shared_tensor_id)->second;
    RETURN_IF_ERROR(shared_mem_manager->RegisterExternalConstantTensors(
        shared_tensor_id, shared_tflite_tensor, local_to_global_id_map));
  }
  // If GPU weights conversion is enabled, trigger the GPU conversion to produce
  // GPU tensors for weights.
if (delegate_data_->options->convert_weights_on_gpu &&
      delegate_data_->options->enable_constant_tensors_sharing) {
    ASSIGN_OR_RETURN(auto gpu_info, backend_->GetInfo());
    auto& buffer_map = GetBufferIdToSpatialTensorMap(*delegate_data_);
    auto& quant_map = GetQuantParamIdToSpatialTensorMap(*delegate_data_);
    // TODO: b/403337563 - Enable prepare_weights_in_batches with options.
    if ((gpu_info.IsApple() || gpu_info.IsApiWebGpu()) &&
        prepare_weights_in_batches) {
      bool use_serialization_cache = shared_memory_serialization_cache;
      // On Apple devices, reading weights from clean, file-backed, memory
      // mmapped pages is strongly preferred and it is worth forcing the first
      // load to use the serialization cache.
      bool require_serialization_cache_on_first_load =
          use_serialization_cache && gpu_info.IsApple();
      absl::flat_hash_set<::ml_drift::ValueId> prepared_tensor_ids;
      ASSIGN_OR_RETURN(auto batches,
                       backend_->GetBatchesForWeightsPreparation(
                           shared_mem_manager->GetWeightsManager()));
      for (auto& batch : batches) {
        ASSIGN_OR_RETURN(auto tensor_map_for_batch,
                         backend_->PrepareWeightsInBatch(
                             shared_mem_manager->GetWeightsManager(), batch));
        for (auto& [main_model_id, tensor] : tensor_map_for_batch) {
          ::ml_drift::SharedMemoryManager::GlobalId global_id =
              local_to_global_id_map[main_model_id];
          if (use_serialization_cache) {
            // Download from GPU to CPU memory.
            ::ml_drift::TensorDescriptor descriptor = tensor->GetDescriptor();
            RETURN_IF_ERROR(
                backend_->ReadSpatialTensorToDescriptor(*tensor, descriptor));
            // Insert the descriptor to the cache.
            RETURN_IF_ERROR(shared_memory_serialization_cache->Insert(
                global_id.value, !global_id.IsSourceId(), descriptor));
            // Release the tensor memory.
            if (require_serialization_cache_on_first_load) {
              RETURN_IF_ERROR(
                  backend_->ReleaseSpatialTensorMemory(tensor.get()));
            }
          }
          if (global_id.IsSourceId()) {
            buffer_map[global_id.value].weights = std::move(tensor);
          } else {
            quant_map[global_id.value].weights = std::move(tensor);
          }
          prepared_tensor_ids.insert(main_model_id);
        }
        for (auto& op_info : batch) {
          if (delegate_data_->options->madvise_original_shared_tensors) {
            ::ml_drift::MadviseData(const_cast<void*>(op_info.data_ptr),
                                    op_info.size);
          }
        }
      }

      if (require_serialization_cache_on_first_load) {
        // Flush the cache to disk.
        RETURN_IF_ERROR(CleanupExternalTensorsSerialization(
            shared_memory_serialization_cache));

        // Load the cache from disk.
        ASSIGN_OR_RETURN(
            shared_memory_serialization_cache,
            TryInitializingExternalTensorsSerialization(
                context, delegate_params, prepare_weights_in_batches));

        for (const auto& main_model_id : prepared_tensor_ids) {
          ::ml_drift::SharedMemoryManager::GlobalId global_id =
              local_to_global_id_map[main_model_id];

          // Read the descriptor from the cache.
          ::ml_drift::TensorDescriptor descriptor;
          ml_drift_delegate::UnownedDataTensorDescriptor
              unowned_data_tensor_desc;
          size_t page_adjusted_offset;
          ml_drift_delegate::ReleaseDataCallback release_data_callback;
          RETURN_IF_ERROR(shared_memory_serialization_cache->LookUp(
              global_id.value, global_id.IsParamId(), unowned_data_tensor_desc,
              page_adjusted_offset, release_data_callback));
          ::ml_drift::GpuSpatialTensor* spatial_tensor = nullptr;
          if (global_id.IsSourceId()) {
            spatial_tensor = buffer_map[global_id.value].GetWeights();
          } else {
            spatial_tensor = quant_map[global_id.value].GetWeights();
          }

          // Update tensor from cached descriptor.
          RETURN_IF_ERROR(backend_->UpdateSpatialTensor(
              spatial_tensor, unowned_data_tensor_desc, page_adjusted_offset,
              std::move(release_data_callback)));
        }
      }
      // Flush the cache to disk.
      RETURN_IF_ERROR(CleanupExternalTensorsSerialization(
          shared_memory_serialization_cache));
    } else {
      ::ml_drift::GpuModel gpu_weights_conversion_model;
      absl::flat_hash_map<::ml_drift::ValueId, ::ml_drift::ValueId> io_mapping;
      std::vector<::ml_drift::WeightsManager::UploadWeightsInfo>
          upload_weights_infos;
      RETURN_IF_ERROR(
          shared_mem_manager->GetWeightsManager()->CreateConversionGpuModel(
              gpu_info, &gpu_weights_conversion_model, &io_mapping,
              &upload_weights_infos));
      if (!upload_weights_infos.empty()) {
        ASSIGN_OR_RETURN(conversion_context_,
                         backend_->CreateInferenceContext(
                             create_info, gpu_weights_conversion_model));
        // The preferred number is picked based on the performance of Gemma3N
        // 4B running on Linux + Nvidia GTX 4090 with WebGPU backend. We
        // observed that the peak memory usage keeps dropping, as the
        // preferred number being reduced from 256, until it reaches the below
        // value.
        const int kPreferredNumNodesPerCommandEncoderForWeightsPreparation = 8;
        // Not all backends support customizing num_nodes_per_command_encoder
        // yet.
        conversion_context_
            ->SetCommandBufferHint(
                kPreferredNumNodesPerCommandEncoderForWeightsPreparation)
            .IgnoreError();
        if (delegate_data_->upload_executor) {
          weights_converting_ = std::make_unique<absl::BlockingCounter>(
              upload_weights_infos.size());
          if (delegate_data_->weights_conversion_counter) {
            delegate_data_->weights_conversion_counter->Increment();
          } else {
            delegate_data_->weights_conversion_counter =
                std::make_unique<IncrementableBlockingCounter>(1);
          }
        }
#ifdef __EMSCRIPTEN__
        RETURN_IF_ERROR(conversion_context_->UploadWeightsOnWeb(
            delegate_data_->weight_loader, gpu_weights_conversion_model,
            io_mapping, shared_mem_manager->GetWeightIdToExternalBufferIdMap(),
            upload_weights_infos));
#else
        for (const auto& upload_info : upload_weights_infos) {
          auto upload_fn = [this, upload_info]() -> absl::Status {
            RETURN_IF_ERROR(conversion_context_->WriteDataToWeightTensor(
                upload_info.input_id,
                absl::MakeConstSpan(
                    static_cast<const uint8_t*>(upload_info.data),
                    upload_info.size)));
            if (delegate_data_->options->madvise_original_shared_tensors) {
              ::ml_drift::MadviseData(const_cast<void*>(upload_info.data),
                                      upload_info.size);
            }
            return absl::OkStatus();
          };
          if (delegate_data_->upload_executor) {
            delegate_data_->upload_executor->Schedule(
                [this, upload_fn = std::move(upload_fn)]() {
                  if (auto s = upload_fn(); !s.ok()) {
                    ABSL_LOG(ERROR) << "Failed to upload weights: " << s;
                  }
                  weights_converting_->DecrementCount();
                });
          } else {
            RETURN_IF_ERROR(upload_fn());
          }
        }
#endif  // __EMSCRIPTEN__
        if (delegate_data_->upload_executor) {
          delegate_data_->upload_executor->Schedule([this]() {
            weights_converting_->Wait();
            if (auto s = conversion_context_->Dispatch(); !s.ok()) {
              ABSL_LOG(ERROR) << "Failed to dispatch weights conversion: " << s;
            }
            delegate_data_->weights_conversion_counter->Decrement();
          });
        } else {
          RETURN_IF_ERROR(conversion_context_->Dispatch());
        }
        RETURN_IF_ERROR(backend_->WaitForCompletion());
        if (delegate_data_->options->wait_for_weights_conversion_complete &&
            delegate_data_->weights_conversion_counter) {
          delegate_data_->weights_conversion_counter->Wait();
          delegate_data_->weights_conversion_counter.reset();
        }
        for (const auto& [converted_weight_id, main_model_weight_id] :
             io_mapping) {
          ASSIGN_OR_RETURN(auto* tensor, conversion_context_->GetSpatialTensor(
                                             converted_weight_id));
          ::ml_drift::SharedMemoryManager::GlobalId global_id =
              local_to_global_id_map[main_model_weight_id];
          if (global_id.IsSourceId()) {
            buffer_map[global_id.value].external_weights = tensor;
          } else {
            quant_map[global_id.value].external_weights = tensor;
          }
        }
      }
    }
  }
  // Register the shared tensors as the external immutable tensors for the
  // current DelegateKernel instance's execution with ML Drift.
  for (auto& [local_id, global_id] : local_to_global_id_map) {
    ASSIGN_OR_RETURN(::ml_drift::GpuSpatialTensor * tensor,
                     shared_mem_manager->GetExternalConstantTensor(global_id));
    create_info.external_immutable_tensors.try_emplace(local_id, tensor);
  }

  external_shared_constant_tensor_ids_.clear();
  external_shared_constant_tensor_ids_.insert(
      shared_tensor_ids_ordered_by_size.begin(),
      shared_tensor_ids_ordered_by_size.end());

  RETURN_IF_ERROR(
      CleanupExternalTensorsSerialization(shared_memory_serialization_cache));

#ifdef ML_DRIFT_MEM_STATS
  logger.Stop("Loading: created external tensors");
#endif  // ML_DRIFT_MEM_STATS

  return absl::OkStatus();
}

// Returns false if serialization prerequisites are not initialized. Otherwise
// initializes the serialization params and returns true.
bool DelegateKernel::ReadFromSerialzedData() {
  bool has_valid_fd_with_model_token =
      delegate_data_->options->program_cache_fd > 0 &&
      !delegate_data_->model_token.empty();
  bool has_valid_path_with_model_token =
      !delegate_data_->serialization_dir.empty() &&
      !delegate_data_->model_token.empty();

  if ((!has_valid_fd_with_model_token && !has_valid_path_with_model_token) ||
      !delegate_data_->options->serialize_program_cache ||
      // Compiled programs are cached by backend directly.
      delegate_data_->options->cache_compiled_programs_only) {
    return false;
  }
  if (!serialization_) {
    tflite::delegates::SerializationParams params;
    params.model_token = delegate_data_->model_token.c_str();
    params.cache_dir = delegate_data_->serialization_dir.c_str();
    serialization_ = std::make_unique<tflite::delegates::Serialization>(params);
  }
  return true;
}

// Restores inference context from the serialized data.
absl::Status DelegateKernel::InitInferenceContextFromSerializedData(
    TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
    const ::ml_drift::CreateGpuModelInfo& create_info,
    ::ml_drift::GraphFloat32* graph, LiteRtOpSelector& op_selector) {
  // Generate fingerprints for the relevant delegate data options.
  struct {
    MlDriftDelegatePrecision precision;
    bool convert_weights_on_gpu;
  } options_to_fingerprint = {};
  options_to_fingerprint.precision = delegate_data_->options->precision;
  options_to_fingerprint.convert_weights_on_gpu =
      delegate_data_->options->convert_weights_on_gpu;
  std::string options_fingerprint = tflite::delegates::StrFingerprint(
      reinterpret_cast<const char*>(&options_to_fingerprint),
      sizeof(options_to_fingerprint));

  std::unique_ptr<tflite::delegates::SerializationEntry> data_key;
  std::unique_ptr<::ml_drift::SerializationProgramCache> program_cache;
  uint64_t fingerprint_key;
  std::string model_data;
  if (delegate_data_->options->program_cache_fd > 0) {
    // Duplicate the fd since the program cache will take ownership of the fd.
    // The original fd is owned by the delegate options and may be used after
    // this function returns.
    int dup_fd = dup(delegate_data_->options->program_cache_fd);
    if (dup_fd < 0) {
      return absl::InternalError("Failed to duplicate program cache fd");
    }
    program_cache =
        std::make_unique<::ml_drift::SerializationProgramCache>(dup_fd);
    fingerprint_key = tflite::delegates::Serialization::GetFingerprint(
        delegate_data_->model_token, options_fingerprint, context,
        delegate_params);
    auto program_data = program_cache->LookUp(fingerprint_key);
    if (program_data.ok()) {
      model_data = program_data.value();
    }
  } else {
    program_cache = std::make_unique<::ml_drift::SerializationProgramCache>(
        delegate_data_->serialization_dir, delegate_data_->model_token);
    fingerprint_key = tflite::delegates::Serialization::GetFingerprint(
        delegate_data_->model_token, options_fingerprint, context,
        delegate_params);
    auto program_data = program_cache->LookUp(fingerprint_key);
    if (program_data.ok()) {
      model_data = program_data.value();
    }
  }

  if (!model_data.empty()) {
    // Restore InferenceContext from serialized data.
    absl::Span<const uint8_t> model_span = absl::Span<const uint8_t>{
        reinterpret_cast<const uint8_t*>(model_data.data()), model_data.size()};

    // If convert_weights_on_gpu is enabled (prepare weights on GPU),
    // trigger weights preparation and register the prepared weights into
    // create_info before restoring the inference context.
    ::ml_drift::CreateGpuModelInfo create_info_main = create_info;
    if (delegate_data_->options->convert_weights_on_gpu &&
        !delegate_data_->options->enable_constant_tensors_sharing) {
      ::ml_drift::GpuModel gpu_model;
      RETURN_IF_ERROR(GraphToGpuModelWithGpuConverters(
          *graph, create_info_main, &gpu_model, op_selector));
    }

    auto res = backend_->RestoreInferenceContext(create_info_main, model_span);
    if (res.ok()) {
      ctx_ = std::move(res.value());
      ABSL_LOG(INFO) << "Initialized InferenceContext from serialized data.";
      return absl::OkStatus();
    }
    ABSL_LOG(WARNING) << "Deserialization failed: " << res.status();
    // If the restore fails, fallback to the default graph initialization.
  }

  //  Init InferenceContext and serialize it.
  std::vector<uint8_t> serialized_model;
  RETURN_IF_ERROR(InitInferenceContextFromGraph(context, delegate_params,
                                                create_info, graph, op_selector,
                                                &serialized_model));
  auto save_status = program_cache->Insert(
      fingerprint_key,
      absl::string_view(reinterpret_cast<const char*>(serialized_model.data()),
                        serialized_model.size()));
  if (!save_status.ok()) {
    ABSL_LOG(WARNING) << "Failed to save serialized data: " << save_status;
  }
  return absl::OkStatus();
}

absl::Status DelegateKernel::GraphToGpuModelWithGpuConverters(
    const ::ml_drift::GraphFloat32& graph,
    ::ml_drift::CreateGpuModelInfo& create_info,
    ::ml_drift::GpuModel* gpu_model, LiteRtOpSelector& op_selector) {
  ASSIGN_OR_RETURN(auto gpu_info, backend_->GetInfo());
  ::ml_drift::CreateGpuModelInfo create_info_for_conversion = create_info;
  ::ml_drift::GpuModel gpu_weights_conversion_model;
  absl::flat_hash_map<::ml_drift::ValueId, ::ml_drift::ValueId> weights_mapping;
  std::vector<::ml_drift::WeightsManager::UploadWeightsInfo>
      upload_weights_info;

  RETURN_IF_ERROR(::ml_drift::GraphToGpuModelWithWeightsConversion(
      graph, create_info_for_conversion, gpu_info, gpu_model,
      &gpu_weights_conversion_model, &weights_mapping, &upload_weights_info,
      &op_selector));
  ASSIGN_OR_RETURN(conversion_context_, backend_->CreateInferenceContext(
                                            create_info_for_conversion,
                                            gpu_weights_conversion_model));

  // this can be done in a separate thread. Need to sync then with
  // main.enqueue. Upload data should be alive until completion of all
  // WriteData calls if we have async=true.
  for (const auto& upload_info : upload_weights_info) {
    RETURN_IF_ERROR(conversion_context_->WriteDataToWeightTensor(
        upload_info.input_id,
        absl::MakeConstSpan(reinterpret_cast<const uint8_t*>(upload_info.data),
                            upload_info.size)));
  }

  // this can be called later(but before ctx_ enqueue calls) and without
  // explicit wait(if the same queue used for ctx_ enqueue calls).
  RETURN_IF_ERROR(conversion_context_->Dispatch());
  RETURN_IF_ERROR(backend_->WaitForCompletion());

  for (const auto& [converted_weight_id, main_model_weight_id] :
       weights_mapping) {
    ASSIGN_OR_RETURN(auto* tensor, conversion_context_->GetSpatialTensor(
                                       converted_weight_id));
    create_info.external_immutable_tensors.insert(
        {main_model_weight_id, tensor});
  }
  return absl::OkStatus();
}

// Builds inference context from the graph.
absl::Status DelegateKernel::InitInferenceContextFromGraph(
    TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
    const ::ml_drift::CreateGpuModelInfo& create_info,
    ::ml_drift::GraphFloat32* graph, LiteRtOpSelector& op_selector,
    std::vector<uint8_t>* serialized_model) {
  ::ml_drift::CreateGpuModelInfo create_info_main = create_info;
  ASSIGN_OR_RETURN(auto gpu_info, backend_->GetInfo());
  ::ml_drift::GpuModel gpu_model;
  if (delegate_data_->options->convert_weights_on_gpu &&
      !delegate_data_->options->enable_constant_tensors_sharing &&
      ::ml_drift::WeightsManager::IsGpuWeightsPreparationSupported(gpu_info)) {
    RETURN_IF_ERROR(GraphToGpuModelWithGpuConverters(*graph, create_info_main,
                                                     &gpu_model, op_selector));
  } else {
    RETURN_IF_ERROR(::ml_drift::GraphToGpuModel(*graph, create_info, gpu_info,
                                                &gpu_model, &op_selector));
  }
  ASSIGN_OR_RETURN(ctx_, backend_->CreateInferenceContext(
                             create_info_main, gpu_model, serialized_model,
                             /*may_share_memory_manager=*/true));
  return absl::OkStatus();
}

absl::Status DelegateKernel::InitInferenceContext(
    TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
    const ::ml_drift::CreateGpuModelInfo& create_info,
    ::ml_drift::GraphFloat32* graph) {
#ifdef ML_DRIFT_MEM_STATS
  tflite::profiling::memory::MemoryLatencyLogger logger;
  logger.Start();
#endif  // ML_DRIFT_MEM_STATS

  ASSIGN_OR_RETURN(auto gpu_info, backend_->GetInfo());
  LiteRtOpSelector op_selector(&create_info, &gpu_info);
  if (ReadFromSerialzedData()) {
    ABSL_LOG(INFO) << "Initializing " << backend_->GetBackendName()
                   << "-based API from serialized data.";
    RETURN_IF_ERROR(InitInferenceContextFromSerializedData(
        context, delegate_params, create_info, graph, op_selector));
  } else {
    ABSL_LOG(INFO) << "Initializing " << backend_->GetBackendName()
                   << "-based API from graph.";
    RETURN_IF_ERROR(InitInferenceContextFromGraph(
        context, delegate_params, create_info, graph, op_selector,
        /*serialized_model=*/nullptr));
  }

#ifdef ML_DRIFT_MEM_STATS
  logger.Stop("Loading: inference context init");
#endif
  return absl::OkStatus();
}

// Creates a new SerializationWeightCache and returns it. If the serialization
// is disabled, it will return a nullptr. If there is an error, it will return
// the error status.
absl::StatusOr<::ml_drift::SerializationWeightCache*>
DelegateKernel::TryInitializingExternalTensorsSerialization(
    TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
    bool prepare_weights_in_batches) {
  bool has_valid_fd = delegate_data_->options->weight_cache_fd > 0;
  bool has_valid_path = !delegate_data_->model_token.empty() &&
                        !delegate_data_->serialization_dir.empty();

  if ((!has_valid_fd && !has_valid_path) ||
      !delegate_data_->options->serialize_external_tensors) {
    return nullptr;
  }

  if (!delegate_data_->serialization_cache) {
    // Create the cache.
    delegate_data_->serialization_cache =
        std::make_unique<::ml_drift::SerializationWeightCache>();
  }

  // Create a unique identifier for this subgraph.
  uint64_t unique_model_identifier =
      ::ml_drift::SerializationWeightCache::GenerateUniqueModelIdentifier(
          delegate_data_->model_token, context, delegate_params,
          backend_->GetSerializedDataPrefix(),
          delegate_data_->options->precision,
          delegate_data_->options->prefer_texture_weights,
          delegate_data_->options->allow_src_quantized_fc_conv_ops,
          prepare_weights_in_batches,
          delegate_data_->options->serialize_external_tensors,
          /*ordered_by_size=*/true);

  // Try loading from the cache.
  absl::Status load_status;
  if (has_valid_fd) {
    // Duplicate the fd since the serialization cache will take ownership of the
    // fd. The original fd is owned by the delegate options and may be used
    // after this function returns.
    int dup_fd = dup(delegate_data_->options->weight_cache_fd);
    if (dup_fd < 0) {
      return absl::InternalError("Failed to duplicate weight cache fd");
    }
    load_status = delegate_data_->serialization_cache->Load(
        dup_fd, unique_model_identifier);
  } else {
    load_status = delegate_data_->serialization_cache->Load(
        delegate_data_->serialization_dir, delegate_data_->model_token,
        unique_model_identifier);
  }

  if (!load_status.ok() &&
      load_status.code() != absl::StatusCode::kUnimplemented) {
    if (has_valid_fd) {
      // Duplicate the fd since the serialization cache will take ownership of
      // the fd. The original fd is owned by the delegate options and may be
      // used after this function returns.
      int dup_fd = dup(delegate_data_->options->weight_cache_fd);
      if (dup_fd < 0) {
        return absl::InternalError("Failed to duplicate weight cache fd");
      }
      RETURN_IF_ERROR(delegate_data_->serialization_cache->StartBuild(
          dup_fd, unique_model_identifier));
    } else {
      RETURN_IF_ERROR(delegate_data_->serialization_cache->StartBuild(
          delegate_data_->serialization_dir, delegate_data_->model_token,
          unique_model_identifier));
    }
  }
  return delegate_data_->serialization_cache.get();
}

absl::Status DelegateKernel::CleanupExternalTensorsSerialization(
    ::ml_drift::SerializationWeightCache* shared_memory_serialization_cache) {
  // Cache was never initialized so there is nothing to clean up.
  if (shared_memory_serialization_cache == nullptr) {
    return absl::OkStatus();
  }
  // If the cache is ready for insert, we have called StartBuild() previously
  // and need to call StopBuild() before releasing the cache. If we have
  // inserted anything previously, this will cause the cache to be written to
  // disk.
  if (shared_memory_serialization_cache->IsReadyForInsert()) {
    RETURN_IF_ERROR(shared_memory_serialization_cache->StopBuild());
  }
  return absl::OkStatus();
}

}  // namespace litert::ml_drift
