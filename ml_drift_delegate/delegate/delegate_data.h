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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_DELEGATE_DATA_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_DELEGATE_DATA_H_

#include <memory>
#include <string>
#include <vector>

#include "ml_drift/common/executor.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "third_party/odml/infra/ml_drift_delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "third_party/odml/infra/ml_drift_delegate/shared_memory_manager.h"
#include "ml_drift_delegate/delegate/delegate_options.h"
#include "ml_drift_delegate/delegate/incrementable_blocking_counter.h"
#include "ml_drift_delegate/tflite/shared_const_tensor_map.h"

namespace weight_loader {
class WeightLoader;
}  // namespace weight_loader

namespace litert::ml_drift {

// Forward declaration to avoid circular dependency.
class GpuBackend;

// Data structure to hold the delegate data.
struct MlDriftDelegateData {
  std::unique_ptr<MlDriftDelegateOptions> options;

  // GPU backend is shared between different instances of the delegate
  // kernel, where each one gets created for every tflite model's subgraph.
  std::unique_ptr<GpuBackend> backend;

  // Shared GPU backend is used to share GPU resources between different
  // instances of the LiteRT Accelerator.
  std::shared_ptr<GpuBackend> shared_backend;

  // Maps global id (in scope of the whole tflite model) of the tflite tensor to
  // the SpatialTensor that we want to share between subgraphs.
  ::ml_drift::ValueIdToSharedTensorMap buffer_id_to_spatial_tensor;

  // Maps ids of scale and zero point tensors to SpatialTensors, which get
  // created to support runtime quantization of the shared tflite tensor.
  ::ml_drift::ValueIdToSharedTensorMap quant_param_id_to_spatial_tensor;

  // List of External Mutable and Immutable Tensors.
  // The Tensors are accessed by the BufferHandle which is actually the index in
  // this vector. Valid indices of these two vectors should not overlap: When
  // one has value, the other should be nullptr at the same index.
  // Mutable tensors are used when binding tensors after model initialization is
  // required. Otherwise, prefer immutable tensors for better performance.
  std::vector<std::unique_ptr<::ml_drift::GpuSpatialTensor>>
      external_mutable_tensors;
  // Immutable tensors are registered at mode initialization, and cannot be
  // rebind after that.
  std::vector<std::unique_ptr<::ml_drift::GpuSpatialTensor>>
      external_immutable_tensors;

  // Copy of user provided options->serialization_dir.
  std::string serialization_dir;

  // Copy of user provided options->model_token.
  std::string model_token;

  // Precision used by the delegate.
  ::ml_drift::CalculationsPrecision calculation_precision;

  // WebGPU only.
  //
  // The map which stores the information about the shared constant tensors. It
  // needs to be shared between different instances of the delegate kernel.
  SharedConstTensorsMap shared_const_tensors;

  // Executor used to upload weight data to the GPU.
  std::shared_ptr<::ml_drift::Executor> upload_executor;

  // The number of delegate kernels converting weights. All delegates using this
  // data must wait for all the conversion to complete before proceeding.
  // Only used when upload_executor is not null and weights are prepared on Gpu.
  std::unique_ptr<IncrementableBlockingCounter> weights_conversion_counter;

  // Non-owning pointer to the shared LiteRT WeightLoader.
  weight_loader::WeightLoader* weight_loader = nullptr;

  // Shared weight cache for all subgraphs of the model.
  std::unique_ptr<::ml_drift::SerializationWeightCache> serialization_cache;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_DELEGATE_KERNEL_H_
