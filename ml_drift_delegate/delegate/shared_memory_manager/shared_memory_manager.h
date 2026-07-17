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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/node_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/shared_const_tensor_map.h"
// clang-format off
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
// clang-format on
#include "third_party/odml/infra/ml_drift_delegate/util.h"
#include "tflite/core/c/common.h"

namespace ml_drift {

void MadviseData(void* ptr, size_t space);

// SharedConstTensor structure holds the cl::Tensor handle of the constant
// shared weight. For floating point tensors only .weights field is set. For
// int8 tensors, the .scale_global_tensor_id and .zero_point_global_tensor_id
// fields are set. Those ids are the fake global ids of the additionally created
// scale and zero_point tensors in the GraphFloat32, they should be used to look
// up the tensor in the buffer_id_to_spatial_tensor_.
// TODO: b/424473932 - Change struct to class.
struct SharedConstTensor {
  // The struct instance owns the weights. These weights are created directly
  // via `CreateTensor` call.
  std::unique_ptr<GpuSpatialTensor> weights;
  // The struct instance does not own the weights, but only have a view to the
  // Gpu tensor. These weights are created by Gpu weights conversion and owned
  // by an inference context.
  // TODO (linchan): b/378522761 - Have the struct always own the weights
  // tensor. We haven't done so yet because: The `external_weights` are created
  // by Gpu weights conversion and then owned by an inference context. As a
  // result, we need a way to export the ownership of Gpu tensors from the
  // inference context.
  GpuSpatialTensor* external_weights;
  // The `weight_sum_i` field stores the weight_sum_i value the first time it is
  // calculated, and is reused for all following executions. It is possible for
  // this field to be set even if weights_sum_i_global_tensor_id is not set.
  // This can happen if the first time the weights_sum_i is calculated, that
  // subgraph did not need the weights_sum_i tensor.
  Tensor<Linear, DataType::INT32> weights_sum_i;
  std::optional<uint32_t> scale_global_tensor_id;
  std::optional<uint32_t> zero_point_global_tensor_id;
  std::optional<uint32_t> weights_sum_i_global_tensor_id;

  GpuSpatialTensor* GetWeights() const {
    if (external_weights) {
      return external_weights;
    }
    return weights.get();
  }
};

// ValueIdToSharedTensorMap maps the GraphFloat32 value id to the
// GpuSpatialTensor.
using ValueIdToSharedTensorMap =
    absl::node_hash_map<ValueId, ml_drift::SharedConstTensor>;

// Manages creating the gpu tensors for constant weights sharing between
// different signatures of a single tflite model. Handles both floating point
// and quantized tensors.
class SharedMemoryManager {
 public:
  // Methods to process the data from a tflite tensor into the format needed by
  // the model.
  static TensorDescriptor GetInt8TensorDesc(
      const GpuInfo& gpu_info, const CreateGpuModelInfo& create_info,
      const OHWI& shape, const int8_t* data,
      bool is_weight_sum_i_required = false,
      Tensor<Linear, DataType::INT32>* weights_sum_i = nullptr);
  static TensorDescriptor GetInt4TensorDesc(
      const GpuInfo& gpu_info, const CreateGpuModelInfo& create_info,
      const OHWI& shape, const int8_t* data, size_t bytes,
      bool is_weight_sum_i_required = false,
      Tensor<Linear, DataType::INT32>* weights_sum_i = nullptr,
      bool experimental_int4_unpacking = false);
  static TensorDescriptor GetInt2TensorDesc(
      const GpuInfo& gpu_info, const CreateGpuModelInfo& create_info,
      const OHWI& shape, const int8_t* data, size_t bytes,
      bool is_weight_sum_i_required = false,
      Tensor<Linear, DataType::INT32>* weights_sum_i = nullptr,
      bool experimental_int2_unpacking = false);
  // Experimental version of GetInt4TensorDesc that rearranges in a 4 bit format
  // instead of a 8 bit format to save on memory and latency.
  //
  // TODO: b/423950292 - Remove this method once the experimental int4 unpacking
  // is enabled by default.
  static absl::StatusOr<TensorDescriptor> GetInt4TensorDescExperimental(
      const GpuInfo& gpu_info, const CreateGpuModelInfo& create_info,
      const OHWI& shape, const int8_t* data, size_t bytes,
      bool is_weight_sum_i_required = false,
      Tensor<Linear, DataType::INT32>* weights_sum_i = nullptr);

  // Experimental version of GetInt2TensorDesc that rearranges in a 2 bit format
  // instead of a 8 bit format to save on memory and latency.
  //
  // TODO: b/423950292 - Remove this method once the experimental int2 unpacking
  // is enabled by default.
  static absl::StatusOr<TensorDescriptor> GetInt2TensorDescExperimental(
      const GpuInfo& gpu_info, const CreateGpuModelInfo& create_info,
      const OHWI& shape, const int8_t* data, size_t bytes,
      bool is_weight_sum_i_required = false,
      Tensor<Linear, DataType::INT32>* weights_sum_i = nullptr);

  // Calculates the weights_sum_i tensor, for use with int8 convs.
  static absl::Status CalculateWeightsSumI(
      const TfLiteTensor& tflite_tensor, Value* shared_const_tensor,
      Tensor<Linear, DataType::INT32>* weights_sum_i);

  using CreateTensorFunc = std::function<absl::Status(
      TensorDescriptor&, size_t /*page_adjusted_offset*/,
      ml_drift_delegate::ReleaseDataCallback,
      std::unique_ptr<GpuSpatialTensor>& tensor)> const;
  using CreateTensorFromDeviceBufferFunc = std::function<absl::Status(
      const ::litert::ml_drift::SharedTfliteTensor&, const TensorDescriptor&,
      std::unique_ptr<GpuSpatialTensor>& tensor)>;
  using MaybeBindTensorDataFunc = std::function<absl::Status(
      const ::litert::ml_drift::SharedTfliteTensor&, TfLiteTensor&)>;
  using PackingLookupFunc =
      std::function<absl::StatusOr<std::string>(uint32_t global_id)>;
  using MaybeGetExternalBufferIdFunc =
      std::function<absl::StatusOr<uint32_t>(int shared_tflite_tensor_id)>;

  using DiscardTensorDataFunc = std::function<absl::Status(
      const ::litert::ml_drift::SharedTfliteTensor&)>;
  SharedMemoryManager(
      const GpuInfo& gpu_info, const CreateGpuModelInfo& create_info,
      GraphFloat32& graph, CreateTensorFunc create_tensor_func,
      TfLiteContext* context,
      ValueIdToSharedTensorMap& buffer_id_to_spatial_tensor,
      ValueIdToSharedTensorMap& quant_param_id_to_spatial_tensor,
      bool has_prepacked_tflite_tensors,
      SerializationWeightCache* serialization_cache,
      bool madvise_original_tensors, bool experimental_int4_unpacking = false,
      bool experimental_int2_unpacking = false,
      CreateTensorFromDeviceBufferFunc create_tensor_from_device_buffer_func =
          nullptr,
      MaybeBindTensorDataFunc maybe_bind_tensor_data_func = nullptr,
      PackingLookupFunc packing_lookup_func = nullptr,
      MaybeGetExternalBufferIdFunc maybe_get_external_buffer_id_func = nullptr,
      DiscardTensorDataFunc discard_tensor_data_func = nullptr);

  static constexpr uint32_t kInvalidExternalBufferId =
      std::numeric_limits<uint32_t>::max();

  static inline bool IsValidExternalBufferId(uint32_t id) {
    return id != 0 && id != kInvalidExternalBufferId;
  }

  // Holds the global id of the current SpatialTensor: either the id of the
  // source tflite tensor or newly created scale or zero point tensor.
  struct GlobalId {
    enum { kUnset, kSourceTensor, kParamTensor } type;
    uint32_t value;

    bool IsSourceId() const { return type == kSourceTensor; }
    bool IsParamId() const { return type == kParamTensor; }
    static GlobalId BuildSourceId(uint32_t value) {
      return GlobalId{kSourceTensor, value};
    }
    static GlobalId BuildParamId(uint32_t value) {
      return GlobalId{kParamTensor, value};
    }
  };

  // The RegisterExternalTensors function does a couple of things:
  // 1. Creates a Gpu tensor out of the given tflite_tensor, if no Gpu tensor
  // was created before for the tflite tensor. The Gpu tensor is shared across
  // different DelegateKernel instances, by being cached in the
  // SharedMemoryManager's maps for retrieving.
  // 2. Update the map of local to global id for the shared tensor. In this
  // context, local id - is the ValueId in GraphFloat32 and the global id in the
  // id taken from tflite model. Local id is used to bind the tensor as external
  // immutable, the global id is used to look up the tensor in the
  // SharedMemoryManager's cache. If the weight tensor is quantized, the
  // output map will also consist of the scale and zero_point tensors needed for
  // dequantization.
  //
  // The tensor to register will be added to `local_to_global_id_map`. Zero
  // point and scale tensors, if any, will be added to the map as well.
  //  * Key: the tensor's ID in GraphFloat32.
  //  * Value: the tensor's ID in SharedMemoryManager instance, and will be used
  // to get the Gpu tensor from the SharedMemoryManager instance.
  absl::Status RegisterExternalConstantTensors(
      const ValueId& shared_tensor_id,
      const ::litert::ml_drift::SharedTfliteTensor& shared_tflite_tensor,
      absl::flat_hash_map<ValueId, GlobalId>& local_to_global_id_map);

  // Get the Gpu tensor from the SharedMemoryManager's cache, which is shared
  // across different DelegateKernel instances and in the same Delegate
  // instance.
  absl::StatusOr<ml_drift::GpuSpatialTensor*> GetExternalConstantTensor(
      const GlobalId& global_id);

  void SetWeightsManager(
      std::shared_ptr<ml_drift::WeightsManager> weights_manager) {
    weights_manager_ = weights_manager;
  }
  WeightsManager* GetWeightsManager() { return weights_manager_.get(); }
  const absl::flat_hash_map<ValueId, uint32_t>&
  GetWeightIdToExternalBufferIdMap() const {
    return weight_id_to_external_buffer_id_;
  }

 private:
  absl::Status CreateSharedTensor(
      const ValueId& shared_tensor_id,
      const ::litert::ml_drift::SharedTfliteTensor& shared_tflite_tensor,
      std::unique_ptr<GpuSpatialTensor>& gpu_spatial_tensor,
      absl::flat_hash_map<ValueId, GlobalId>* external_tensors);

  // Tries to create a GpuSpatialTensor from a device buffer. This is only
  // supported on OpenCL and when the device buffer is already registered with
  // the OpenCL context.
  absl::Status TryCreateTensorFromDeviceBuffer(
      const ::litert::ml_drift::SharedTfliteTensor& shared_tflite_tensor,
      const TensorDescriptor& tensor_desc,
      std::unique_ptr<GpuSpatialTensor>& gpu_spatial_tensor);

  // Creates the quantized gpu tensor out of the given tflite_tensor. Adds
  // the scale and zero_point values to the graph and stores their ids in the
  // SharedConstTensor fields for the later retrieval. This logic is supposed to
  // be called for the first model signature.
  // NOTE: Only int8 quantization is currently supported.
  absl::Status CreateQuantizedTensorWithScaleAndZeroPoint(
      const ValueId& shared_tensor_id, uint32_t global_tensor_id,
      const TfLiteTensor& tflite_tensor,
      absl::flat_hash_map<ValueId, GlobalId>* external_tensors);

  // Retrieves the existing weights, scale and zero_point tensors from
  // buffer_id_to_spatial_tensor_ for the given shared_tensor_id. This logic is
  // supposed to be called for the second and all following model signatures.
  absl::Status RetrieveTensorWithScaleAndZeroPoint(
      const ValueId& shared_tensor_id, uint32_t global_tensor_id,
      const TfLiteTensor& tflite_tensor,
      absl::flat_hash_map<ValueId, GlobalId>* external_tensors);

  // Binds the data from the shared_tflite_tensor to the tensor if the
  // shared_tflite_tensor has a device buffer.
  absl::Status MaybeBindTensorData(
      const ::litert::ml_drift::SharedTfliteTensor& shared_tflite_tensor,
      TfLiteTensor& tensor);
  absl::Status DiscardTensorData(
      const ::litert::ml_drift::SharedTfliteTensor& shared_tflite_tensor);

  // If the tensor was prepacked and serialized previously in the
  // serialization_cache_, restore it from the serialized data.
  absl::Status TryRestoringSerializedTensor(uint32_t global_tensor_id,
                                            SharedConstTensor& shared_tensor);

  // If serialization is enabled, store the serialized tensor descriptor.
  absl::Status TryStoringSerializedTensor(uint32_t global_tensor_id,
                                          const TensorDescriptor& tensor_desc);

  // Creates quantized int8 weights tensor, applying weights rearrangement
  // required by inference.
  absl::Status CreateQuantizedInt8WeightsTensor(
      const ValueId& shared_tensor_id, uint32_t global_tensor_id,
      const TfLiteTensor& tflite_tensor, SharedConstTensor& shared_tensor,
      bool is_weight_sum_i_required = false,
      Tensor<Linear, DataType::INT32>* weights_sum_i = nullptr);

  // Creates quantized int4 weights tensor, applying weights rearrangement
  // required by inference.
  absl::Status CreateQuantizedInt4WeightsTensor(
      const ValueId& shared_tensor_id, uint32_t global_tensor_id,
      const TfLiteTensor& tflite_tensor, SharedConstTensor& shared_tensor,
      bool is_weight_sum_i_required = false,
      Tensor<Linear, DataType::INT32>* weights_sum_i = nullptr);

  // Creates quantized int2 weights tensor, applying weights rearrangement
  // required by inference.
  absl::Status CreateQuantizedInt2WeightsTensor(
      const ValueId& shared_tensor_id, uint32_t global_tensor_id,
      const TfLiteTensor& tflite_tensor, SharedConstTensor& shared_tensor,
      bool is_weight_sum_i_required = false,
      Tensor<Linear, DataType::INT32>* weights_sum_i = nullptr);

  // Creates scale and zero point tensors for affine quantized weights.
  absl::Status CreateAffineQuantizationParams(
      const ValueId& shared_tensor_id, uint32_t global_tensor_id,
      const TfLiteTensor& tflite_tensor,
      absl::flat_hash_map<ValueId, GlobalId>* external_tensors,
      bool is_weight_sum_i_required, Node* fc_node, DataType data_type,
      const OHWI& weights_shape);

  // Creates scale and zero point tensors for blockwise quantized weights.
  absl::Status CreateBlockwiseQuantizationParams(
      const ValueId& shared_tensor_id, uint32_t global_tensor_id,
      const TfLiteTensor& tflite_tensor,
      absl::flat_hash_map<ValueId, GlobalId>* external_tensors, Node* fc_node,
      DataType data_type);

  // Creates a linear tensor with a given shape, adds it to the graph and sets
  // the memory provided by the data pointer.
  //
  // If the TensorDescriptor is found by its global id in the serialization
  // cache, the descriptor will be retrieved from the cache instead of creating
  // a new one. In this case, the data pointer will be ignored.
  template <typename InputDataType>
  absl::StatusOr<ValueId> AddInputWithData(uint32_t global_tensor_id,
                                           const Linear& shape,
                                           const Node& consumer_node,
                                           const InputDataType* data,
                                           DataType data_type);
  template <typename InputDataType>
  absl::StatusOr<ValueId> AddScaleNodeWithData(uint32_t global_tensor_id,
                                               const Node& consumer_node,
                                               const InputDataType* data,
                                               DataType data_type,
                                               int num_channels,
                                               int num_blocks);
  // Adds a new value to the consumer_node. The tensor_id and shape parameters
  // are set as give, the data type is default.
  absl::StatusOr<ValueId> AddInputNode(uint32_t tensor_id, const BHWC& shape,
                                       const Node& consumer_node,
                                       DataType data_type);

  // Creates a tensor from a TfLiteTensor containing prepacked weight data.
  absl::StatusOr<std::unique_ptr<GpuSpatialTensor>>
  CreatePrepackedWeightsTensorFromTfliteTensor(
      const WeightsDescription& weights_desc, const OHWI& shape,
      const TfLiteTensor& tflite_tensor);

  uint32_t MaybeGetExternalBufferId(const ValueId& shared_tensor_id);

  const GpuInfo& gpu_info_;
  const CreateGpuModelInfo& create_info_;
  GraphFloat32& graph_;
  CreateTensorFunc create_tensor_func_;
  TfLiteContext* context_;
  ValueIdToSharedTensorMap& buffer_id_to_spatial_tensor_;
  ValueIdToSharedTensorMap& quant_param_id_to_spatial_tensor_;
  // Largest value id in the graph, gets incremented and used when scale and
  // zero_point tensors are added to the graph.
  uint32_t next_const_tensor_id_;
  // Default data type used for intermediate tensors.
  DataType data_type_;
  // Whether the tflite tensors in the model have already been prepacked into
  // the format the GPU expects.
  bool has_prepacked_tflite_tensors_;
  // If external tensor serialization is enabled, cache may contain serialized
  // tensors. These tensors should be used instead of the un-prepacked tflite
  // tensors in the model. If the cache is empty, the un-prepacked tflite
  // tensors will be packed and serialized to the cache.
  //
  // This a parallel and separate option to "has_prepacked_tflite_tensors_".
  // has_prepacked_tflite_tensors_ is more efficient in terms of disk space and
  // the memory usage and loading of the first initialization but it cannot
  // support redoing the weight rearrangement if ML Drift uses a different
  // packing format in the future. If both options are enabled, the serialized
  // cache will be used.
  SerializationWeightCache* serialization_cache_;

  // Tell the kernel that we do not expect to access this memory in the
  // near future. This allows the kernel to free resources associated
  // with this memory. This is NOT the same as unmapping the memory.
  // If we do attempt to access this later, the memory will be
  // repopulated from the original file on disk which would add latency.
  // Currently madvise will only apply to tensors that are greater than or equal
  // to 1MB.
  bool madvise_original_tensors_;

  // If true, use the experimental int4 unpacking method. This method rearranges
  // the weights in a 4 bit format instead of a 8 bit format to save on memory
  // and latency.
  //
  // TODO: b/423950292 - Remove this flag once the experimental int4 unpacking
  // is enabled by default.
  bool experimental_int4_unpacking_;

  // If true, use the experimental int2 unpacking method. This method rearranges
  // the weights in a 2 bit format instead of a 4 bit format to save on memory
  // and latency.
  //
  // TODO: b/423950292 - Remove this flag once the experimental int2 unpacking
  // is enabled by default.
  bool experimental_int2_unpacking_;

  MaybeBindTensorDataFunc maybe_bind_tensor_data_func_;
  PackingLookupFunc packing_lookup_func_;
  CreateTensorFromDeviceBufferFunc create_tensor_from_device_buffer_func_;
  MaybeGetExternalBufferIdFunc maybe_get_external_buffer_id_func_;

  absl::flat_hash_map<ValueId, int> local_to_tflite_tensor_id_;
  DiscardTensorDataFunc discard_tensor_data_func_;

  std::shared_ptr<WeightsManager> weights_manager_ = nullptr;
  absl::flat_hash_map<ValueId, uint32_t> weight_id_to_external_buffer_id_;
};

// Get the total shared memory size from the tensors in the map.
uint64_t GetSharedMemorySizeFromMap(
    const ::ml_drift::ValueIdToSharedTensorMap& map);

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_H_
