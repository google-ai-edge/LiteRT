// Copyright 2025 Google LLC
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

#ifndef THIRD_PARTY_ODML_LITERT_EXTERNAL_WEIGHT_EXTERNAL_WEIGHT_LOADER_LITERT_H_
#define THIRD_PARTY_ODML_LITERT_EXTERNAL_WEIGHT_EXTERNAL_WEIGHT_LOADER_LITERT_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/internal/scoped_weight_source.h"
#include "tflite/schema/schema_generated.h"

class LiteRtEnvironmentT;

// This header defines the interface for loading external weights in LiteRT.
// External weights are model parameters (e.g., weights of a convolutional
// layer) that are not stored in the model flatbuffer itself, but in external
// files. This is useful for large models where the weights can be several
// gigabytes in size.
//
// The main components of this interface are:
// - `WeightLoader`: An abstract class that defines the interface for loading
//   external weights.
// - `WeightInfo`: A struct that contains information about a single external
//   weight tensor.
// - `WeightAccess`: A struct that provides access to the data of an external
//   weight tensor.
// - `CreateLiteRtWeightLoader`: A factory function that creates a
// `WeightLoader`
//   instance from a TFLite model flatbuffer.
//
// The typical usage of this interface is as follows:
// 1. Create a `WeightLoader` instance using `CreateLiteRtWeightLoader`.
// 2. Get the list of external weight tensors using `GetWeightInfo`.
// 3. For each external weight tensor, prepare access to its data using
//    `PrepareAccess`.
// 4. Get access to the data of an external weight tensor using
//    `GetExternalWeightByBuffer`.
// 5. Set the external weight tensor in the LiteRT model using
//    `SetExternalWeightByBuffer`.

// Weight loader contains functionality for loading external weights in LiteRT.
namespace weight_loader {

struct LiteRtTensorBufferDeleter {
  void operator()(LiteRtTensorBufferT* buffer) const {
    if (buffer) {
      LiteRtDestroyTensorBuffer(buffer);
    }
  }
};

using LiteRtTensorBufferPtr =
    std::unique_ptr<LiteRtTensorBufferT, LiteRtTensorBufferDeleter>;

struct WeightInfo {
  // The ID of the external buffer that contains the tensor data.
  uint32_t external_buffer_id;
  // The packing format of the tensor data.
  absl::string_view packing;
};

// A request to access the data of an external weight tensor.
struct WeightAccessRequest {
  // Whether to access the data on the CPU.
  bool cpu = true;
  // Whether to access the data on an OpenCL device.
  bool opencl = false;
};

struct WeightAccess;

// An abstract class that defines the interface for loading external weights.
class WeightLoader {
 public:
  virtual ~WeightLoader() = default;

  // Returns a list of all external weight tensors in the model.
  virtual absl::Span<const WeightInfo> GetWeightInfo() const = 0;

  // Prepares access to the data of the external weight tensors. This function
  // should be called before `GetExternalWeightByBuffer` or
  // `SetExternalWeightByBuffer`. The `request` parameter specifies how the
  // data should be accessed (e.g., on the CPU or on an OpenCL device). The
  // `env` parameter is the LiteRT environment.
  virtual absl::Status PrepareAccess(const WeightAccessRequest& request,
                                     LiteRtEnvironmentT* env) = 0;

  // Finds the `WeightInfo` for the external weight tensor with the given
  // buffer ID. Returns `nullptr` if no such tensor exists.
  virtual const WeightInfo* FindWeightInfoByBuffer(
      uint32_t external_buffer_id) const = 0;

  // Sets the external weight tensor with the given buffer ID. The `access`
  // parameter provides access to the tensor data.
  virtual absl::Status SetExternalWeightByBuffer(uint32_t external_buffer_id,
                                                 WeightAccess access) = 0;

  // Gets access to the data of the external weight tensor with the given
  // buffer ID. Returns `nullptr` if no such tensor exists.
  virtual const WeightAccess* GetExternalWeightByBuffer(
      uint32_t external_buffer_id) const = 0;
};

// Provides access to the data of an external weight tensor.
struct WeightAccess {
  WeightAccess();
  WeightAccess(const WeightAccess&) = delete;
  WeightAccess& operator=(const WeightAccess&) = delete;
  WeightAccess(WeightAccess&& other) noexcept;
  WeightAccess& operator=(WeightAccess&& other) noexcept;
  ~WeightAccess();

  // Resets the `WeightAccess` to its initial state.
  void Reset();

  // Sets the host buffer for the tensor data.
  void SetHostBuffer(LiteRtTensorBufferPtr buffer);
  // Sets the device buffer for the tensor data.
  void SetDeviceBuffer(LiteRtTensorBufferPtr buffer);

  // Returns the host buffer for the tensor data.
  LiteRtTensorBuffer GetHostBuffer() const { return host_tensor_buffer.get(); }
  // Returns the device buffer for the tensor data.
  LiteRtTensorBuffer GetDeviceBuffer() const {
    return device_tensor_buffer.get();
  }

 private:
  // TODO(b/456581477): Use a single tensor buffer for both host and device.
  LiteRtTensorBufferPtr host_tensor_buffer;
  LiteRtTensorBufferPtr device_tensor_buffer;
};

// Creates a `WeightLoader` instance from a TFLite model flatbuffer.
// The `flatbuffer` parameter is the TFLite model flatbuffer. The
// `model_directory` parameter is the directory where the external weight files
// are located. If `model_directory` is not specified, treat the group path
// as the absolute path.
std::unique_ptr<WeightLoader> CreateLiteRtWeightLoader(
    const tflite::Model* flatbuffer,
    std::optional<std::string> model_directory = std::nullopt,
    std::unique_ptr<litert::ScopedWeightSource> scoped_weight_source = nullptr);

}  // namespace weight_loader

#endif  // THIRD_PARTY_ODML_LITERT_EXTERNAL_WEIGHT_EXTERNAL_WEIGHT_LOADER_LITERT_H_
