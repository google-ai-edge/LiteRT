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

#include "litert/cc/kernels/audio_frontend/irfft_kernel.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "tflite/c/c_api_types.h"
#include "tflite/schema/schema_generated.h"
#include "third_party/tflite_micro/tensorflow/lite/micro/audio_frontend/src/complex.h"
#include "third_party/tflite_micro/tensorflow/lite/micro/audio_frontend/src/irfft.h"

namespace litert {
namespace audio_frontend {

namespace {
TfLiteType ConvertElementTypeToTfLiteType(ElementType element_type) {
  switch (element_type) {
    case ElementType::Float32:
      return kTfLiteFloat32;
    case ElementType::Int16:
      return kTfLiteInt16;
    case ElementType::Int32:
      return kTfLiteInt32;
    default:
      return kTfLiteNoType;
  }
}
}  // namespace

Expected<void> IrfftKernel::Init(const void* init_data, size_t init_data_size) {
  if (init_data == nullptr || init_data_size == 0) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "IrfftKernel: init_data is null or empty.");
  }

  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(init_data);
  const flexbuffers::Map& m =
      flexbuffers::GetRoot(buffer_t, init_data_size).AsMap();
  fft_length_ = m["fft_length"].AsInt32();
  auto tensor_type = static_cast<tflite::TensorType>(m["T"].AsInt32());

  size_t state_size = 0;
  switch (tensor_type) {
    case tflite::TensorType_FLOAT32:
      fft_type_ = kTfLiteFloat32;
      state_size = IrfftFloatGetNeededMemory(fft_length_);
      break;
    case tflite::TensorType_INT16:
      fft_type_ = kTfLiteInt16;
      state_size = IrfftInt16GetNeededMemory(fft_length_);
      break;
    case tflite::TensorType_INT32:
      fft_type_ = kTfLiteInt32;
      state_size = IrfftInt32GetNeededMemory(fft_length_);
      break;
    default:
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "IrfftKernel: Unsupported tensor type.");
  }

  state_ = new int8_t[state_size];
  if (state_ == nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "IrfftKernel: Failed to allocate state buffer.");
  }

  switch (fft_type_) {
    case kTfLiteFloat32:
      IrfftFloatInit(fft_length_, state_, state_size);
      break;
    case kTfLiteInt16:
      IrfftInt16Init(fft_length_, state_, state_size);
      break;
    case kTfLiteInt32:
      IrfftInt32Init(fft_length_, state_, state_size);
      break;
    default:
      // Should not happen due to checks above.
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "IrfftKernel: Unexpected fft_type_ after init.");
  }

  return {};
}

Expected<void> IrfftKernel::Destroy() {
  if (state_ != nullptr) {
    delete[] reinterpret_cast<int8_t*>(state_);
    state_ = nullptr;
  }
  return {};
}

Expected<void> IrfftKernel::GetOutputLayouts(
    const std::vector<Layout>& input_layouts,
    std::vector<Layout>& output_layouts) {
  if (input_layouts.size() != 1 || output_layouts.size() != 1) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "IrfftKernel: Expected 1 input and 1 output.");
  }

  const Layout& input_layout = input_layouts[0];
  if (input_layout.Dimensions().empty()) {
    return Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        "IrfftKernel: Input tensor must have at least 1 dimension.");
  }

  if (input_layout.Dimensions().back() != fft_length_ + 2) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "IrfftKernel: Input tensor last dimension must be "
                      "fft_length + 2.");
  }

  // The output last dimension is fft_length.
  std::vector<int> output_dims(input_layout.Dimensions().begin(),
                               input_layout.Dimensions().end());
  output_dims.back() = fft_length_;

  output_layouts[0] =
      Layout(Dimensions(output_dims.begin(), output_dims.end()));
  return {};
}

Expected<void> IrfftKernel::Run(const std::vector<TensorBuffer>& inputs,
                                std::vector<TensorBuffer>& outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "IrfftKernel: Expected 1 input and 1 output.");
  }

  const TensorBuffer& input = inputs[0];
  TensorBuffer& output = outputs[0];

  LITERT_ASSIGN_OR_RETURN(auto input_tensor_type, input.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto output_tensor_type, output.TensorType());

  TfLiteType input_tflite_type =
      ConvertElementTypeToTfLiteType(input_tensor_type.ElementType());
  TfLiteType output_tflite_type =
      ConvertElementTypeToTfLiteType(output_tensor_type.ElementType());

  if (input_tflite_type != fft_type_ || output_tflite_type != fft_type_) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "IrfftKernel: Tensor type mismatch.");
  }

  const Layout& input_layout = input_tensor_type.Layout();
  const Layout& output_layout = output_tensor_type.Layout();

  // Input last dimension is fft_length / 2.
  int input_length = input_layout.Dimensions().back();
  // Output last dimension is fft_length.
  int output_length = output_layout.Dimensions().back();

  LITERT_ASSIGN_OR_RETURN(size_t input_size, input_layout.NumElements());

  // Assuming batching, iterate through the input.
  size_t num_batches = input_size / input_length;

  for (size_t i = 0; i < num_batches; ++i) {
    size_t input_offset = i * input_length;
    size_t output_offset = i * output_length;

    switch (fft_type_) {
      case kTfLiteFloat32: {
        LITERT_ASSIGN_OR_RETURN(auto input_lock,
                                TensorBufferScopedLock::Create<const float>(
                                    input, TensorBuffer::LockMode::kRead));
        LITERT_ASSIGN_OR_RETURN(auto output_lock,
                                TensorBufferScopedLock::Create<float>(
                                    output, TensorBuffer::LockMode::kWrite));
        IrfftFloatApply(state_,
                        reinterpret_cast<const Complex<float>*>(
                            input_lock.second + input_offset),
                        output_lock.second + output_offset);
        break;
      }
      case kTfLiteInt16: {
        LITERT_ASSIGN_OR_RETURN(auto input_lock,
                                TensorBufferScopedLock::Create<const int16_t>(
                                    input, TensorBuffer::LockMode::kRead));
        LITERT_ASSIGN_OR_RETURN(auto output_lock,
                                TensorBufferScopedLock::Create<int16_t>(
                                    output, TensorBuffer::LockMode::kWrite));
        IrfftInt16Apply(state_,
                        reinterpret_cast<const Complex<int16_t>*>(
                            input_lock.second + input_offset),
                        output_lock.second + output_offset);
        break;
      }
      case kTfLiteInt32: {
        LITERT_ASSIGN_OR_RETURN(auto input_lock,
                                TensorBufferScopedLock::Create<const int32_t>(
                                    input, TensorBuffer::LockMode::kRead));
        LITERT_ASSIGN_OR_RETURN(auto output_lock,
                                TensorBufferScopedLock::Create<int32_t>(
                                    output, TensorBuffer::LockMode::kWrite));
        IrfftInt32Apply(state_,
                        reinterpret_cast<const Complex<int32_t>*>(
                            input_lock.second + input_offset),
                        output_lock.second + output_offset);
        break;
      }
      default:
        return Unexpected(kLiteRtStatusErrorInvalidArgument,
                          "IrfftKernel: Unsupported fft_type_.");
    }
  }

  return {};
}

}  // namespace audio_frontend
}  // namespace litert
